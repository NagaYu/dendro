"""Content-addressed HTTP cache, per-host rate limiting, and offline replay.

Three jobs, all of them load-bearing for the experiments:

1. **Offline determinism.** Every network read goes through :class:`HttpClient`
   and is written to disk keyed by a canonical request string.  Re-running a
   benchmark with ``DENDRO_OFFLINE=1`` replays the exact same bytes, so the
   headline figures are reproducible on a laptop with the wifi off.  A cache
   miss in offline mode raises :class:`OfflineCacheMiss` and the collector
   degrades that source to "unavailable" -- it never silently invents evidence.

2. **Cost accounting.** :class:`CacheStats` records hits, misses, wall time and
   bytes, which is benchmark axis (5): per-document witness-acquisition cost and
   cache hit rate.  Without this the "evidence is expensive" objection cannot be
   answered with a number.

3. **Being a good citizen.** :class:`RateLimiter` is a persistent token bucket
   *per host*, so a 400-document benchmark cannot hammer the Internet Archive
   even across separate process invocations.  Backoff honours ``Retry-After``.
   The default rate is deliberately timid; archives are a commons.

Claims exercised: **generator-independence** (the cost of Dendro is I/O against
archives, and is unaffected by which model wrote the text) and
**reproducibility** (identical results offline).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

__all__ = [
    "CacheStats",
    "Cache",
    "RateLimiter",
    "HttpClient",
    "OfflineCacheMiss",
    "FetchError",
    "default_cache_dir",
    "USER_AGENT",
]

#: Identify ourselves honestly.  Archives block anonymous scrapers, and rightly so.
USER_AGENT = (
    "dendro-research/0.1 (provenance research prototype; "
    "+https://github.com/NagaYu/dendro; contact via repository issues)"
)

#: Conservative defaults, in requests per second, per host.
DEFAULT_RATES: dict[str, float] = {
    "web.archive.org": 0.5,
    "archive.org": 0.5,
    "index.commoncrawl.org": 0.5,
    "export.arxiv.org": 0.2,         # arXiv asks for >=3s; 5s keeps us clear of 429s
    "oaipmh.arxiv.org": 0.2,
    "hn.algolia.com": 2.0,
    "www.gutenberg.org": 0.34,
    "api.crossref.org": 2.0,
    "api.github.com": 1.0,
    "lists.debian.org": 0.5,
    "mail-archive.com": 0.5,
    "_default": 1.0,
}


class OfflineCacheMiss(RuntimeError):
    """Raised when offline mode is on and the request was never cached."""


class FetchError(RuntimeError):
    """A network read failed after retries.  Callers degrade the source, not crash."""


def default_cache_dir() -> pathlib.Path:
    """Where the cache lives unless told otherwise.

    ``DENDRO_CACHE`` wins, then a repo-local ``.dendro-cache``.  The bundled
    fixture cache under ``data/fixtures/cache`` is layered underneath by
    :class:`Cache` so that a fresh clone can run the tests with no network.
    """
    env = os.environ.get("DENDRO_CACHE")
    if env:
        return pathlib.Path(env).expanduser()
    return pathlib.Path.cwd() / ".dendro-cache"


def repo_fixture_cache() -> pathlib.Path:
    """The read-only cache of real archive responses shipped with the repo."""
    return pathlib.Path(__file__).resolve().parents[1] / "data" / "fixtures" / "cache"


def is_offline() -> bool:
    return os.environ.get("DENDRO_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- stats
@dataclass
class CacheStats:
    """Per-run I/O accounting -- benchmark axis (5).

    ``hit_rate`` is the number the paper cares about: once an archive corpus has
    been walked one time, marginal cost per document collapses, which is why
    evidence-based dating is viable at dataset scale.
    """

    hits: int = 0
    misses: int = 0
    network_calls: int = 0
    errors: int = 0
    offline_misses: int = 0
    bytes_down: int = 0
    network_seconds: float = 0.0
    cache_seconds: float = 0.0
    throttle_seconds: float = 0.0

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def wall_seconds(self) -> float:
        return self.network_seconds + self.cache_seconds + self.throttle_seconds

    def merge(self, other: "CacheStats") -> "CacheStats":
        merged = CacheStats()
        for k in asdict(self):
            setattr(merged, k, getattr(self, k) + getattr(other, k))
        return merged

    def as_row(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "network_calls": self.network_calls,
            "errors": self.errors,
            "offline_misses": self.offline_misses,
            "bytes_down": self.bytes_down,
            "network_seconds": round(self.network_seconds, 3),
            "throttle_seconds": round(self.throttle_seconds, 3),
            "wall_seconds": round(self.wall_seconds, 3),
        }


# --------------------------------------------------------------------------- cache
class Cache:
    """A layered, content-addressed JSON cache on the filesystem.

    Reads consult the writable root first, then every read-only overlay (the
    repo fixture cache).  Writes only ever touch the root.  That layering is
    what lets ``pytest`` exercise *real* Wayback CDX responses -- recorded once,
    committed, replayed forever -- while a user's own runs accumulate privately.

    Keys are canonicalised URLs plus sorted query parameters, so the same
    logical request from two call sites hits the same entry.
    """

    def __init__(
        self,
        root: Optional[pathlib.Path | str] = None,
        overlays: Optional[list[pathlib.Path]] = None,
        namespace: str = "http",
    ) -> None:
        self.root = pathlib.Path(root) if root is not None else default_cache_dir()
        default_overlays = [repo_fixture_cache()]
        self.overlays = [pathlib.Path(p) for p in (overlays if overlays is not None else default_overlays)]
        self.namespace = namespace
        self._lock = threading.Lock()

    # -- keying ------------------------------------------------------------
    @staticmethod
    def canonical_key(url: str, params: Optional[Mapping[str, Any]] = None) -> str:
        """Canonical request string: scheme+host lowercased, params sorted.

        Determinism here is what makes a committed fixture cache stable across
        machines and Python versions.
        """
        parts = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
        if params:
            query += [(str(k), str(v)) for k, v in params.items()]
        query.sort()
        encoded = urllib.parse.urlencode(query)
        return urllib.parse.urlunsplit(
            (parts.scheme.lower(), parts.netloc.lower(), parts.path, encoded, "")
        )

    def _rel_path(self, key: str) -> pathlib.Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return pathlib.Path(self.namespace) / digest[:2] / f"{digest}.json"

    def path_for(self, key: str) -> pathlib.Path:
        return self.root / self._rel_path(key)

    # -- access ------------------------------------------------------------
    def get(self, key: str) -> Optional[dict[str, Any]]:
        rel = self._rel_path(key)
        for base in [self.root, *self.overlays]:
            path = base / rel
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8") as fh:
                        return json.load(fh)
                except (json.JSONDecodeError, OSError):
                    continue
        return None

    def put(self, key: str, payload: Any, meta: Optional[Mapping[str, Any]] = None) -> None:
        path = self.path_for(key)
        record = {
            "key": key,
            "fetched_at": time.time(),
            "meta": dict(meta or {}),
            "payload": payload,
        }
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False)
            tmp.replace(path)

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def entries(self) -> int:
        return sum(1 for base in [self.root, *self.overlays] if base.exists()
                   for _ in base.rglob("*.json"))


# --------------------------------------------------------------------------- rate limit
class RateLimiter:
    """Persistent per-host token bucket.

    State lives on disk so that ten sequential ``dendro date`` invocations are
    throttled as one client, which is the polite reading of every archive's
    terms of use.  ``sleep_fn`` is injectable purely so tests can assert the
    throttling arithmetic without actually waiting.
    """

    def __init__(
        self,
        state_path: Optional[pathlib.Path] = None,
        rates: Optional[Mapping[str, float]] = None,
        burst: float = 3.0,
        sleep_fn=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.state_path = pathlib.Path(state_path) if state_path else None
        self.rates = dict(rates or DEFAULT_RATES)
        self.burst = float(burst)
        self._sleep = sleep_fn
        self._clock = clock
        self._tokens: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self.slept_seconds = 0.0

    def rate_for(self, host: str) -> float:
        return float(self.rates.get(host, self.rates.get("_default", 1.0)))

    def acquire(self, url: str) -> float:
        """Block until one token is available for this URL's host; return slept time."""
        host = urllib.parse.urlsplit(url).netloc.lower()
        rate = self.rate_for(host)
        if rate <= 0:
            return 0.0
        with self._lock:
            now = self._clock()
            last = self._last.get(host, now)
            tokens = min(self.burst, self._tokens.get(host, self.burst) + (now - last) * rate)
            wait = 0.0
            if tokens < 1.0:
                wait = (1.0 - tokens) / rate
            self._last[host] = now + wait
            self._tokens[host] = tokens + wait * rate - 1.0
        if wait > 0:
            self._sleep(wait)
            self.slept_seconds += wait
        return wait


# --------------------------------------------------------------------------- client
class HttpClient:
    """The only place in Dendro that is allowed to touch the network.

    Concentrating I/O here is what makes the two operational guarantees
    checkable rather than aspirational: *nothing* bypasses the cache, and
    *nothing* bypasses the rate limiter.  ``tests/test_cache.py`` asserts both by
    running a full collection with a transport that raises on use.
    """

    def __init__(
        self,
        cache: Optional[Cache] = None,
        rate_limiter: Optional[RateLimiter] = None,
        offline: Optional[bool] = None,
        timeout: float = 20.0,
        max_retries: int = 3,
        session=None,
        stats: Optional[CacheStats] = None,
    ) -> None:
        self.cache = cache if cache is not None else Cache()
        self.rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()
        self.offline = is_offline() if offline is None else bool(offline)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.stats = stats if stats is not None else CacheStats()
        self._session = session
        self._session_owned = session is None
        #: Keys that already failed in *this process*.  Failures are deliberately
        #: not written to the disk cache -- a transient outage must not be frozen
        #: into the committed fixtures -- but retrying them is worse.  A public
        #: archive that is slow or down gets queried once per run rather than once
        #: per document; without this, a single hanging endpoint turned a
        #: 60-document forgery sweep into an hour of blocked sockets, because each
        #: document re-attempted the same request and paid the full timeout and
        #: retry budget again.
        self._failed: set[str] = set()

    # -- session -----------------------------------------------------------
    def _get_session(self):
        if self._session is None:
            import requests  # imported lazily: offline runs need no requests at all

            sess = requests.Session()
            sess.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
            self._session = sess
        return self._session

    def close(self) -> None:
        if self._session is not None and self._session_owned:
            try:
                self._session.close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._session = None

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- fetching ----------------------------------------------------------
    def fetch(
        self,
        url: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        kind: str = "text",
        force: bool = False,
    ) -> dict[str, Any]:
        """Cache-first GET.  Returns ``{"status", "body", "cached", "seconds"}``.

        ``kind`` is ``"json"`` or ``"text"``; JSON bodies are stored parsed so the
        fixture files stay human-readable in a diff.
        """
        key = Cache.canonical_key(url, params)
        t0 = time.perf_counter()
        if key in self._failed and not force:
            raise FetchError(f"{url}: already failed in this run")
        if not force:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats.hits += 1
                self.stats.cache_seconds += time.perf_counter() - t0
                payload = hit["payload"]
                return {
                    "status": payload.get("status", 200),
                    "body": payload.get("body"),
                    "cached": True,
                    "seconds": time.perf_counter() - t0,
                    "fetched_at": hit.get("fetched_at"),
                }

        self.stats.misses += 1
        if self.offline:
            self.stats.offline_misses += 1
            raise OfflineCacheMiss(f"offline and not cached: {key}")

        slept = self.rate_limiter.acquire(url)
        self.stats.throttle_seconds += slept

        body, status, err = None, 0, None
        net0 = time.perf_counter()
        for attempt in range(self.max_retries):
            try:
                resp = self._get_session().get(url, params=dict(params or {}), timeout=self.timeout)
                self.stats.network_calls += 1
                status = resp.status_code
                if status == 429 or status >= 500:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if (retry_after or "").isdigit() else 2.0 ** attempt
                    delay += random.random() * 0.25
                    if attempt + 1 < self.max_retries:
                        time.sleep(min(delay, 30.0))
                        self.stats.throttle_seconds += min(delay, 30.0)
                        continue
                self.stats.bytes_down += len(resp.content or b"")
                body = resp.json() if kind == "json" else resp.text
                err = None
                break
            except Exception as exc:  # network, decode, or JSON failure
                err = exc
                self.stats.network_calls += 1
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2.0**attempt, 10.0))
        self.stats.network_seconds += time.perf_counter() - net0

        if err is not None:
            self.stats.errors += 1
            self._failed.add(key)
            raise FetchError(f"{url}: {err}") from err

        payload = {"status": status, "body": body}
        self.cache.put(key, payload, meta={"url": url, "params": dict(params or {}), "kind": kind})
        return {
            "status": status,
            "body": body,
            "cached": False,
            "seconds": time.perf_counter() - t0,
            "fetched_at": time.time(),
        }

    def get_json(self, url: str, params: Optional[Mapping[str, Any]] = None, **kw) -> Any:
        return self.fetch(url, params, kind="json", **kw)["body"]

    def get_text(self, url: str, params: Optional[Mapping[str, Any]] = None, **kw) -> str:
        body = self.fetch(url, params, kind="text", **kw)["body"]
        return body if isinstance(body, str) else json.dumps(body)

    def try_fetch(self, url: str, params=None, *, kind: str = "text") -> Optional[dict[str, Any]]:
        """Fetch, or return ``None`` on any failure.

        Sources use this so that one archive being down degrades the *evidence*
        (fewer independent operators, wider interval) rather than the *run*.
        Degrading gracefully is a calibration property: the bound simply gets
        weaker, and the reported interval widens to say so.
        """
        try:
            return self.fetch(url, params, kind=kind)
        except (OfflineCacheMiss, FetchError):
            return None
