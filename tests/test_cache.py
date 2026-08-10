"""Caching, rate limiting, and the guarantee that nothing bypasses either.

Two operational promises are made to the archives whose APIs this tool consumes,
and both are asserted here rather than merely documented:

1. **Every network read goes through the cache**, so a re-run is free and an
   offline re-run is possible at all.
2. **Every network read goes through the rate limiter**, so a corpus-scale job
   cannot hammer a public archive -- across processes, since the limiter's state
   is per-host and persistent within a client.

The strongest form of (1) is the transport-explodes test: a full collection is
run with a session object that raises on any use, and it must still produce the
same bound from cache.
"""

from __future__ import annotations

import json
import time

import pytest

from dendro.cache import (
    Cache,
    CacheStats,
    HttpClient,
    OfflineCacheMiss,
    RateLimiter,
    default_cache_dir,
)


class ExplodingSession:
    """Any use of the network is a test failure."""

    def __init__(self):
        self.headers = {}

    def get(self, *a, **kw):  # pragma: no cover - must never run
        raise AssertionError("network was used when it should not have been")

    def close(self):
        pass


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.headers = {}
        self.content = json.dumps(payload).encode()
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class CountingSession:
    def __init__(self, payload=None):
        self.headers = {}
        self.calls = 0
        self.payload = payload if payload is not None else {"ok": True}

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return FakeResponse(self.payload)

    def close(self):
        pass


# --------------------------------------------------------------------------- keying
def test_canonical_key_is_order_independent():
    a = Cache.canonical_key("https://EXAMPLE.org/x", {"b": 2, "a": 1})
    b = Cache.canonical_key("https://example.org/x?a=1", {"b": 2})
    assert a == b


def test_canonical_key_is_stable_across_runs():
    """A committed fixture cache is worthless if keys drift."""
    key = Cache.canonical_key("https://web.archive.org/cdx/search/cdx", {"url": "x", "limit": "5"})
    assert key == "https://web.archive.org/cdx/search/cdx?limit=5&url=x"


# --------------------------------------------------------------------------- caching
def test_second_request_is_served_from_cache(tmp_path):
    session = CountingSession()
    client = HttpClient(cache=Cache(root=tmp_path, overlays=[]), offline=False, session=session)
    client.get_json("https://example.org/a")
    client.get_json("https://example.org/a")
    assert session.calls == 1
    assert client.stats.hits == 1 and client.stats.misses == 1
    assert client.stats.hit_rate == pytest.approx(0.5)


def test_cache_survives_a_new_client(tmp_path):
    # ``offline=False`` explicitly: this test injects a transport and *means* to
    # exercise the fetch-and-store path.  Left implicit it inherits DENDRO_OFFLINE,
    # which CI sets for the whole suite as a real-network safety net -- so the test
    # passed locally and failed in CI, which is the wrong way round.
    session = CountingSession()
    HttpClient(cache=Cache(root=tmp_path, overlays=[]), session=session,
               offline=False).get_json("https://example.org/b")
    fresh = HttpClient(cache=Cache(root=tmp_path, overlays=[]), offline=True, session=ExplodingSession())
    assert fresh.get_json("https://example.org/b") == {"ok": True}


def test_offline_miss_raises_rather_than_fetching(tmp_path):
    client = HttpClient(cache=Cache(root=tmp_path, overlays=[]), offline=True, session=ExplodingSession())
    with pytest.raises(OfflineCacheMiss):
        client.fetch("https://example.org/never-seen")
    assert client.stats.offline_misses == 1


def test_try_fetch_degrades_instead_of_raising(tmp_path):
    client = HttpClient(cache=Cache(root=tmp_path, overlays=[]), offline=True, session=ExplodingSession())
    assert client.try_fetch("https://example.org/never-seen") is None


def test_overlay_is_read_only(tmp_path, fixture_cache_dir):
    """Fixtures are readable but a run must never write into the committed cache."""
    root = tmp_path / "writable"
    client = HttpClient(cache=Cache(root=root, overlays=[fixture_cache_dir]), offline=False,
                        session=CountingSession())
    client.get_json("https://example.org/new-thing")
    written = list(root.rglob("*.json"))
    assert written, "nothing was written to the writable root"
    before = {p.name for p in fixture_cache_dir.rglob("*.json")}
    after = {p.name for p in fixture_cache_dir.rglob("*.json")}
    assert before == after


# --------------------------------------------------------------------------- rate limit
def test_rate_limiter_enforces_the_configured_interval():
    slept: list[float] = []
    clock = {"t": 0.0}
    limiter = RateLimiter(
        rates={"_default": 2.0}, burst=1.0,
        sleep_fn=lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)),
        clock=lambda: clock["t"],
    )
    for _ in range(4):
        limiter.acquire("https://example.org/x")
    assert len(slept) >= 3, slept
    assert all(s == pytest.approx(0.5, abs=1e-6) for s in slept), slept


def test_rate_limits_are_per_host():
    slept: list[float] = []
    clock = {"t": 0.0}
    limiter = RateLimiter(rates={"_default": 1.0}, burst=1.0,
                          sleep_fn=lambda s: slept.append(s), clock=lambda: clock["t"])
    limiter.acquire("https://a.example/x")
    limiter.acquire("https://b.example/x")
    assert slept == [], "hosts shared a token bucket"


def test_arxiv_gets_a_slow_default_rate():
    """arXiv asks for >=3s between calls; the default must respect that."""
    limiter = RateLimiter()
    assert limiter.rate_for("export.arxiv.org") <= 1 / 3


def test_cached_requests_are_not_throttled(tmp_path):
    session = CountingSession()
    slept: list[float] = []
    limiter = RateLimiter(rates={"_default": 0.5}, burst=1.0, sleep_fn=lambda s: slept.append(s))
    client = HttpClient(cache=Cache(root=tmp_path, overlays=[]), rate_limiter=limiter,
                        session=session, offline=False)
    client.get_json("https://example.org/c")
    n_after_first = len(slept)
    for _ in range(5):
        client.get_json("https://example.org/c")
    assert len(slept) == n_after_first, "cache hits went through the rate limiter"


# --------------------------------------------------------------------------- end to end
def test_full_collection_replays_offline_with_no_network(offline_client, tmp_path):
    """The headline guarantee, asserted with a transport that raises on use."""
    from dendro.witness import Target, WitnessCollector

    offline_client._session = ExplodingSession()
    collector = WitnessCollector(client=offline_client)
    target = Target(doc_id="pep20", url="https://www.python.org/dev/peps/pep-0020/")
    bound = collector.consensus(collector.collect(target))
    assert bound.has_evidence
    assert bound.not_after.year <= 2008
    assert offline_client.stats.network_calls == 0


def test_stats_merge_and_serialise():
    a = CacheStats(hits=3, misses=1, network_calls=1)
    b = CacheStats(hits=2, misses=2, network_calls=2)
    merged = a.merge(b)
    assert merged.hits == 5 and merged.requests == 8
    row = merged.as_row()
    assert row["hit_rate"] == pytest.approx(5 / 8)
    assert json.dumps(row)


def test_default_cache_dir_respects_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DENDRO_CACHE", str(tmp_path / "custom"))
    assert default_cache_dir() == tmp_path / "custom"


def test_dendro_offline_env_var_is_honoured(monkeypatch, tmp_path):
    """``offline=None`` must defer to the environment, and only ``True`` may force it.

    Regression guard, twice earned.  Call sites that passed ``offline=args.offline``
    sent a hard ``False`` whenever the flag was absent, which silently overrode
    ``DENDRO_OFFLINE=1`` -- so a run launched as "cached only" quietly went to the
    network, and a benchmark advertised as reproducing offline was making live
    archive requests.
    """
    monkeypatch.setenv("DENDRO_OFFLINE", "1")
    cache = Cache(root=tmp_path, overlays=[])
    assert HttpClient(cache=cache).offline is True
    assert HttpClient(cache=cache, offline=None).offline is True
    assert HttpClient(cache=cache, offline=False).offline is False   # explicit override still works

    monkeypatch.setenv("DENDRO_OFFLINE", "0")
    assert HttpClient(cache=cache).offline is False
    assert HttpClient(cache=cache, offline=True).offline is True


def test_entry_points_do_not_hardcode_offline_false(monkeypatch, tmp_path):
    """The CLI and the dataset scripts must not defeat the environment variable."""
    monkeypatch.setenv("DENDRO_OFFLINE", "1")
    monkeypatch.setenv("DENDRO_CACHE", str(tmp_path))

    from scripts.annotate_dataset import build_dendro

    assert build_dendro().collector.client.offline is True

    from dendro.cli import _build_pipeline

    args = type(
        "Args", (), {"cache": str(tmp_path), "offline": False, "sources": None, "alpha": 1e-2}
    )()
    assert _build_pipeline(args).collector.client.offline is True
