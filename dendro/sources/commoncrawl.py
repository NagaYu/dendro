"""Common Crawl index: a second, institutionally separate web archive.

Common Crawl matters here for exactly one reason: it is **not the Internet
Archive**.  Its crawls are run by a different organisation, stored in a
different place (public S3 WARC files with fixed release dates), and indexed by
different software.  Compromising both is a strictly harder problem than
compromising either, and :func:`dendro.witness.combine_failure_probability`
multiplies the two failure probabilities precisely because of that.

A practical constraint shapes the implementation: there are 120+ monthly
collections and no aggregated endpoint, so querying all of them per document is
absurd.  We query a small, deterministic subset chosen around the era of
interest.  The choice is deterministic so that the committed fixture cache stays
valid, and the subset is cached per collection so a whole-corpus annotation run
pays for each collection once (**cost**, benchmark axis 5).
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from typing import Any, Optional

from ..cache import HttpClient
from ..types import Witness, WitnessKind, to_utc
from ..witness import Target, WitnessSource, register_source

COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"


@register_source
class CommonCrawlSource(WitnessSource):
    """Capture witnesses from the Common Crawl CDX indexes."""

    source_id = "commoncrawl"
    operator = "commoncrawl-org"
    kind = WitnessKind.SNAPSHOT
    reliability = 0.99
    forgeability = 1e-3

    #: Each collection is a separate HTTP request against a rate-limited public
    #: index, so this number multiplies directly into wall-clock for a corpus run:
    #: four collections over sixty documents is 240 requests, and at the polite
    #: 0.5 req/s in :data:`dendro.cache.DEFAULT_RATES` that alone is eight minutes.
    #: Two is enough to establish presence; raise it when you care more about
    #: recall on obscure URLs than about finishing quickly.
    def __init__(self, max_collections: int = 2, max_captures: int = 4) -> None:
        self.max_collections = int(max_collections)
        self.max_captures = int(max_captures)

    def supports(self, target: Target) -> bool:
        return bool(target.url)

    # -- collections -------------------------------------------------------
    def _collections(self, client: HttpClient) -> list[dict[str, Any]]:
        got = client.try_fetch(COLLINFO_URL, kind="json")
        body = got.get("body") if got else None
        return [c for c in body if isinstance(c, dict) and c.get("cdx-api")] if isinstance(body, list) else []

    def _pick(self, collections: list[dict[str, Any]], around: Optional[_dt.datetime]) -> list[dict[str, Any]]:
        """Deterministically choose which monthly indexes to ask.

        One collection at the era of interest, one a year later (so a document
        that appeared *after* a claimed date still gets found), and the two most
        recent.  Deterministic selection keeps the fixture cache reproducible.
        """
        if not collections:
            return []
        def start(c: dict[str, Any]) -> _dt.datetime:
            try:
                return to_utc(c.get("from") or "1990-01-01")
            except ValueError:
                return to_utc("1990-01-01")

        ordered = sorted(collections, key=start)
        picks: list[dict[str, Any]] = []
        if around is not None:
            for offset in (0, 1):
                want = around.replace(tzinfo=around.tzinfo) + _dt.timedelta(days=365 * offset)
                nearest = min(ordered, key=lambda c: abs((start(c) - want).total_seconds()))
                if nearest not in picks:
                    picks.append(nearest)
        for c in reversed(ordered):
            if len(picks) >= self.max_collections:
                break
            if c not in picks:
                picks.append(c)
        return picks[: self.max_collections]

    # -- witnesses ---------------------------------------------------------
    def collect(self, target: Target, client: HttpClient) -> list[Witness]:
        if not target.url:
            return []
        around = target.claimed_date or to_utc("2019-01-01")
        out: list[Witness] = []
        for coll in self._pick(self._collections(client), around):
            for rec in _cc_query(client, coll["cdx-api"], {"url": target.url, "output": "json", "limit": "3"}):
                ts = rec.get("timestamp")
                if not ts:
                    continue
                try:
                    observed = to_utc(ts)
                except ValueError:
                    continue
                out.append(
                    Witness(
                        source_id=self.source_id,
                        operator=self.operator,
                        kind=self.kind,
                        observed_at=observed,
                        target=target.url,
                        reliability=self.reliability,
                        forgeability=self.forgeability,
                        content_digest=rec.get("digest"),
                        url=f"https://index.commoncrawl.org/{coll.get('id', '')}?url={target.url}",
                        raw={"collection": coll.get("id"), "record": rec},
                    )
                )
                if len(out) >= self.max_captures:
                    return out
        return out

    # -- coverage ----------------------------------------------------------
    def coverage(self, target: Target, when: _dt.datetime, client: HttpClient) -> float:
        """Crawl depth on this host in the collection nearest ``when``.

        Same logic as the Wayback probe, on a different operator's data.  Two
        independent coverage measurements are what make a backdate log-LR grow
        past the "interesting" threshold; one alone rarely should.
        """
        host = target.host
        if not host:
            return 0.0
        colls = self._pick(self._collections(client), to_utc(when))
        if not colls:
            return 0.0
        best = 0.0
        for coll in colls[:2]:
            recs = _cc_query(
                client, coll["cdx-api"], {"url": f"{host}/*", "output": "json", "limit": "200", "fl": "urlkey"}
            )
            n = len({r.get("urlkey") for r in recs if r.get("urlkey")})
            if n:
                best = max(best, min(0.90, 1.0 - math.exp(-n / 40.0)))
        return float(best)


def _cc_query(client: HttpClient, api: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Common Crawl returns newline-delimited JSON, and 404 for "no captures".

    Both are normal answers, not errors, so both map to a list (possibly empty)
    rather than an exception.
    """
    got = client.try_fetch(api, params, kind="text")
    if not got:
        return []
    if got.get("status") == 404:
        return []
    body = got.get("body")
    if not isinstance(body, str):
        return []
    out: list[dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and "message" not in rec:
            out.append(rec)
    return out
