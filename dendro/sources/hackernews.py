"""Hacker News via the public Algolia index: a third-party posting archive.

This is the "public mailing-list / forum archive" slot in the source line-up.
Hacker News is used rather than a mailman archive for a boring but decisive
reason -- it has a stable, documented JSON API, so the parser is not a screen
scraper that breaks when a template changes.

What it contributes is a **different failure mode**.  Web crawlers observe pages
they choose to visit; a posting archive observes what *third parties chose to
share*, on infrastructure run by a company with no relationship to either the
Internet Archive or Common Crawl.  For a URL that was discussed publicly, this
adds a fourth independent operator to the product in
:func:`dendro.witness.combine_failure_probability`, and the failure probability
drops by another three orders of magnitude.

Search is fuzzy, so every hit is re-checked against the target URL client-side.
A witness attached to the wrong document is worse than no witness at all.
"""

from __future__ import annotations

import datetime as _dt
import urllib.parse
from typing import Any, Optional

from ..cache import HttpClient
from ..types import Witness, WitnessKind, to_utc
from ..witness import Target, WitnessSource, register_source

ALGOLIA_API = "https://hn.algolia.com/api/v1/search"


@register_source
class HackerNewsSource(WitnessSource):
    """Submission timestamps for URLs that were posted publicly."""

    source_id = "hackernews"
    operator = "ycombinator"
    kind = WitnessKind.POSTING
    reliability = 0.99
    forgeability = 3e-3

    def __init__(self, max_hits: int = 20) -> None:
        self.max_hits = int(max_hits)

    def supports(self, target: Target) -> bool:
        return bool(target.url)

    def collect(self, target: Target, client: HttpClient) -> list[Witness]:
        """Earliest submission of this exact URL.

        Only the earliest matters for an upper bound; later resubmissions are
        dropped rather than padded into the witness list, because inflating a
        single operator's witness count buys nothing under the consensus model
        and would only make the evidence table misleading to a reader.
        """
        if not target.url:
            return []
        got = client.try_fetch(
            ALGOLIA_API,
            {"query": _search_key(target.url), "tags": "story", "hitsPerPage": self.max_hits},
            kind="json",
        )
        if not got or got.get("status") != 200 or not isinstance(got.get("body"), dict):
            return []

        want = _normalise_url(target.url)
        best: Optional[dict[str, Any]] = None
        for hit in got["body"].get("hits", []):
            hit_url = hit.get("url")
            if not hit_url or _normalise_url(hit_url) != want:
                continue
            created = hit.get("created_at")
            if not created:
                continue
            if best is None or created < best["created_at"]:
                best = hit
        if best is None:
            return []
        object_id = best.get("objectID")
        return [
            Witness(
                source_id=self.source_id,
                operator=self.operator,
                kind=self.kind,
                observed_at=to_utc(best["created_at"]),
                target=target.url,
                reliability=self.reliability,
                forgeability=self.forgeability,
                cached=bool(got.get("cached")),
                url=f"https://news.ycombinator.com/item?id={object_id}" if object_id else None,
                raw={"title": best.get("title"), "objectID": object_id},
            )
        ]

    def coverage(self, target: Target, when: _dt.datetime, client: HttpClient) -> float:
        """Deliberately zero.

        Most documents that exist are never posted to Hacker News, so silence
        here carries no information and must not contribute to a backdate
        accusation.  Being explicit about which sources may and may not fuel an
        accusation is what keeps the detector from manufacturing suspicion.
        """
        return 0.0


def _normalise_url(url: str) -> str:
    """Compare URLs modulo scheme, ``www.``, trailing slash, and tracking params."""
    parts = urllib.parse.urlsplit(url.strip())
    host = (parts.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = (parts.path or "/").rstrip("/") or "/"
    query = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parts.query)
        if not k.lower().startswith(("utm_", "fbclid", "gclid", "ref"))
    ]
    query.sort()
    return urllib.parse.urlunsplit(("", host, path, urllib.parse.urlencode(query), ""))


def _search_key(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    host = (parts.netloc or "").lower().removeprefix("www.")
    return f"{host}{parts.path}".rstrip("/")
