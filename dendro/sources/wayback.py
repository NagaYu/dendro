"""Internet Archive Wayback Machine, via the public CDX server.

The Wayback Machine is the strongest single witness available for web content,
for a reason worth stating precisely: **its API lets you create a capture now,
but not a capture dated 2019.**  An adversary who wants a document to look old
cannot obtain an old Wayback record without compromising the archive's storage.
That asymmetry is why ``forgeability`` here is 1e-3 while a git commit date --
which is literally an environment variable -- is 5e-1.

The coverage probe is the other half.  Asking "did Wayback see *this page* in
2019?" is useless on its own; asking "how deeply was Wayback crawling *this
host* in 2019, and did it nonetheless never see this page?" turns silence into
evidence.  That is what makes the backdate detector sound rather than
trigger-happy (**adversarial-robustness**).
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Optional

from ..cache import HttpClient
from ..types import Witness, WitnessKind, to_utc
from ..witness import Target, WitnessSource, register_source

CDX_URL = "https://web.archive.org/cdx/search/cdx"


@register_source
class WaybackSource(WitnessSource):
    """Snapshot witnesses and host-level coverage from the CDX index."""

    source_id = "wayback"
    operator = "internet-archive"
    kind = WitnessKind.SNAPSHOT
    reliability = 0.995
    forgeability = 1e-3

    #: NOTE: ``coverage_limit`` is a query parameter, so it is part of the cache
    #: key.  Changing it silently invalidates every committed coverage probe and
    #: sends a whole benchmark run back to the network against an API where a
    #: single domain query can take 90 seconds.  Treat it as frozen unless you
    #: intend to re-record the fixtures.
    def __init__(self, max_captures: int = 6, coverage_limit: int = 400) -> None:
        self.max_captures = int(max_captures)
        self.coverage_limit = int(coverage_limit)

    def supports(self, target: Target) -> bool:
        return bool(target.url)

    # -- witnesses ---------------------------------------------------------
    def collect(self, target: Target, client: HttpClient) -> list[Witness]:
        """Earliest distinct captures of the exact URL.

        ``collapse=digest`` asks the server for content-distinct captures, so the
        witnesses we keep are *different observations of the content*, not a
        thousand identical daily crawls.  We keep the earliest few because only
        the earliest matters for an upper bound -- the rest are kept purely so
        that the operator's within-group accidental-error term has something to
        multiply.
        """
        if not target.url:
            return []
        served = {"cached": False}
        rows = _cdx(
            client,
            {
                "url": target.url,
                "output": "json",
                "fl": "timestamp,original,digest,statuscode,mimetype",
                "collapse": "digest",
                "limit": str(self.max_captures * 4),
            },
            served,
        )
        out: list[Witness] = []
        seen: set[str] = set()
        for row in rows:
            ts, original, digest, status = (row + ["", "", "", ""])[:4]
            if status and status not in ("200", "-"):
                continue
            if digest in seen:
                continue
            seen.add(digest)
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
                    content_digest=digest or None,
                    cached=served["cached"],
                    url=f"https://web.archive.org/web/{ts}/{original}",
                    raw={"cdx": row},
                )
            )
            if len(out) >= self.max_captures:
                break
        return out

    # -- coverage ----------------------------------------------------------
    def coverage(self, target: Target, when: _dt.datetime, client: HttpClient) -> float:
        """How thoroughly was this host crawled around ``when``?

        Measured as the number of *distinct URLs* the archive holds for the host
        in a +-1 year window.  A host with thousands of distinct archived paths
        was being crawled deeply, so a page that existed then and was never
        captured is genuinely surprising.  A host with three archived paths tells
        us nothing, and this returns a number near zero so no accusation follows.

        The saturating form ``1 - exp(-n/tau)`` is a heuristic, and it is applied
        to a *lower* bound on crawl depth (the CDX ``limit`` truncates), so it
        errs toward claiming less coverage than there is -- the safe direction for
        an accusation.
        """
        host = target.host
        if not host:
            return 0.0
        when = to_utc(when)
        rows = _cdx(
            client,
            {
                "url": host,
                "matchType": "domain",
                "from": str(when.year - 1),
                "to": str(when.year + 1),
                "collapse": "urlkey",
                "fl": "urlkey",
                "limit": str(self.coverage_limit),
                "output": "json",
            },
        )
        n_distinct = len({r[0] for r in rows if r and r[0]})
        if n_distinct == 0:
            return 0.0
        # Saturates well before the query limit, so lowering ``coverage_limit`` for
        # speed does not silently change the answer for well-covered hosts: 150
        # distinct paths already gives ~0.92, and hosts that matter are far above
        # the knee.
        return float(min(0.95, 1.0 - math.exp(-n_distinct / 60.0)))


def _cdx(client: HttpClient, params: dict[str, Any], flag: Optional[dict] = None) -> list[list[str]]:
    """Fetch a CDX query and drop the header row.

    Returns ``[]`` on any failure, including offline cache misses: a source that
    cannot answer contributes no evidence, which widens the interval rather than
    corrupting it.
    """
    got = client.try_fetch(CDX_URL, params, kind="json")
    if flag is not None and got:
        flag["cached"] = bool(got.get("cached"))
    if not got or got.get("status") != 200:
        return []
    body = got.get("body")
    if not isinstance(body, list) or len(body) < 2:
        return []
    header = body[0]
    if isinstance(header, list) and header and header[0] in ("timestamp", "urlkey", "original"):
        return [r for r in body[1:] if isinstance(r, list)]
    return [r for r in body if isinstance(r, list)]
