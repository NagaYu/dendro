"""Scholarly registries: arXiv and Crossref.

Registration witnesses are qualitatively different from crawl witnesses.  A
crawl says "someone's robot fetched these bytes"; a registration says "an
institution minted a permanent, publicly-mirrored identifier for this work on
this date".  Registrations are harder to forge and easier to verify by a third
party, which is why arXiv carries the lowest ``forgeability`` of any source here.

Crossref exposes two dates and they deserve different trust, which this module
takes seriously:

``created``
    when Crossref itself first received the deposit.  Crossref's clock, not the
    publisher's.  Genuine independent evidence -- ``forgeability`` 2e-3.
``issued``
    the publication date the publisher asserts.  A cooperative publisher can
    restate it, so it is closer to self-assertion -- ``forgeability`` 5e-2.

Collapsing those two into one "publication date" is the sort of shortcut that
makes a provenance system quietly unsound, so they are emitted as separate
witnesses of different kinds and the consensus estimator weighs them apart.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any, Optional
from xml.etree import ElementTree as ET

from ..cache import HttpClient
from ..types import Witness, WitnessKind, to_utc
from ..witness import Target, WitnessSource, register_source

ARXIV_API = "http://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
_ATOM = "{http://www.w3.org/2005/Atom}"


@register_source
class ArxivSource(WitnessSource):
    """arXiv submission dates, by identifier or by exact-title search."""

    source_id = "arxiv"
    operator = "arxiv-cornell"
    kind = WitnessKind.REGISTRATION
    reliability = 0.999
    forgeability = 5e-4

    def supports(self, target: Target) -> bool:
        return bool(target.arxiv_id or target.title)

    def collect(self, target: Target, client: HttpClient) -> list[Witness]:
        """Emit the *v1* submission date, which is the tightest honest bound.

        arXiv reports ``published`` for v1 and ``updated`` for the latest
        revision.  Only ``published`` bounds first existence; using ``updated``
        would silently loosen every bound on a revised paper.
        """
        if target.arxiv_id:
            params = {"id_list": target.arxiv_id, "max_results": 1}
        elif target.title:
            params = {"search_query": f'ti:"{_clean(target.title)}"', "max_results": 3}
        else:
            return []

        got = client.try_fetch(ARXIV_API, params, kind="text")
        if not got or got.get("status") != 200 or not isinstance(got.get("body"), str):
            return []
        try:
            root = ET.fromstring(got["body"])
        except ET.ParseError:
            return []

        out: list[Witness] = []
        for entry in root.findall(f"{_ATOM}entry"):
            published = entry.findtext(f"{_ATOM}published")
            entry_id = entry.findtext(f"{_ATOM}id") or ""
            title = _clean(entry.findtext(f"{_ATOM}title") or "")
            if not published:
                continue
            # A title search can return near-misses; require a real title match
            # before treating the hit as evidence about *this* document.
            if not target.arxiv_id and target.title and not _title_matches(target.title, title):
                continue
            out.append(
                Witness(
                    source_id=self.source_id,
                    operator=self.operator,
                    kind=self.kind,
                    observed_at=to_utc(published),
                    target=target.arxiv_id or title,
                    reliability=self.reliability,
                    forgeability=self.forgeability,
                    cached=bool(got.get("cached")),
                    url=entry_id or None,
                    raw={"title": title, "id": entry_id},
                )
            )
        return out

    def coverage(self, target: Target, when: _dt.datetime, client: HttpClient) -> float:
        """arXiv covers arXiv, completely, since 1991 -- and nothing else.

        Returning a high number for arXiv papers and zero otherwise is what keeps
        the backdate detector from accusing a blog post of being forged merely
        because arXiv has never heard of it.
        """
        if not target.arxiv_id:
            return 0.0
        return 0.98 if to_utc(when).year >= 1992 else 0.0


@register_source
class CrossrefSource(WitnessSource):
    """DOI deposit and publication records."""

    source_id = "crossref"
    operator = "crossref"
    kind = WitnessKind.REGISTRATION
    reliability = 0.99
    forgeability = 2e-3

    def supports(self, target: Target) -> bool:
        return bool(target.doi or target.title)

    def collect(self, target: Target, client: HttpClient) -> list[Witness]:
        if target.doi:
            got = client.try_fetch(f"{CROSSREF_API}/{target.doi}", kind="json")
            items = [got["body"]["message"]] if _ok(got) and "message" in got["body"] else []
        elif target.title:
            got = client.try_fetch(
                CROSSREF_API,
                {"query.bibliographic": _clean(target.title), "rows": 3, "select": "DOI,title,created,issued"},
                kind="json",
            )
            items = got["body"]["message"].get("items", []) if _ok(got) and "message" in got["body"] else []
        else:
            return []

        out: list[Witness] = []
        for item in items:
            titles = item.get("title") or []
            title = _clean(titles[0]) if titles else ""
            if not target.doi and target.title and not _title_matches(target.title, title):
                continue
            doi = item.get("DOI") or target.doi or title

            created = (item.get("created") or {}).get("date-time")
            if created:
                out.append(
                    Witness(
                        source_id=self.source_id,
                        operator=self.operator,
                        kind=WitnessKind.REGISTRATION,
                        observed_at=to_utc(created),
                        target=doi,
                        reliability=self.reliability,
                        forgeability=self.forgeability,
                        url=f"https://doi.org/{doi}" if doi else None,
                        raw={"field": "created", "title": title},
                    )
                )
            issued = _date_parts((item.get("issued") or {}).get("date-parts"))
            if issued:
                out.append(
                    Witness(
                        source_id=self.source_id,
                        operator=self.operator,
                        kind=WitnessKind.PUBLICATION,
                        observed_at=issued,
                        target=doi,
                        reliability=0.95,
                        # Publisher-asserted: much closer to self-assertion than
                        # the deposit timestamp, and priced accordingly.
                        forgeability=5e-2,
                        url=f"https://doi.org/{doi}" if doi else None,
                        raw={"field": "issued", "title": title},
                    )
                )
        return out

    def coverage(self, target: Target, when: _dt.datetime, client: HttpClient) -> float:
        if not target.doi:
            return 0.0
        return 0.95 if to_utc(when).year >= 2000 else 0.4


# --------------------------------------------------------------------------- helpers
def _ok(got: Optional[dict[str, Any]]) -> bool:
    return bool(got and got.get("status") == 200 and isinstance(got.get("body"), dict))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _title_matches(a: str, b: str, threshold: float = 0.75) -> bool:
    """Token-overlap gate on title-search hits.

    A fuzzy registry search that returns a *different* paper would attach that
    paper's date to this document -- a false early bound, the one error class
    this system must not make.  The gate is deliberately strict.
    """
    ta = {w for w in re.findall(r"[a-z0-9]+", a.casefold()) if len(w) > 2}
    tb = {w for w in re.findall(r"[a-z0-9]+", b.casefold()) if len(w) > 2}
    if not ta or not tb:
        return False
    return len(ta & tb) / max(1, min(len(ta), len(tb))) >= threshold


def _date_parts(parts: Any) -> Optional[_dt.datetime]:
    if not parts or not isinstance(parts, list) or not parts[0]:
        return None
    p = list(parts[0]) + [1, 1]
    try:
        return _dt.datetime(int(p[0]), int(p[1]), int(p[2]), tzinfo=_dt.timezone.utc)
    except (ValueError, TypeError):
        return None
