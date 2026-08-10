"""The document's own claim about its date -- the thing to be *contradicted*.

This source produces witnesses of kind ``SELF_ASSERTED``, which
:meth:`dendro.witness.WitnessCollector.consensus` refuses to let tighten a
bound.  So why collect them at all?

Because the backdating attack has to make a claim somewhere, and you cannot
detect a lie you never read.  A synthetic 2026 document with
``<meta name="date" content="2019-03-11">`` is invisible to every statistical
detector -- they consume prose, not metadata -- and is caught here the moment the
claim is put next to what independent archives that were *demonstrably looking*
actually saw.

Extraction covers the realistic surfaces: HTML meta tags, ``<time>`` elements,
JSON-LD, YAML front-matter, and the "Published on ..." line that most CMSs
render into the body.  Every hit is emitted separately with its provenance in
``raw['field']``, so a document that claims 2019 in its front-matter and 2026 in
its JSON-LD produces two conflicting self-assertions -- itself a signal.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from typing import Iterator, Optional

from ..cache import HttpClient
from ..types import Witness, WitnessKind, to_utc
from ..witness import Target, WitnessSource, register_source

_META_RE = re.compile(
    r"""<meta[^>]+(?:name|property|itemprop)\s*=\s*["']?"""
    r"""(date|dc\.date[\w.]*|article:published_time|article:modified_time|"""
    r"""og:updated_time|pubdate|publish[-_]?date|datePublished|created)["']?[^>]*"""
    r"""content\s*=\s*["']([^"']{4,40})["']""",
    re.IGNORECASE,
)
_META_REV_RE = re.compile(
    r"""<meta[^>]+content\s*=\s*["']([^"']{4,40})["'][^>]*(?:name|property|itemprop)\s*=\s*["']?"""
    r"""(date|article:published_time|datePublished|pubdate)["']?""",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"""<time[^>]+datetime\s*=\s*["']([^"']{4,40})["']""", re.IGNORECASE)
_JSONLD_RE = re.compile(
    r"""<script[^>]+type\s*=\s*["']application/ld\+json["'][^>]*>(.*?)</script>""",
    re.IGNORECASE | re.DOTALL,
)
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FM_DATE_RE = re.compile(r"^\s*(date|published|created|pubdate)\s*:\s*['\"]?([^'\"\n]{4,40})", re.IGNORECASE | re.MULTILINE)
_PROSE_RE = re.compile(
    r"(?:published|posted|updated|written|created|first\s+appeared|last\s+modified)"
    r"(?:\s+(?:on|at|in))?\s*:?\s*"
    r"(\d{4}-\d{2}-\d{2}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_ISO_LIKE = re.compile(r"\b(19[89]\d|20[0-4]\d)-\d{2}-\d{2}\b")


@register_source
class SelfAssertedSource(WitnessSource):
    """Reads dates the document asserts about itself.  Never evidence, always checked."""

    source_id = "self_asserted"
    operator = "document-itself"
    kind = WitnessKind.SELF_ASSERTED
    reliability = 0.6
    forgeability = 0.98

    def __init__(self, max_claims: int = 6) -> None:
        self.max_claims = int(max_claims)

    def supports(self, target: Target) -> bool:
        return bool(target.text or target.claimed_date)

    def collect(self, target: Target, client: HttpClient) -> list[Witness]:
        out: list[Witness] = []
        seen: set[tuple[str, str]] = set()

        if target.claimed_date is not None:
            out.append(self._witness(target, target.claimed_date, "target.claimed_date"))
            seen.add(("target.claimed_date", target.claimed_date.date().isoformat()))

        for field, when in extract_claimed_dates(target.text or ""):
            key = (field, when.date().isoformat())
            if key in seen:
                continue
            seen.add(key)
            out.append(self._witness(target, when, field))
            if len(out) >= self.max_claims:
                break
        return out

    def _witness(self, target: Target, when: _dt.datetime, field: str) -> Witness:
        return Witness(
            source_id=self.source_id,
            operator=self.operator,
            kind=WitnessKind.SELF_ASSERTED,
            observed_at=when,
            target=target.url or target.doc_id,
            reliability=self.reliability,
            forgeability=self.forgeability,
            coverage=0.0,
            url=target.url,
            raw={"field": field},
        )

    def coverage(self, target: Target, when: _dt.datetime, client: HttpClient) -> float:
        """Always zero.  A document's silence about itself proves nothing."""
        return 0.0


def extract_claimed_dates(text: str) -> list[tuple[str, _dt.datetime]]:
    """Every date the text asserts about itself, tagged with where it came from.

    Exposed as a module function because the corpus builder needs it to *plant*
    forged dates in the adversarial split, and the Space needs it to show a user
    what their document is claiming before Dendro contradicts it.
    """
    return list(_iter_claims(text or ""))


def _iter_claims(text: str) -> Iterator[tuple[str, _dt.datetime]]:
    head = text[:20000]

    for m in _META_RE.finditer(head):
        when = _parse(m.group(2))
        if when:
            yield f"meta:{m.group(1).lower()}", when
    for m in _META_REV_RE.finditer(head):
        when = _parse(m.group(1))
        if when:
            yield f"meta:{m.group(2).lower()}", when
    for m in _TIME_RE.finditer(head):
        when = _parse(m.group(1))
        if when:
            yield "time.datetime", when

    for m in _JSONLD_RE.finditer(head):
        try:
            blob = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        for key, value in _walk_jsonld(blob):
            when = _parse(str(value))
            if when:
                yield f"jsonld:{key}", when

    fm = _FRONTMATTER_RE.match(text or "")
    if fm:
        for m in _FM_DATE_RE.finditer(fm.group(1)):
            when = _parse(m.group(2))
            if when:
                yield f"frontmatter:{m.group(1).lower()}", when

    for m in _PROSE_RE.finditer(head):
        when = _parse(m.group(1))
        if when:
            yield "prose", when


def _walk_jsonld(node, depth: int = 0):
    if depth > 4:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("datePublished", "dateCreated", "uploadDate", "dateModified") and isinstance(v, str):
                yield k, v
            else:
                yield from _walk_jsonld(v, depth + 1)
    elif isinstance(node, list):
        for item in node[:20]:
            yield from _walk_jsonld(item, depth + 1)


def _parse(raw: str) -> Optional[_dt.datetime]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        when = to_utc(raw)
    except ValueError:
        m = _ISO_LIKE.search(raw)
        if not m:
            return None
        try:
            when = to_utc(m.group(0))
        except ValueError:
            return None
    # Reject nonsense that would otherwise poison the backdate comparison.
    if not (1990 <= when.year <= 2100):
        return None
    return when
