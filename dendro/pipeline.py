"""The end-to-end path: text or URL in, calibrated verdict out.

Everything the CLI and the Gradio Space do goes through :class:`Dendro`, so
there is exactly one place where the four subsystems are wired together and
exactly one definition of what "dating a document" means:

    1. **Witness** the target -- ask every independent archive what it saw and when.
    2. **Fingerprint** the text -- format-blind, so a re-render is the same document.
    3. **Align** it to the archive layer -- so a rewrite inherits its source's date.
    4. **Propagate** -- turn the evidence into a probability with an interval, and
       check the document's own claims against what the archives actually hold.

Steps 1 and 3 are deliberately independent evidence paths.  A URL that no archive
crawled can still be dated through an ancestor; a document whose text matches
nothing can still be dated by a crawl of its URL.  Either path alone produces a
bound; having both is what makes the system degrade gracefully instead of
falling off a cliff (**adversarial-robustness**).
"""

from __future__ import annotations

import datetime as _dt
import pathlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from .alignment import Aligner, ArchiveLayer
from .cache import CacheStats, HttpClient
from .fingerprint import ReflowFingerprint
from .propagate import Calibrator, ContaminationPropagator, DerivationEdge, PropagationConfig
from .sources.selfasserted import extract_claimed_dates
from .types import ConsensusBound, Verdict, to_utc
from .witness import ConsensusConfig, Target, WitnessCollector, WitnessSource

__all__ = ["Dendro", "date_document"]


class Dendro:
    """Facade over witness collection, fingerprinting, alignment and propagation."""

    def __init__(
        self,
        collector: Optional[WitnessCollector] = None,
        archive: Optional[ArchiveLayer] = None,
        aligner: Optional[Aligner] = None,
        propagator: Optional[ContaminationPropagator] = None,
        sources: Optional[Sequence[WitnessSource]] = None,
        client: Optional[HttpClient] = None,
        consensus_config: Optional[ConsensusConfig] = None,
    ) -> None:
        self.fingerprinter = ReflowFingerprint()
        self.archive = archive if archive is not None else ArchiveLayer(self.fingerprinter)
        self.aligner = aligner or Aligner(self.fingerprinter)
        self.collector = collector or WitnessCollector(
            sources=sources, client=client, config=consensus_config
        )
        self.propagator = propagator or ContaminationPropagator()

    @property
    def stats(self) -> CacheStats:
        return self.collector.stats

    # -- single document ---------------------------------------------------
    def date(
        self,
        text: Optional[str] = None,
        url: Optional[str] = None,
        doc_id: Optional[str] = None,
        path: Optional[str] = None,
        claimed_date: Optional[_dt.datetime] = None,
        probe_coverage: bool = True,
    ) -> Verdict:
        """Date one document from every evidence path available for it.

        ``probe_coverage`` controls whether the extra archive queries that make
        backdate detection possible are issued.  It costs roughly one additional
        request per source and is the difference between "we found no 2019
        record" and "archives that crawled 340 pages on this host in 2019 found
        no record" -- so it is on by default, and benchmark axis (5) reports what
        it costs.
        """
        target = Target(
            doc_id=doc_id or url or path or "document",
            url=url,
            text=text,
            path=path,
            claimed_date=claimed_date,
            title=_guess_title(text),
        )
        if target.claimed_date is None and text:
            claims = extract_claimed_dates(text)
            if claims:
                target.claimed_date = min(when for _, when in claims)

        witnesses = self.collector.collect(target)
        bound = self.collector.consensus(witnesses)

        ancestor = None
        if text and len(self.archive):
            fp = self.fingerprinter.fingerprint(target.doc_id, text)
            ancestor = self.aligner.oldest_ancestor(fp, self.archive)

        coverage: Mapping[str, float] = {}
        if probe_coverage and target.claimed_date is not None:
            coverage = self.collector.coverage_profile(target, target.claimed_date)

        flags = self.collector.detect_inconsistencies(target, witnesses, bound, coverage)
        return self.propagator.verdict(target.doc_id, bound, ancestor, flags)

    # -- many documents ----------------------------------------------------
    def date_many(
        self,
        documents: Iterable[Mapping],
        edges: Iterable[DerivationEdge] = (),
        probe_coverage: bool = False,
    ) -> dict[str, Verdict]:
        """Date a collection, then apply the graph constraint.

        ``probe_coverage`` defaults to *off* here: at corpus scale the extra
        request per document per source dominates the cost, and the graph pass is
        usually run over material whose claimed dates are already known to be
        absent.  ``dendro report`` exposes the switch.
        """
        verdicts: dict[str, Verdict] = {}
        for doc in documents:
            d = dict(doc)
            doc_id = d.get("doc_id") or d.get("id") or d.get("url") or f"doc{len(verdicts)}"
            verdicts[doc_id] = self.date(
                text=d.get("text"),
                url=d.get("url"),
                doc_id=doc_id,
                path=d.get("path"),
                claimed_date=d.get("claimed_date"),
                probe_coverage=probe_coverage,
            )
        return self.propagator.propagate(verdicts, edges)

    # -- archive layer -----------------------------------------------------
    def index_archive(self, entries: Iterable[Mapping], fit_null: bool = True) -> "Dendro":
        """Load documents with witnessed dates into the alignment layer.

        Enables the second evidence path, which is what delivers
        **adversarial-robustness** against paraphrase: without an archive layer a
        rewritten document has only its own (absent) witnesses to go on; with
        one it can inherit its ancestor's proven date. ``fit_null`` calibrates
        match confidence against this specific corpus.
        """
        self.archive.add_many(entries)
        if fit_null:
            self.aligner.fit_null(self.archive)
        return self


def date_document(
    text: Optional[str] = None,
    url: Optional[str] = None,
    **kwargs,
) -> Verdict:
    """One-shot convenience wrapper around :class:`Dendro`."""
    return Dendro(**{k: v for k, v in kwargs.items() if k in _DENDRO_KWARGS}).date(
        text=text, url=url, **{k: v for k, v in kwargs.items() if k not in _DENDRO_KWARGS}
    )


_DENDRO_KWARGS = {
    "collector",
    "archive",
    "aligner",
    "propagator",
    "sources",
    "client",
    "consensus_config",
}


def _guess_title(text: Optional[str]) -> Optional[str]:
    """First substantial line, used only to seed registry title searches.

    Deliberately crude, and it never becomes evidence on its own: the registry
    sources gate every title hit through a strict token-overlap check before
    turning it into a witness.
    """
    if not text:
        return None
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if 12 <= len(stripped) <= 240 and len(stripped.split()) >= 3:
            return stripped
    return None
