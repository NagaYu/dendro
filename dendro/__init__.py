"""Dendro -- prove a document existed before a date, without reading it.

    Synthetic-text detectors ask "does this *look* generated?".  That question
    gets harder every time a model ships.  Dendro asks a different one --
    "who, other than the author, saw this content, and when?" -- and that
    question does not get harder at all.

The name is from dendrochronology: you do not date a beam by inspecting the
grain for signs of modernity, you match its ring pattern against an independent
chronology built from wood whose age is already known.  Dendro matches a
document's fingerprint against archives whose timestamps are already witnessed.

The four pieces:

``witness``
    Collects "operator *O* observed this at time *T*" records from independent
    archives and combines them into an existence bound whose confidence is driven
    by the number of *independent operators*, not the number of records.
``fingerprint``
    Format-blind, paraphrase-resistant sketches, so the same prose recognises
    itself across HTML, WET, Markdown, and rewrites.
``alignment``
    Finds the oldest archived ancestor of a candidate document, including from
    partial coverage, and transports its bound onto the candidate.
``propagate``
    Turns evidence into a *calibrated probability with an interval*, propagates
    contamination along derivation edges, and flags documents whose claims
    contradict the archives.

What Dendro does **not** do, by construction: decide that a person wrote something
with a machine.  It reports evidence of prior existence and its strength.  A
document with no witnesses gets an abstention, never an accusation.  See
``README.md`` for the intended-use statement.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .types import (  # noqa: F401
    AlignmentScore,
    AncestorMatch,
    ConsensusBound,
    Document,
    Fingerprint,
    InconsistencyFlag,
    Relation,
    Verdict,
    Witness,
    WitnessKind,
)

__all__ = [
    "__version__",
    "AlignmentScore",
    "AncestorMatch",
    "ConsensusBound",
    "Document",
    "Fingerprint",
    "InconsistencyFlag",
    "Relation",
    "Verdict",
    "Witness",
    "WitnessKind",
    "WitnessCollector",
    "Target",
    "ReflowFingerprint",
    "Aligner",
    "ArchiveLayer",
    "ContaminationPropagator",
    "CorpusReport",
    "date_document",
]


def __getattr__(name: str):
    """Lazy re-exports so ``import dendro`` stays fast and dependency-light."""
    if name in ("WitnessCollector", "Target"):
        from . import witness

        return getattr(witness, name)
    if name == "ReflowFingerprint":
        from .fingerprint import ReflowFingerprint

        return ReflowFingerprint
    if name in ("Aligner", "ArchiveLayer"):
        from . import alignment

        return getattr(alignment, name)
    if name == "ContaminationPropagator":
        from .propagate import ContaminationPropagator

        return ContaminationPropagator
    if name in ("Dendro", "date_document"):
        from . import pipeline

        return getattr(pipeline, name)
    if name == "CorpusReport":
        from .corpus_report import CorpusReport

        return CorpusReport
    raise AttributeError(f"module 'dendro' has no attribute {name!r}")
