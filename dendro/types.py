"""Shared data contracts for Dendro.

Every type here exists to keep one distinction sharp, because the whole project
collapses if it is blurred:

    Dendro proves *existence before a time*.  It does not prove *authorship*.

A `ConsensusBound.not_after` is a statement about the world ("some archive that
does not answer to the document's author observed this content at time T").  A
`Verdict.human_origin_p` is an *inference* from that statement plus a prior over
when synthetic text became abundant.  The two are stored separately, and every
consumer -- CLI, Space, dataset annotator -- is expected to surface the bound,
not just the probability.

Claims exercised by this module:

* **adversarial-robustness** -- `Witness.forgeability` and `Witness.operator`
  are what make backdate forgery expensive: an adversary must compromise
  independent operators, not just edit a `<meta>` tag.
* **generator-independence** -- nothing in these types references a text model,
  a perplexity, or a token distribution.  The evidence channel is structurally
  incapable of degrading when a new generator ships.
* **calibration** -- probabilities always travel with an interval and an
  `abstained` flag, so "no evidence" is representable and never collapses into
  "evidence of synthesis".
"""

from __future__ import annotations

import datetime as _dt
import enum
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

UTC = _dt.timezone.utc

__all__ = [
    "UTC",
    "WitnessKind",
    "Witness",
    "ConsensusBound",
    "InconsistencyFlag",
    "ChannelSketch",
    "WindowSketch",
    "Fingerprint",
    "AlignmentScore",
    "AncestorMatch",
    "Verdict",
    "Document",
    "utcnow",
    "to_utc",
    "logit",
    "sigmoid",
]


# --------------------------------------------------------------------------- time
def utcnow() -> _dt.datetime:
    """Timezone-aware now(), so no naive/aware comparison can ever raise."""
    return _dt.datetime.now(tz=UTC)


def to_utc(value: _dt.datetime | _dt.date | str | int | float) -> _dt.datetime:
    """Coerce anything date-like into a tz-aware UTC datetime.

    Archive APIs disagree wildly on format (``20190101040106`` from the Wayback
    CDX, ISO-8601 from arXiv, unix epochs from git).  Normalising at the
    boundary means the consensus code below never has to think about it.
    """
    if isinstance(value, _dt.datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, (int, float)):
        return _dt.datetime.fromtimestamp(float(value), tz=UTC)

    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")

    # Wayback CDX 14-digit stamp (and its truncated variants).
    if text.isdigit() and len(text) in (4, 6, 8, 10, 12, 14):
        padded = text + "00000101000000"[len(text) :]
        return _dt.datetime.strptime(padded[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC)

    cleaned = text.replace("Z", "+00:00")
    try:
        return to_utc(_dt.datetime.fromisoformat(cleaned))
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",
        # "Published on March 11, 2019" is how most CMSs render a date into the
        # body, which makes it the most common surface a backdate forgery uses.
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%a, %d %b %Y %H:%M:%S %z",
    ):
        try:
            return to_utc(_dt.datetime.strptime(text, fmt))
        except ValueError:
            continue
    raise ValueError(f"unparseable timestamp: {value!r}")


# --------------------------------------------------------------------------- math
def logit(p: float) -> float:
    """Log-odds with saturation, so a probability of exactly 0 or 1 is finite."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    """Inverse of :func:`logit`, overflow-safe on both tails."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# --------------------------------------------------------------------------- witness
class WitnessKind(str, enum.Enum):
    """What *kind* of act produced the timestamp.

    The ordering matters for :mod:`dendro.propagate`: ``SELF_ASSERTED`` is the
    only kind that the document's own author controls, so it is never allowed to
    tighten a bound.  It exists solely so that backdate forgery has something to
    be *inconsistent with* -- demonstrating **adversarial-robustness**.
    """

    SNAPSHOT = "snapshot"           # a crawler stored the bytes at this instant
    REGISTRATION = "registration"   # a registry (arXiv, DOI, ISBN) minted a record
    COMMIT = "commit"               # a VCS object carries this author/commit date
    POSTING = "posting"             # a public list/forum archived a message
    PUBLICATION = "publication"     # a publisher's own dated record
    CITATION = "citation"           # a third party dated document referenced this one
    SELF_ASSERTED = "self_asserted" # the document says so about itself -- NOT evidence

    @property
    def is_independent_evidence(self) -> bool:
        """True when the timestamp is produced by someone other than the author."""
        return self is not WitnessKind.SELF_ASSERTED


@dataclass(frozen=True)
class Witness:
    """One archival observation: "operator *O* saw content *C* at time *T*".

    Attributes are split into three families on purpose.

    *Identity* -- ``source_id`` names the plugin, ``operator`` names the legal
    entity running the infrastructure.  Two witnesses that share an ``operator``
    are treated as **fully correlated under adversarial compromise**: whoever can
    forge one can forge the other.  Wayback and Common Crawl are separate
    operators; two Wayback captures are not.  This is the mechanism behind the
    **adversarial-robustness** claim.

    *Timing* -- ``observed_at`` is the moment the observation happened, which
    upper-bounds the content's first existence.

    *Quality* -- three independent probabilities:

    ``reliability``
        P(the timestamp is what it claims to be | witness is not forged).
        Captures clock skew and sloppy metadata, not attack.
    ``forgeability``
        P(a motivated adversary could have injected or rewritten this witness).
        ``SELF_ASSERTED`` sits at ~1.0; an Internet Archive capture is ~1e-3.
    ``coverage``
        P(this source would have observed the document *had it existed* at the
        claimed time).  Without this, "no early snapshot" is uninterpretable --
        it is the difference between *evidence of absence* and *absence of
        evidence*, and it is what makes backdate detection sound.
    """

    source_id: str
    operator: str
    kind: WitnessKind
    observed_at: _dt.datetime
    target: str
    reliability: float = 0.99
    forgeability: float = 1e-3
    coverage: float = 0.0
    content_digest: Optional[str] = None
    fingerprint_id: Optional[str] = None
    url: Optional[str] = None
    cached: bool = False
    fetch_seconds: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", to_utc(self.observed_at))
        for name in ("reliability", "forgeability", "coverage"):
            v = float(getattr(self, name))
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
            object.__setattr__(self, name, v)

    @property
    def is_independent_evidence(self) -> bool:
        return self.kind.is_independent_evidence

    def as_row(self) -> dict[str, Any]:
        """Flat dict for tables in the CLI, the Space, and dataset columns."""
        return {
            "source": self.source_id,
            "operator": self.operator,
            "kind": self.kind.value,
            "observed_at": self.observed_at.isoformat(),
            "target": self.target,
            "url": self.url,
            "reliability": round(self.reliability, 4),
            "forgeability": round(self.forgeability, 6),
            "coverage": round(self.coverage, 4),
            "cached": self.cached,
        }


@dataclass(frozen=True)
class ConsensusBound:
    """The output of :class:`dendro.witness.WitnessCollector`.

    ``not_after`` is the headline: *the content existed at or before this
    instant*.  It is deliberately **not** the minimum observation time -- a
    single spoofed early witness would then move the bound arbitrarily far into
    the past.  It is the earliest time at which the surviving independent
    evidence mass clears a threshold; see
    :meth:`dendro.witness.WitnessCollector.consensus` for the estimator.

    ``forgery_logodds`` is ``log P(bound is genuine) / P(every supporting
    witness was forged)``.  It grows linearly in the number of *independent
    operators* and not at all in the number of captures from one operator,
    which is exactly the **adversarial-robustness** claim in numeric form.
    """

    not_after: Optional[_dt.datetime]
    not_after_low: Optional[_dt.datetime] = None
    not_after_high: Optional[_dt.datetime] = None
    independent_operators: int = 0
    effective_witnesses: float = 0.0
    forgery_logodds: float = 0.0
    total_coverage: float = 0.0
    supporting: tuple[Witness, ...] = ()
    all_witnesses: tuple[Witness, ...] = ()
    method: str = "weighted-order-statistic"

    @property
    def has_evidence(self) -> bool:
        return self.not_after is not None and self.independent_operators > 0

    @property
    def year(self) -> Optional[int]:
        return self.not_after.year if self.not_after else None

    def as_row(self) -> dict[str, Any]:
        return {
            "not_after": self.not_after.isoformat() if self.not_after else None,
            "not_after_low": self.not_after_low.isoformat() if self.not_after_low else None,
            "not_after_high": self.not_after_high.isoformat() if self.not_after_high else None,
            "independent_operators": self.independent_operators,
            "effective_witnesses": round(self.effective_witnesses, 3),
            "forgery_logodds": round(self.forgery_logodds, 3),
            "n_witnesses": len(self.all_witnesses),
        }


@dataclass(frozen=True)
class InconsistencyFlag:
    """A detected conflict between what a document claims and what archives say.

    ``log_lr`` is a likelihood ratio, ``log P(observation | forged) / P(observation
    | genuine)``, so flags compose additively and can be fed straight into the
    propagation log-odds.  Every flag carries the coverage figure it was computed
    from, because a conflict is only meaningful where the archives were looking.
    """

    kind: str
    log_lr: float
    detail: str
    claimed: Optional[_dt.datetime] = None
    observed: Optional[_dt.datetime] = None
    coverage: float = 0.0
    sources: tuple[str, ...] = ()

    @property
    def severity(self) -> str:
        if self.log_lr >= 4.0:
            return "high"
        if self.log_lr >= 1.5:
            return "medium"
        return "low"

    def as_row(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "log_lr": round(self.log_lr, 3),
            "detail": self.detail,
            "claimed": self.claimed.isoformat() if self.claimed else None,
            "observed": self.observed.isoformat() if self.observed else None,
            "coverage": round(self.coverage, 4),
            "sources": list(self.sources),
        }


# --------------------------------------------------------------------------- fingerprint
@dataclass(frozen=True)
class ChannelSketch:
    """A MinHash + SimHash sketch of one *view* of a document.

    Dendro fingerprints every document through several channels at once
    (``word5`` exact shingles, ``char5`` character n-grams, ``rare`` content-word
    sets, ``num`` numeral/entity sets).  Channels degrade under different
    attacks: reflow destroys nothing, light editing chips ``word5``, paraphrase
    guts ``word5`` and ``char5`` but leaves ``rare``/``num`` largely intact.
    Keeping them separate is what lets alignment survive paraphrase and is the
    measured core of the **adversarial-robustness** claim.
    """

    name: str
    minhash: np.ndarray
    simhash: int
    cardinality: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        arr = np.asarray(self.minhash, dtype=np.uint64)
        arr.setflags(write=False)
        object.__setattr__(self, "minhash", arr)
        object.__setattr__(self, "simhash", int(self.simhash) & ((1 << 64) - 1))
        object.__setattr__(self, "cardinality", int(self.cardinality))


@dataclass(frozen=True)
class WindowSketch:
    """A sketch of a contiguous token window.

    Whole-document Jaccard cannot see a 300-token quotation inside a 12k-token
    page.  Window sketches can: the quote matches two or three windows exactly,
    and alignment reports partial coverage instead of "unrelated".  This is what
    makes **partial-coverage ancestry** decidable.
    """

    index: int
    start_token: int
    end_token: int
    minhash: np.ndarray
    simhash: int

    def __post_init__(self) -> None:
        arr = np.asarray(self.minhash, dtype=np.uint64)
        arr.setflags(write=False)
        object.__setattr__(self, "minhash", arr)
        object.__setattr__(self, "simhash", int(self.simhash) & ((1 << 64) - 1))


@dataclass(frozen=True)
class Fingerprint:
    """A reflow-invariant, format-blind sketch of a document.

    Two byte-streams that render to the same prose -- different HTML wrapper,
    different line wrapping, different nav chrome, one with a cookie banner --
    produce the *same* ``normalized_sha256``.  That equality is the strongest
    form of the **reflow-invariance** claim, and it is asserted directly in
    ``tests/test_fingerprint.py``.
    """

    doc_id: str
    n_tokens: int
    normalized_sha256: str
    channels: Mapping[str, ChannelSketch]
    windows: tuple[WindowSketch, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def primary(self) -> ChannelSketch:
        return self.channels["word"]


@dataclass(frozen=True)
class AlignmentScore:
    """Similarity of a query document to a reference document, per channel.

    ``containment_query_in_ref`` answers "how much of the query is inside the
    reference" -- the asymmetric quantity ancestry actually needs.  Jaccard is
    reported too, but a 200-word excerpt of a 20-page paper has Jaccard ~0.01
    and containment ~1.0, and only the second number is informative.
    """

    jaccard: float
    containment_query_in_ref: float
    containment_ref_in_query: float
    window_coverage_query: float
    window_coverage_ref: float
    simhash_similarity: float
    channel_scores: Mapping[str, float] = field(default_factory=dict)
    combined: float = 0.0
    matched_windows: int = 0

    def as_row(self) -> dict[str, Any]:
        row = {
            "jaccard": round(self.jaccard, 4),
            "containment_q_in_r": round(self.containment_query_in_ref, 4),
            "containment_r_in_q": round(self.containment_ref_in_query, 4),
            "window_coverage_q": round(self.window_coverage_query, 4),
            "simhash_sim": round(self.simhash_similarity, 4),
            "combined": round(self.combined, 4),
            "matched_windows": self.matched_windows,
        }
        row.update({f"ch_{k}": round(v, 4) for k, v in self.channel_scores.items()})
        return row


class Relation(str, enum.Enum):
    """How a query document relates to a reference document."""

    IDENTICAL = "identical"       # same normalized text
    NEAR_DUPLICATE = "near_dup"   # reflow / light edit
    DERIVED = "derived"           # paraphrase or rewrite of the reference
    CONTAINS = "contains"         # query contains the reference (quotes it)
    CONTAINED_BY = "contained_by" # query is an excerpt of the reference
    UNRELATED = "unrelated"


@dataclass(frozen=True)
class AncestorMatch:
    """A reference document identified as an ancestor of the query.

    Carries ``witness_time`` so the caller can transport the reference's proven
    existence bound onto the query -- that transfer is the whole point:
    a 2026 paraphrase of a 2019 page inherits the 2019 bound, which is why
    paraphrase attacks do not move Dendro's answer.
    """

    ref_doc_id: str
    witness_time: Optional[_dt.datetime]
    relation: Relation
    score: AlignmentScore
    confidence: float
    ref_url: Optional[str] = None

    @property
    def is_ancestral(self) -> bool:
        return self.relation in (
            Relation.IDENTICAL,
            Relation.NEAR_DUPLICATE,
            Relation.DERIVED,
            Relation.CONTAINED_BY,
        )


# --------------------------------------------------------------------------- documents & verdicts
@dataclass
class Document:
    """A unit of text under examination, plus whatever it claims about itself."""

    doc_id: str
    text: str
    url: Optional[str] = None
    claimed_date: Optional[_dt.datetime] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.claimed_date is not None:
            self.claimed_date = to_utc(self.claimed_date)


@dataclass(frozen=True)
class Verdict:
    """Dendro's answer for one document.

    Deliberately *not* a label.  ``human_origin_p`` is a calibrated probability
    with a credible interval, and ``abstained`` is set whenever the interval is
    too wide to act on.  A document nobody archived gets ``abstained=True`` and
    an interval that spans the prior -- never "synthetic".  That asymmetry is the
    **calibration** claim and the ethical core of the tool: absence of evidence
    is reported as absence of evidence.
    """

    doc_id: str
    bound: ConsensusBound
    human_origin_p: float
    ci_low: float
    ci_high: float
    flags: tuple[InconsistencyFlag, ...] = ()
    ancestor: Optional[AncestorMatch] = None
    abstained: bool = False
    evidence_logodds: float = 0.0
    prior_logodds: float = 0.0
    explanation: str = ""

    @property
    def not_after(self) -> Optional[_dt.datetime]:
        """Existence upper bound, inherited from an ancestor when one is found."""
        if self.ancestor is not None and self.ancestor.witness_time is not None:
            if self.bound.not_after is None or self.ancestor.witness_time < self.bound.not_after:
                return self.ancestor.witness_time
        return self.bound.not_after

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def as_row(self) -> dict[str, Any]:
        na = self.not_after
        return {
            "doc_id": self.doc_id,
            "not_after": na.isoformat() if na else None,
            "not_after_year": na.year if na else None,
            "human_origin_p": round(self.human_origin_p, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "abstained": self.abstained,
            "n_witnesses": len(self.bound.all_witnesses),
            "independent_operators": self.bound.independent_operators,
            "forgery_logodds": round(self.bound.forgery_logodds, 3),
            "flags": [f.kind for f in self.flags],
            "max_flag_log_lr": round(max((f.log_lr for f in self.flags), default=0.0), 3),
            "ancestor": self.ancestor.ref_doc_id if self.ancestor else None,
            "ancestor_relation": self.ancestor.relation.value if self.ancestor else None,
        }

    def with_explanation(self, text: str) -> "Verdict":
        return replace(self, explanation=text)


def summarise_verdicts(verdicts: Sequence[Verdict]) -> dict[str, float]:
    """Tiny helper used by both the CLI and the corpus report."""
    if not verdicts:
        return {"n": 0}
    ps = np.array([v.human_origin_p for v in verdicts], dtype=float)
    return {
        "n": float(len(verdicts)),
        "mean_human_origin_p": float(ps.mean()),
        "median_human_origin_p": float(np.median(ps)),
        "abstain_rate": float(np.mean([v.abstained for v in verdicts])),
        "flagged_rate": float(np.mean([bool(v.flags) for v in verdicts])),
    }
