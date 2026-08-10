"""CorpusReport: dataset-level provenance summaries and low-background subsets.

The practical question a dataset builder has is not "is this document
synthetic?" but "how much of this corpus predates the contamination, and can I
carve out a part that is clean enough to train on?".  This module answers both,
and it answers the second one *optimally* rather than heuristically.

The name is borrowed from the physics.  Low-background steel is steel smelted
before the 1945 atmospheric tests: it is not special steel, it is ordinary steel
that happens to have been made before the contamination existed, and it is
valuable precisely because no amount of post-1945 care can reproduce it.  Text
archived before the models shipped has the same property, and the same
scarcity.

Everything reported here is a distribution with intervals, never a set of
labels.  Documents Dendro abstains on are counted as abstentions, not as
synthetic -- a corpus report that silently reclassifies "unknown" as "generated"
would be worse than no report at all.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from .types import Verdict, to_utc

__all__ = ["CorpusReport", "SubsetPlan", "low_background_ratio", "build_low_background_subset"]


# --------------------------------------------------------------------------- subset
@dataclass(frozen=True)
class SubsetPlan:
    """The result of a constrained subset selection."""

    doc_ids: tuple[str, ...]
    expected_synthetic_fraction: float
    constraint: float
    mode: str
    n_selected: int
    n_total: int
    threshold_p: float

    @property
    def retention(self) -> float:
        return self.n_selected / self.n_total if self.n_total else 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "constraint": self.constraint,
            "n_selected": self.n_selected,
            "n_total": self.n_total,
            "retention": round(self.retention, 4),
            "expected_synthetic_fraction": round(self.expected_synthetic_fraction, 5),
            "threshold_p": round(self.threshold_p, 4),
        }


def build_low_background_subset(
    verdicts: Sequence[Verdict],
    max_synthetic_fraction: float = 0.05,
    mode: str = "expected",
) -> SubsetPlan:
    """Largest subset whose expected synthetic fraction stays under the constraint.

    Demonstrates **calibration** put to work: the guarantee is only as good as the
    probabilities, and ``mode="conservative"`` spends the *interval* rather than
    the point estimate so a published subset does not quietly assume the
    optimistic reading of its own uncertainty.

    **The greedy solution is optimal, not approximate.**  Let ``q_i = 1 - p_i`` be
    the probability document ``i`` is synthetic, and ``x`` the allowed fraction.
    A set ``S`` is feasible when ``sum_{i in S} q_i <= x * |S|``, i.e. when
    ``sum_{i in S} (q_i - x) <= 0``.  Write ``c_i = q_i - x``.  Among all sets of
    a given size ``k``, the one minimising ``sum c_i`` is the ``k`` documents with
    the smallest ``c_i`` -- which is the ``k`` with the *highest* ``p_i``.  So if
    sorting descending by ``p`` and taking the first ``k`` is infeasible, no set
    of size ``k`` is feasible.  Scanning ``k`` and keeping the largest feasible
    prefix therefore returns a maximum-cardinality solution exactly.

    ``mode``:

    ``expected``
        uses the point estimate ``p``.  The constraint holds in expectation.
    ``conservative``
        uses ``ci_low``, so the constraint holds even if every document sits at
        the pessimistic end of its interval.  This is the mode to use when the
        subset will be published, because an interval is a statement about what
        is *not* known and a published dataset should not spend that budget.
    """
    if not verdicts:
        return SubsetPlan((), 0.0, max_synthetic_fraction, mode, 0, 0, 1.0)

    def p_of(v: Verdict) -> float:
        return v.ci_low if mode == "conservative" else v.human_origin_p

    ordered = sorted(verdicts, key=lambda v: (-p_of(v), v.doc_id))
    x = float(max_synthetic_fraction)

    best_k, running = 0, 0.0
    for k, v in enumerate(ordered, start=1):
        running += (1.0 - p_of(v)) - x
        if running <= 1e-12:
            best_k = k

    chosen = ordered[:best_k]
    exp_frac = float(np.mean([1.0 - p_of(v) for v in chosen])) if chosen else 0.0
    return SubsetPlan(
        doc_ids=tuple(v.doc_id for v in chosen),
        expected_synthetic_fraction=exp_frac,
        constraint=x,
        mode=mode,
        n_selected=len(chosen),
        n_total=len(verdicts),
        threshold_p=float(p_of(chosen[-1])) if chosen else 1.0,
    )


def low_background_ratio(
    verdicts: Sequence[Verdict],
    before: Optional[_dt.datetime] = None,
    min_p: float = 0.90,
    use_ci: bool = True,
) -> dict[str, float]:
    """Share of the corpus with *proven* pre-contamination existence.

    Deliberately strict, and deliberately three-valued.  A document counts as
    low-background only when an independent archive bound places it before the
    cutoff **and** the calibrated probability (its lower interval end, if
    ``use_ci``) clears ``min_p``.  Everything else splits into ``unknown`` -- no
    evidence either way -- and ``contaminated`` -- evidence of post-cutoff
    origin.  The three numbers are reported separately because collapsing
    ``unknown`` into either of the others is the single easiest way to turn this
    tool into a misinformation machine.
    """
    if not verdicts:
        return {"n": 0, "low_background": 0.0, "unknown": 0.0, "contaminated": 0.0}

    cutoff = to_utc(before) if before is not None else _dt.datetime(2021, 1, 1, tzinfo=_dt.timezone.utc)
    low = unknown = contaminated = 0
    for v in verdicts:
        p = v.ci_low if use_ci else v.human_origin_p
        na = v.not_after
        if na is not None and na < cutoff and p >= min_p:
            low += 1
        elif na is not None and na >= cutoff:
            contaminated += 1
        else:
            unknown += 1
    n = len(verdicts)
    return {
        "n": float(n),
        "low_background": low / n,
        "unknown": unknown / n,
        "contaminated": contaminated / n,
        "cutoff": cutoff.date().isoformat(),
        "min_p": min_p,
        "criterion": "ci_low" if use_ci else "point",
    }


# --------------------------------------------------------------------------- report
@dataclass
class CorpusReport:
    """Aggregate provenance statistics for a collection of verdicts."""

    verdicts: list[Verdict] = field(default_factory=list)

    @classmethod
    def from_verdicts(cls, verdicts: Iterable[Verdict]) -> "CorpusReport":
        return cls(list(verdicts))

    # -- distribution ------------------------------------------------------
    def histogram(self, n_bins: int = 20) -> list[dict[str, Any]]:
        ps = np.array([v.human_origin_p for v in self.verdicts], dtype=float)
        if ps.size == 0:
            return []
        counts, edges = np.histogram(ps, bins=n_bins, range=(0.0, 1.0))
        return [
            {"bin_low": float(edges[i]), "bin_high": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(len(counts))
        ]

    def year_histogram(self) -> list[dict[str, Any]]:
        """Distribution of proven existence years -- the corpus's age profile."""
        years: dict[Any, int] = {}
        for v in self.verdicts:
            na = v.not_after
            key = na.year if na else "no-evidence"
            years[key] = years.get(key, 0) + 1
        known = sorted(k for k in years if isinstance(k, int))
        rows = [{"year": y, "count": years[y]} for y in known]
        if "no-evidence" in years:
            rows.append({"year": "no-evidence", "count": years["no-evidence"]})
        return rows

    def evidence_profile(self) -> dict[str, Any]:
        """Where the evidence comes from, and how independent it is."""
        ops: dict[str, int] = {}
        srcs: dict[str, int] = {}
        n_ops: list[int] = []
        for v in self.verdicts:
            n_ops.append(v.bound.independent_operators)
            for w in v.bound.all_witnesses:
                if w.is_independent_evidence:
                    ops[w.operator] = ops.get(w.operator, 0) + 1
                    srcs[w.source_id] = srcs.get(w.source_id, 0) + 1
        return {
            "operators": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
            "sources": dict(sorted(srcs.items(), key=lambda kv: -kv[1])),
            "mean_independent_operators": float(np.mean(n_ops)) if n_ops else 0.0,
            "single_operator_share": float(np.mean([n == 1 for n in n_ops])) if n_ops else 0.0,
        }

    # -- headline ----------------------------------------------------------
    def summary(self, cutoff: Optional[_dt.datetime] = None, min_p: float = 0.90) -> dict[str, Any]:
        vs = self.verdicts
        if not vs:
            return {"n": 0}
        ps = np.array([v.human_origin_p for v in vs], dtype=float)
        widths = np.array([v.ci_width for v in vs], dtype=float)
        flagged = [v for v in vs if v.flags]
        return {
            "n": len(vs),
            "mean_human_origin_p": float(ps.mean()),
            "median_human_origin_p": float(np.median(ps)),
            "mean_ci_width": float(widths.mean()),
            "abstain_rate": float(np.mean([v.abstained for v in vs])),
            "with_evidence_rate": float(np.mean([v.bound.has_evidence for v in vs])),
            "flagged_rate": len(flagged) / len(vs),
            "flag_kinds": _count_flags(vs),
            **{f"lowbackground_{k}": v for k, v in low_background_ratio(vs, cutoff, min_p).items()},
            "evidence": self.evidence_profile(),
        }

    def subset(self, max_synthetic_fraction: float = 0.05, mode: str = "expected") -> SubsetPlan:
        return build_low_background_subset(self.verdicts, max_synthetic_fraction, mode)

    def subset_curve(
        self, fractions: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.20), mode: str = "expected"
    ) -> list[dict[str, Any]]:
        """Retention as a function of the purity constraint -- the trade-off curve.

        This is the plot a dataset builder actually needs: "what does 1% cost me
        versus 10%?".  It is also the honest way to present the method, because
        it makes the price of purity explicit instead of quoting one flattering
        operating point.
        """
        return [self.subset(f, mode).as_row() for f in fractions]

    def to_rows(self) -> list[dict[str, Any]]:
        return [v.as_row() for v in self.verdicts]


def _count_flags(verdicts: Sequence[Verdict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in verdicts:
        for f in v.flags:
            out[f.kind] = out.get(f.kind, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
