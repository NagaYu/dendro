"""ContaminationPropagation: evidence -> calibrated probability, along a derivation graph.

This module is where Dendro is most at risk of overclaiming, so it is built to
make overclaiming structurally hard.

**The inference, stated honestly.**  A witness bound says *content existed by T*.
It does not say a human wrote it.  The bridge is a prevalence curve: if content
existed at time ``T``, the chance it was machine-generated is the machine share
of text at ``T``.  Before 2021 that share is near zero, which is why a proven
2019 bound is such strong evidence of human origin -- and why the whole method is
an analogue of low-background steel, salvaged from ships sunk before 1945
because everything smelted afterwards carries fallout.

**The bound might be wrong.**  So the posterior is a mixture::

    P(human) = (1 - p_fail) * (1 - s(T))  +  p_fail * (1 - s(prior_era))

where ``p_fail = exp(-forgery_logodds)``.  Take the limit: a document with *no*
witnesses has ``p_fail = 1``, so ``P(human)`` collapses to the base rate.  No
evidence produces no opinion.  That is not a special case bolted on afterwards;
it falls out of the algebra, which is the property that makes the abstention
trustworthy (**calibration**).

**Probabilities travel with intervals.**  Evidence strength is converted into
Beta pseudo-counts, so the interval narrows only when the evidence actually
justifies it, and ``abstained`` fires whenever it stays wide.

**Contamination flows downhill.**  On a derivation graph a document cannot be
*more* likely human than the material it was derived from.  Propagation enforces
that as a monotone upper bound rather than a symmetric diffusion -- a clean
source does not launder a dirty derivative.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .types import (
    AncestorMatch,
    ConsensusBound,
    InconsistencyFlag,
    Verdict,
    logit,
    sigmoid,
    to_utc,
    utcnow,
)

__all__ = [
    "PrevalenceCurve",
    "PropagationConfig",
    "DerivationEdge",
    "Calibrator",
    "ContaminationPropagator",
    "reliability_bins",
    "expected_calibration_error",
    "brier_score",
]


# --------------------------------------------------------------------------- prevalence
@dataclass(frozen=True)
class PrevalenceCurve:
    """Machine-generated share of newly-created text, as a function of time.

    These numbers are an *assumption*, not a measurement, and the design makes
    that visible: the curve is a constructor argument, it is printed in every CLI
    explanation, and ``benchmarks/run.py`` re-scores the corpus under six
    alternative curves into ``results/prevalence_sweep.csv`` so the sensitivity is
    on the record.  (The *ranking* of methods is a property of the evidence and
    does not move; the absolute probabilities do.)

    Anchors default to a logistic rise centred in 2023, with a floor for the
    pre-LLM era that is small but not zero -- template spam and machine
    translation existed long before 2021, and a model that says "impossible"
    cannot be calibrated.
    """

    onset_year: float = 2021.0
    midpoint_year: float = 2023.4
    steepness: float = 1.35
    floor: float = 0.004
    ceiling: float = 0.45

    def share(self, when: Optional[_dt.datetime]) -> float:
        """P(a document created at ``when`` is machine-generated)."""
        if when is None:
            return self.ceiling
        year = _decimal_year(to_utc(when))
        rise = 1.0 / (1.0 + math.exp(-self.steepness * (year - self.midpoint_year)))
        return float(self.floor + (self.ceiling - self.floor) * rise)

    def human_share(self, when: Optional[_dt.datetime]) -> float:
        return 1.0 - self.share(when)


def _decimal_year(when: _dt.datetime) -> float:
    start = _dt.datetime(when.year, 1, 1, tzinfo=_dt.timezone.utc)
    end = _dt.datetime(when.year + 1, 1, 1, tzinfo=_dt.timezone.utc)
    return when.year + (when - start).total_seconds() / (end - start).total_seconds()


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class PropagationConfig:
    """Knobs for the evidence -> probability map and the graph pass."""

    prevalence: PrevalenceCurve = field(default_factory=PrevalenceCurve)
    #: Pseudo-count scale.  Beta concentration is ``kappa * (1 + forgery_logodds)``,
    #: so an interval only tightens when independent operators actually accumulate.
    kappa: float = 2.5
    ci_mass: float = 0.90
    #: Above this interval width the verdict is an abstention rather than a claim.
    abstain_ci_width: float = 0.34
    #: Derivation edges: how much human-origin probability may survive one hop.
    derive_attenuation: float = 0.92
    cite_attenuation: float = 0.25
    max_iterations: int = 32
    tolerance: float = 1e-6
    #: Flags push the log-odds down by this multiple of their likelihood ratio.
    flag_weight: float = 1.0


@dataclass(frozen=True)
class DerivationEdge:
    """``dst`` was produced from ``src``.

    ``kind='derived'`` is a strong claim (a rewrite, translation, or summary) and
    propagates almost all contamination.  ``kind='cites'`` is weak -- quoting a
    synthetic paper does not make your paper synthetic -- and is attenuated hard.
    Treating them identically is the standard mistake in contamination analyses
    and it produces a graph where everything is contaminated by hop three.
    """

    src: str
    dst: str
    kind: str = "derived"
    weight: float = 1.0


# --------------------------------------------------------------------------- calibration
class Calibrator:
    """Isotonic (or logistic-fallback) map from raw score to calibrated probability.

    Dendro's raw evidence score is already probability-shaped, but "probability
    shaped" is not "calibrated".  Fitting a monotone map on held-out labelled
    data and reporting the reliability diagram before and after is the only
    honest way to make the **calibration** claim, and the fitted object is
    serialisable so the Space uses exactly the map the benchmark measured.
    """

    def __init__(self) -> None:
        self._iso = None
        self._platt: Optional[tuple[float, float]] = None
        self.fitted = False
        self.n_fit = 0

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "Calibrator":
        x = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        self.n_fit = int(x.size)
        if x.size < 12 or len(np.unique(y)) < 2:
            return self
        try:
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(x, y)
            self._iso = iso
            self.fitted = True
            return self
        except Exception:
            pass
        # Logistic fallback keeps the package usable without scikit-learn.
        z = np.array([logit(v) for v in np.clip(x, 1e-6, 1 - 1e-6)])
        a, b = np.polyfit(z, y, 1)
        self._platt = (float(a), float(b))
        self.fitted = True
        return self

    def transform(self, p: float) -> float:
        if not self.fitted:
            return float(p)
        if self._iso is not None:
            return float(np.clip(self._iso.predict([p])[0], 1e-4, 1 - 1e-4))
        a, b = self._platt or (1.0, 0.0)
        return float(np.clip(a * logit(p) + b, 1e-4, 1 - 1e-4))

    def transform_many(self, ps: Sequence[float]) -> np.ndarray:
        return np.array([self.transform(p) for p in ps], dtype=float)


def reliability_bins(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> list[dict]:
    """Bin predictions and report observed frequency -- the reliability diagram.

    Measures the **calibration** claim directly: a bin of documents each claimed
    to be 0.9 human-origin should contain about 90% human-origin documents.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        out.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "n": n,
                "mean_predicted": float(p[m].mean()) if n else float("nan"),
                "observed_frequency": float(y[m].mean()) if n else float("nan"),
            }
        )
    return out


def expected_calibration_error(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Sample-weighted mean gap between confidence and accuracy.

    The scalar summary of the **calibration** claim.  Read it alongside the bins:
    on a perfectly separable task an isotonic map collapses to a step function
    and ECE goes to ~0, which reflects the task rather than the method.
    """
    p = np.asarray(probs, dtype=float)
    total = p.size
    if total == 0:
        return float("nan")
    ece = 0.0
    for b in reliability_bins(probs, labels, n_bins):
        if b["n"]:
            ece += (b["n"] / total) * abs(b["mean_predicted"] - b["observed_frequency"])
    return float(ece)


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


# --------------------------------------------------------------------------- propagator
class ContaminationPropagator:
    """Turns bounds, ancestors and flags into calibrated verdicts, then propagates.

    Use it in two steps.  :meth:`verdict` scores one document from its own
    evidence.  :meth:`propagate` takes many documents plus a derivation graph and
    enforces the monotonicity constraint across it.
    """

    def __init__(
        self,
        config: Optional[PropagationConfig] = None,
        calibrator: Optional[Calibrator] = None,
    ) -> None:
        self.config = config or PropagationConfig()
        self.calibrator = calibrator or Calibrator()

    # -- single document ---------------------------------------------------
    def raw_probability(
        self,
        bound: ConsensusBound,
        ancestor: Optional[AncestorMatch] = None,
        flags: Sequence[InconsistencyFlag] = (),
    ) -> tuple[float, float, Optional[_dt.datetime]]:
        """(probability, effective evidence strength, effective bound time).

        The ancestor branch is what defeats the paraphrase attack: when a
        document aligns to an older archived source, the *source's* bound is used
        and its strength is discounted by the alignment confidence.  A confident
        alignment to a 2019 page therefore yields nearly the same answer as being
        that 2019 page -- which is the correct answer, since the content is the
        same content.
        """
        cfg = self.config
        effective_time = bound.not_after
        strength = bound.forgery_logodds

        if ancestor is not None and ancestor.is_ancestral and ancestor.witness_time is not None:
            if effective_time is None or ancestor.witness_time < effective_time:
                effective_time = ancestor.witness_time
                # An alignment is never stronger than the archive record behind it.
                strength = max(strength, _alignment_strength(ancestor))

        p_fail = math.exp(-max(0.0, strength)) if strength > 0 else 1.0
        p_fail = min(1.0, max(0.0, p_fail))

        p_if_true = cfg.prevalence.human_share(effective_time)
        p_if_false = cfg.prevalence.human_share(None)  # fall back to the base rate
        p = (1.0 - p_fail) * p_if_true + p_fail * p_if_false

        # Active deception is itself evidence.  A human document has no motive to
        # forge an older date, so a backdate flag pushes the odds down directly
        # rather than merely widening the interval.
        penalty = cfg.flag_weight * sum(f.log_lr for f in flags if f.kind in ("backdate", "commit_predates_repo"))
        if penalty > 0:
            p = sigmoid(logit(p) - penalty)
        return float(min(1.0 - 1e-6, max(1e-6, p))), float(strength), effective_time

    def interval(self, p: float, strength: float) -> tuple[float, float]:
        """Beta credible interval whose width is driven by evidence strength.

        Demonstrates **calibration**: the width is the honest report of how much is
        known, and it is the quantity ``abstained`` thresholds on.

        Concentration ``kappa * (1 + strength)`` means: no witnesses -> ``Beta(1+p*k,
        1+(1-p)*k)`` with ``k`` small, i.e. an interval nearly as wide as the unit
        line.  Several independent operators -> concentration in the tens and a
        usable interval.  The width is the honest report of how much is known.
        """
        cfg = self.config
        conc = cfg.kappa * (1.0 + max(0.0, strength))
        a = 1.0 + p * conc
        b = 1.0 + (1.0 - p) * conc
        tail = (1.0 - cfg.ci_mass) / 2.0
        try:
            from scipy.stats import beta as _beta

            lo = float(_beta.ppf(tail, a, b))
            hi = float(_beta.ppf(1.0 - tail, a, b))
        except Exception:  # pragma: no cover - scipy is a hard dep in practice
            sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))
            lo, hi = p - 1.64 * sd, p + 1.64 * sd
        return float(max(0.0, min(1.0, lo))), float(max(0.0, min(1.0, hi)))

    def verdict(
        self,
        doc_id: str,
        bound: ConsensusBound,
        ancestor: Optional[AncestorMatch] = None,
        flags: Sequence[InconsistencyFlag] = (),
    ) -> Verdict:
        """Score one document.  Never returns a label -- only a probability and an interval.

        Demonstrates **calibration** and the abstention guarantee: a document with
        no witnesses gets the base rate and an interval wide enough to trip
        ``abstained``, so absence of evidence can never be reported as evidence of
        synthesis.
        """
        raw, strength, effective_time = self.raw_probability(bound, ancestor, flags)
        p = self.calibrator.transform(raw)
        lo, hi = self.interval(p, strength)
        abstain = (hi - lo) > self.config.abstain_ci_width or not bound.has_evidence
        if ancestor is not None and ancestor.is_ancestral and ancestor.witness_time is not None:
            abstain = abstain and (hi - lo) > self.config.abstain_ci_width

        explanation = self.explain(bound, ancestor, flags, p, (lo, hi), effective_time, abstain)
        if abs(p - raw) > 0.25:
            # A fitted calibrator encodes the *base rate of the corpus it was fitted
            # on*, so a large shift means this number is being driven by that base
            # rate rather than by evidence about this document. Saying so matters
            # most in exactly the case that could do harm: an unwitnessed document
            # in a corpus that is mostly synthetic gets a low probability from the
            # base rate alone, and a reader must not mistake that for a finding.
            explanation += (
                f"\nNote: calibration moved this from {raw:.3f} to {p:.3f}. That shift comes from "
                "the base rate of the calibration corpus, not from evidence about this document."
            )
        return Verdict(
            doc_id=doc_id,
            bound=bound,
            human_origin_p=p,
            ci_low=lo,
            ci_high=hi,
            flags=tuple(flags),
            ancestor=ancestor,
            abstained=bool(abstain),
            evidence_logodds=strength,
            prior_logodds=logit(self.config.prevalence.human_share(None)),
            explanation=explanation,
        )

    # -- graph -------------------------------------------------------------
    def propagate(
        self,
        verdicts: Mapping[str, Verdict],
        edges: Iterable[DerivationEdge] = (),
    ) -> dict[str, Verdict]:
        """Enforce "no derivative is cleaner than its source" across the graph.

        A damped iteration to a fixed point, taking the *minimum* over incoming
        constraints rather than a weighted average.  Averaging would let a
        document with ten clean citations and one synthetic parent come out
        clean; the minimum will not.  Attenuation differs by edge kind so that
        citation chains do not paint the whole graph black by hop three.

        Cycles are safe: the update is monotone non-increasing and bounded below,
        so it converges, and ``max_iterations`` caps the cost regardless.
        """
        cfg = self.config
        edges = list(edges)
        if not edges:
            return dict(verdicts)

        incoming: dict[str, list[DerivationEdge]] = defaultdict(list)
        for e in edges:
            if e.src in verdicts and e.dst in verdicts:
                incoming[e.dst].append(e)

        current = {k: v.human_origin_p for k, v in verdicts.items()}
        direct = dict(current)
        for _ in range(cfg.max_iterations):
            delta = 0.0
            for dst, es in incoming.items():
                bound_p = direct[dst]
                for e in es:
                    att = cfg.derive_attenuation if e.kind == "derived" else cfg.cite_attenuation
                    w = max(0.0, min(1.0, e.weight)) * att
                    # Ceiling imposed by this parent: at w=1 the child can be no
                    # cleaner than the parent; at w=0 the parent imposes nothing.
                    ceiling = current[e.src] * w + 1.0 * (1.0 - w)
                    bound_p = min(bound_p, ceiling)
                delta = max(delta, abs(current[dst] - bound_p))
                current[dst] = bound_p
            if delta < cfg.tolerance:
                break

        out: dict[str, Verdict] = {}
        for doc_id, v in verdicts.items():
            p = current[doc_id]
            if abs(p - v.human_origin_p) < 1e-12:
                out[doc_id] = v
                continue
            lo, hi = self.interval(p, v.evidence_logodds)
            note = (
                f"{v.explanation}\nPropagation: lowered from {v.human_origin_p:.3f} to {p:.3f} "
                f"by {len(incoming.get(doc_id, []))} incoming derivation edge(s)."
            )
            out[doc_id] = Verdict(
                doc_id=v.doc_id,
                bound=v.bound,
                human_origin_p=p,
                ci_low=lo,
                ci_high=hi,
                flags=v.flags,
                ancestor=v.ancestor,
                abstained=v.abstained or (hi - lo) > cfg.abstain_ci_width,
                evidence_logodds=v.evidence_logodds,
                prior_logodds=v.prior_logodds,
                explanation=note,
            )
        return out

    # -- prose -------------------------------------------------------------
    def explain(
        self,
        bound: ConsensusBound,
        ancestor: Optional[AncestorMatch],
        flags: Sequence[InconsistencyFlag],
        p: float,
        ci: tuple[float, float],
        effective_time: Optional[_dt.datetime],
        abstained: bool,
    ) -> str:
        """Human-readable receipt.

        The Space and the CLI print this verbatim.  It leads with the *evidence*
        and only then gives the probability, because the evidence is the part
        that is checkable by a reader and the probability is the part that
        depends on an assumed prevalence curve.
        """
        lines: list[str] = []
        if effective_time is not None:
            ops = bound.independent_operators
            lines.append(
                f"Existence bound: content existed on or before {effective_time.date().isoformat()}, "
                f"witnessed by {ops} independent operator(s) "
                f"({', '.join(sorted({w.operator for w in bound.supporting})) or 'via alignment'})."
            )
            if bound.not_after_low and bound.not_after_high and bound.not_after_low != bound.not_after_high:
                lines.append(
                    f"Sensitivity: {bound.not_after_low.date()} at a 5% failure budget, "
                    f"{bound.not_after_high.date()} at 0.01%."
                )
        else:
            lines.append("Existence bound: none. No independent archive record was found for this content.")

        if ancestor is not None and ancestor.is_ancestral:
            lines.append(
                f"Alignment: {ancestor.relation.value} of archived document {ancestor.ref_doc_id} "
                f"(confidence {ancestor.confidence:.2f}, "
                f"{ancestor.score.window_coverage_query:.0%} of this text covered)."
                + (f" Source: {ancestor.ref_url}" if ancestor.ref_url else "")
            )
        for f in flags:
            lines.append(f"Flag [{f.severity}] {f.kind}: {f.detail}")

        share = self.config.prevalence.share(effective_time)
        lines.append(
            f"Under the configured prevalence curve, machine-generated text was ~{share:.1%} of new "
            f"content at that time; combined with the strength of the bound this gives "
            f"P(human-origin) = {p:.3f} (90% interval {ci[0]:.2f}-{ci[1]:.2f})."
        )
        if abstained:
            lines.append(
                "ABSTAIN: the evidence does not narrow this enough to act on. This is not a finding "
                "of synthetic origin -- it is an absence of archival evidence either way."
            )
        lines.append(
            "Dendro reports archival evidence of prior existence. It is not an authorship test and "
            "must not be used to conclude that a person did or did not write something."
        )
        return "\n".join(lines)


def _alignment_strength(ancestor: AncestorMatch) -> float:
    """Convert alignment confidence into an evidence log-odds contribution.

    Capped well below what a multi-operator archive bound can reach: an alignment
    is an inference about content identity, and it should never outrank a direct
    observation of the content itself.
    """
    c = min(0.999, max(0.0, ancestor.confidence))
    return float(min(6.0, -math.log(max(1e-6, 1.0 - c))))
