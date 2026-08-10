"""Calibration, abstention, graph propagation, and subset optimality.

The invariants here are the ones that keep the tool from being dangerous:
no evidence produces the prior and a wide interval; evidence narrows the
interval only in proportion to how independent it is; and contamination flows
downhill on a derivation graph without a clean parent laundering a dirty child.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from dendro.corpus_report import CorpusReport, build_low_background_subset
from dendro.propagate import (
    Calibrator,
    ContaminationPropagator,
    DerivationEdge,
    PrevalenceCurve,
    PropagationConfig,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)
from dendro.types import UTC, ConsensusBound, Verdict, Witness, WitnessKind


def _bound(year: int | None, operators: int = 2, logodds: float = 14.0) -> ConsensusBound:
    if year is None:
        return ConsensusBound(not_after=None)
    when = _dt.datetime(year, 6, 1, tzinfo=UTC)
    supporting = tuple(
        Witness(
            source_id=f"s{i}", operator=f"op{i}", kind=WitnessKind.SNAPSHOT,
            observed_at=when, target="t", forgeability=1e-3,
        )
        for i in range(operators)
    )
    return ConsensusBound(
        not_after=when, independent_operators=operators,
        forgery_logodds=logodds, supporting=supporting, all_witnesses=supporting,
    )


@pytest.fixture
def prop() -> ContaminationPropagator:
    return ContaminationPropagator()


# --------------------------------------------------------------------------- basics
def test_old_bound_gives_high_probability(prop):
    v = prop.verdict("old", _bound(2019))
    assert v.human_origin_p > 0.95
    assert not v.abstained
    assert v.not_after.year == 2019


def test_no_evidence_gives_the_prior_and_a_wide_interval(prop):
    v = prop.verdict("unknown", _bound(None))
    prior = PrevalenceCurve().human_share(None)
    assert v.human_origin_p == pytest.approx(prior, abs=0.02)
    assert v.abstained
    assert v.ci_high - v.ci_low > 0.4


def test_probability_decreases_monotonically_with_era(prop):
    ps = [prop.verdict(str(y), _bound(y)).human_origin_p for y in (2015, 2019, 2022, 2024, 2026)]
    assert all(a >= b for a, b in zip(ps, ps[1:])), ps
    assert ps[0] - ps[-1] > 0.25


def test_interval_narrows_as_independent_evidence_accumulates(prop):
    widths = [
        prop.verdict("w", _bound(2019, operators=n, logodds=lo)).ci_width
        for n, lo in ((1, 5.0), (2, 12.0), (4, 26.0))
    ]
    assert widths[0] > widths[1] > widths[2], widths


def test_weak_evidence_cannot_produce_a_confident_answer(prop):
    """A single forgeable witness must not yield certainty."""
    weak = _bound(2019, operators=1, logodds=0.7)
    v = prop.verdict("weak", weak)
    assert v.ci_width > 0.3
    assert v.abstained


def test_backdate_flag_pushes_the_probability_down(prop):
    from dendro.types import InconsistencyFlag

    clean = prop.verdict("a", _bound(2019))
    flagged = prop.verdict(
        "b", _bound(2019),
        flags=[InconsistencyFlag(kind="backdate", log_lr=6.0, detail="x", coverage=0.9)],
    )
    assert flagged.human_origin_p < clean.human_origin_p - 0.3


def test_ancestor_transports_an_older_bound(prop):
    from dendro.types import AlignmentScore, AncestorMatch, Relation

    ancestor = AncestorMatch(
        ref_doc_id="src",
        witness_time=_dt.datetime(2019, 1, 1, tzinfo=UTC),
        relation=Relation.DERIVED,
        score=AlignmentScore(0.1, 0.2, 0.2, 0.8, 0.8, 0.7),
        confidence=0.95,
    )
    without = prop.verdict("x", _bound(None))
    with_anc = prop.verdict("x", _bound(None), ancestor=ancestor)
    assert with_anc.not_after.year == 2019
    assert with_anc.human_origin_p > without.human_origin_p + 0.2


# --------------------------------------------------------------------------- graph
def test_contamination_flows_downhill_not_uphill(prop):
    parent = prop.verdict("parent", _bound(None))          # unknown, ~prior
    child = prop.verdict("child", _bound(2019))            # strong old evidence
    verdicts = {"parent": parent, "child": child}

    # child derived FROM parent: the child cannot be cleaner than its source.
    out = prop.propagate(verdicts, [DerivationEdge("parent", "child", "derived", 1.0)])
    assert out["child"].human_origin_p <= child.human_origin_p
    assert out["parent"].human_origin_p == pytest.approx(parent.human_origin_p)


def test_citation_edges_attenuate_far_more_than_derivation(prop):
    dirty = prop.verdict("dirty", _bound(2026, operators=3, logodds=20.0))
    clean = prop.verdict("clean", _bound(2019))
    base = {"dirty": dirty, "clean": clean}

    derived = prop.propagate(dict(base), [DerivationEdge("dirty", "clean", "derived", 1.0)])
    cited = prop.propagate(dict(base), [DerivationEdge("dirty", "clean", "cites", 1.0)])
    assert derived["clean"].human_origin_p < cited["clean"].human_origin_p


def test_propagation_terminates_on_a_cycle(prop):
    a = prop.verdict("a", _bound(2019))
    b = prop.verdict("b", _bound(2026, operators=3, logodds=20.0))
    out = prop.propagate(
        {"a": a, "b": b},
        [DerivationEdge("a", "b", "derived"), DerivationEdge("b", "a", "derived")],
    )
    assert set(out) == {"a", "b"}
    assert all(0.0 <= v.human_origin_p <= 1.0 for v in out.values())


def test_a_clean_parent_does_not_launder_a_dirty_child(prop):
    dirty = prop.verdict("dirty", _bound(2026, operators=3, logodds=20.0))
    clean = prop.verdict("clean", _bound(2019))
    out = prop.propagate(
        {"clean": clean, "dirty": dirty},
        [DerivationEdge("clean", "dirty", "derived", 1.0)],
    )
    assert out["dirty"].human_origin_p <= dirty.human_origin_p + 1e-9


# --------------------------------------------------------------------------- calibration
def test_calibrator_improves_reliability():
    rng = np.random.default_rng(0)
    truth = rng.random(600)
    labels = (rng.random(600) < truth).astype(int)
    skewed = np.clip(truth**2.2, 1e-4, 1 - 1e-4)          # systematically under-confident

    before = expected_calibration_error(skewed, labels)
    cal = Calibrator().fit(skewed[:300], labels[:300])
    after = expected_calibration_error(cal.transform_many(skewed[300:]), labels[300:])
    assert after < before, f"calibration made things worse: {before:.3f} -> {after:.3f}"
    assert after < 0.12


def test_reliability_bins_and_scores_are_sane():
    probs = [0.05, 0.15, 0.5, 0.85, 0.95]
    labels = [0, 0, 1, 1, 1]
    bins = reliability_bins(probs, labels, n_bins=5)
    assert len(bins) == 5
    assert sum(b["n"] for b in bins) == 5
    assert 0.0 <= brier_score(probs, labels) <= 1.0
    assert not np.isnan(expected_calibration_error(probs, labels))


def test_uncalibrated_calibrator_is_the_identity():
    cal = Calibrator()
    assert cal.transform(0.37) == pytest.approx(0.37)


def test_a_large_calibration_shift_is_disclosed(prop):
    """A base-rate-driven number must announce itself as one.

    Regression guard for a real bug: a calibrator fitted on a split containing no
    *recent human* documents mapped them into the accusatory region, reporting
    P(human) ~ 0.03 for genuine 2025 abstracts. The fix is to calibrate on data
    covering the deployment distribution, but the disclosure below is the
    backstop -- whenever calibration moves a verdict a long way, the explanation
    says the shift came from the corpus base rate rather than from evidence
    about this document.
    """

    class _Collapse(Calibrator):
        def __init__(self):
            super().__init__()
            self.fitted = True

        def transform(self, p: float) -> float:
            return 0.02 if p < 0.9 else 0.99

    prop.calibrator = _Collapse()
    v = prop.verdict("x", _bound(None))
    assert v.human_origin_p == pytest.approx(0.02)
    assert "base rate of the calibration corpus" in v.explanation
    assert v.abstained, "a collapsed calibration must not also remove the abstention"


def test_verdict_explanation_always_carries_the_scope_limit(prop):
    """Every verdict says what it is not. This text is the product, not decoration."""
    for bound in (_bound(2019), _bound(None), _bound(2026)):
        v = prop.verdict("x", bound)
        assert "not an authorship test" in v.explanation


# --------------------------------------------------------------------------- subsets
def _verdict(doc_id: str, p: float, lo: float = None, hi: float = None) -> Verdict:
    lo = p - 0.05 if lo is None else lo
    hi = p + 0.05 if hi is None else hi
    return Verdict(doc_id=doc_id, bound=_bound(2019), human_origin_p=p,
                   ci_low=max(0.0, lo), ci_high=min(1.0, hi))


def test_subset_respects_the_purity_constraint():
    rng = np.random.default_rng(3)
    verdicts = [_verdict(f"d{i}", float(p)) for i, p in enumerate(rng.random(500))]
    for limit in (0.01, 0.05, 0.2):
        plan = build_low_background_subset(verdicts, limit)
        assert plan.expected_synthetic_fraction <= limit + 1e-9, plan
        assert plan.n_selected > 0


def test_greedy_subset_is_maximum_cardinality():
    """No other set of the same size, and none larger, can satisfy the constraint.

    Checked by brute force on a small instance: the exchange argument in the
    docstring says the k highest-probability documents minimise the constraint
    sum for every k, so any larger feasible set would contradict it.
    """
    import itertools

    ps = [0.99, 0.97, 0.95, 0.80, 0.70, 0.60, 0.10]
    verdicts = [_verdict(f"d{i}", p) for i, p in enumerate(ps)]
    limit = 0.10
    plan = build_low_background_subset(verdicts, limit)

    best = 0
    for k in range(1, len(ps) + 1):
        for combo in itertools.combinations(ps, k):
            if sum(1 - p for p in combo) <= limit * k + 1e-12:
                best = max(best, k)
    assert plan.n_selected == best, f"greedy took {plan.n_selected}, optimum is {best}"


def test_conservative_mode_is_stricter_than_expected_mode():
    verdicts = [_verdict(f"d{i}", 0.9, lo=0.6, hi=0.99) for i in range(100)]
    lenient = build_low_background_subset(verdicts, 0.1, mode="expected")
    strict = build_low_background_subset(verdicts, 0.1, mode="conservative")
    assert strict.n_selected < lenient.n_selected


def test_low_background_ratio_keeps_unknown_separate():
    """Documents with no evidence must never be counted as contaminated."""
    verdicts = [
        _verdict("old", 0.98),
        Verdict(doc_id="unknown", bound=_bound(None), human_origin_p=0.55, ci_low=0.2, ci_high=0.9,
                abstained=True),
        Verdict(doc_id="new", bound=_bound(2025), human_origin_p=0.6, ci_low=0.5, ci_high=0.7),
    ]
    ratio = CorpusReport.from_verdicts(verdicts).summary()
    assert ratio["lowbackground_low_background"] == pytest.approx(1 / 3)
    assert ratio["lowbackground_unknown"] == pytest.approx(1 / 3)
    assert ratio["lowbackground_contaminated"] == pytest.approx(1 / 3)


def test_report_summary_has_no_labels():
    """The report must not emit a binary verdict column anywhere."""
    report = CorpusReport.from_verdicts([_verdict("a", 0.9), _verdict("b", 0.2)])
    row_keys = set(report.to_rows()[0])
    assert not row_keys & {"is_synthetic", "label", "verdict", "is_ai"}
    assert {"human_origin_p", "ci_low", "ci_high", "abstained"} <= row_keys
