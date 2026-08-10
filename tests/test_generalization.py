"""Dendro's accuracy must not move when the generator changes.

The claim is structural, and the tests are written to check the *structure*, not
just the numbers:

* an end-to-end assertion that per-generation AUC is flat, including on a
  generator family no detector was trained on;
* a mechanism assertion that Dendro's score for a document is a pure function of
  its evidence -- feeding it text from any generator, or no text at all, changes
  nothing about the bound.

The second is the stronger statement.  A flat curve could be luck; a scoring path
that provably never reads the generator's output cannot degrade when the
generator improves.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.baselines import auc
from benchmarks.corpus import archive_entries
from benchmarks.dendro_scorer import DendroScorer


@pytest.fixture(scope="module")
def scorer(small_corpus):
    return DendroScorer(archive_entries(small_corpus), probe_forgeries=False, offline=True)


def _auc_for(scorer, positives, negatives) -> float:
    ps = [scorer.score(d).human_origin_p for d in positives]
    ns = [scorer.score(d).human_origin_p for d in negatives]
    return auc([*ps, *ns], [1] * len(ps) + [0] * len(ns))


def test_auc_is_flat_across_generator_generations(small_corpus, scorer):
    """Axis (3): no generation is harder than any other for an evidence-based method."""
    by_gen: dict[int, list[dict]] = {}
    for d in small_corpus.synthetic:
        by_gen.setdefault(d["generation"], []).append(d)

    aucs = {g: _auc_for(scorer, small_corpus.human_old[:60], docs) for g, docs in sorted(by_gen.items())}
    values = list(aucs.values())
    assert min(values) >= 0.95, f"a generation degraded Dendro: {aucs}"
    assert max(values) - min(values) <= 0.02, f"AUC varied across generations: {aucs}"


def test_auc_holds_on_an_unseen_generator_family(small_corpus, scorer):
    """The family shift that breaks learned detectors does not touch evidence."""
    assert small_corpus.unseen_family, "corpus is missing the held-out generator family"
    seen = _auc_for(scorer, small_corpus.human_old[:60], small_corpus.synthetic[:60])
    unseen = _auc_for(scorer, small_corpus.human_old[:60], small_corpus.unseen_family)
    assert unseen >= 0.95
    assert abs(seen - unseen) <= 0.02, f"seen {seen:.3f} vs unseen family {unseen:.3f}"


def test_generator_ladder_is_a_real_axis(small_corpus):
    """The x-axis of the headline figure must be a measurement, not a label.

    If the measured distance to human text did not shrink along the ladder, then
    "generations improve" would be a stipulation and the baselines' decay would
    prove nothing.
    """
    ladder = [r for r in small_corpus.ladder if r.get("family", "temperature") == "temperature"]
    gaps = [r["logloss_gap"] for r in sorted(ladder, key=lambda r: r["generation"])]
    assert gaps[0] > gaps[-1], f"ladder is not monotone in detectability: {gaps}"
    assert gaps[0] / max(gaps[-1], 1e-6) > 2.0, "the ladder spans too little range to be informative"


def test_score_does_not_depend_on_the_text_when_evidence_is_present(small_corpus, scorer):
    """The mechanism: swap the prose, keep the witnesses, get the same answer.

    This is what "generator-independent" means operationally. No language model
    output -- from any generation, seen or unseen -- can move a verdict that rests
    on archival evidence.
    """
    doc = dict(small_corpus.human_old[0])
    baseline = scorer.score(doc)

    for replacement in (
        small_corpus.synthetic[0]["text"],
        small_corpus.unseen_family[0]["text"],
        "completely unrelated prose about badgers " * 40,
    ):
        mutated = {**doc, "doc_id": doc["doc_id"] + ":mutated", "text": replacement}
        got = scorer.score(mutated)
        assert got.bound.not_after == baseline.bound.not_after
        assert got.human_origin_p == pytest.approx(baseline.human_origin_p, abs=1e-9)


def test_synthetic_documents_land_at_the_prior(small_corpus, scorer):
    """No evidence must produce the base rate and an abstention -- never an accusation."""
    for d in small_corpus.synthetic[:12]:
        v = scorer.score(d)
        assert not v.bound.has_evidence
        assert v.abstained
        assert 0.3 < v.human_origin_p < 0.8, v.human_origin_p
        assert v.ci_high - v.ci_low > 0.3, "an evidence-free document got a narrow interval"


def test_recent_human_documents_are_abstained_not_accused(small_corpus, scorer):
    """Dendro's honest weakness, pinned as a test so it cannot silently change.

    Proving a document existed in 2025 is *not* evidence that a human wrote it.
    Dendro must therefore decline to claim these are human, and must equally
    decline to claim they are not.
    """
    verdicts = [scorer.score(d) for d in small_corpus.human_recent[:40]]
    assert np.mean([v.abstained for v in verdicts]) > 0.5
    assert all(v.human_origin_p > 0.35 for v in verdicts), "recent human docs pushed toward 'synthetic'"
    assert all(v.bound.has_evidence for v in verdicts), "recent docs should still have a date bound"
