"""Known-old documents must produce correct existence upper bounds.

These run against **real, committed archive responses** -- the actual Wayback CDX
and arXiv API payloads, replayed offline.  The expected dates are facts about the
world that can be checked by hand: "Attention Is All You Need" was submitted to
arXiv on 2017-06-12, and the Internet Archive holds captures of PEP 20 from 2006.

The point of pinning real documents rather than synthetic fixtures is that a
regression in the parsing, the timestamp handling, or the consensus estimator
shows up as a *wrong date for a paper you know*, which is impossible to
rationalise away.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from dendro.types import UTC, WitnessKind
from dendro.witness import ConsensusConfig, Target, WitnessCollector, combine_failure_probability


@pytest.fixture
def collector(offline_client) -> WitnessCollector:
    return WitnessCollector(client=offline_client)


def test_arxiv_paper_gets_its_true_submission_date(collector):
    """A registration witness pins the bound to the day, from real API data."""
    target = Target(doc_id="attention", url="https://arxiv.org/abs/1706.03762")
    bound = collector.consensus(collector.collect(target))

    assert bound.has_evidence
    assert bound.not_after is not None
    # arXiv v1 submission: 2017-06-12.
    assert bound.not_after.date() == _dt.date(2017, 6, 12)
    assert any(w.kind is WitnessKind.REGISTRATION for w in bound.supporting)


def test_old_web_page_bounded_well_before_the_llm_era(collector):
    """PEP 20 is witnessed by the Internet Archive in 2006, not 2026."""
    target = Target(doc_id="pep20", url="https://www.python.org/dev/peps/pep-0020/")
    bound = collector.consensus(collector.collect(target))

    assert bound.has_evidence
    assert bound.not_after.year <= 2008, f"bound too late: {bound.not_after}"
    assert bound.not_after.year >= 2000, f"bound implausibly early: {bound.not_after}"
    assert bound.forgery_logodds > 4.0


def test_multiple_operators_are_found_for_a_well_known_url(collector):
    """The independence story needs more than one organisation to be real."""
    target = Target(doc_id="pep20", url="https://www.python.org/dev/peps/pep-0020/")
    witnesses = collector.collect(target)
    operators = {w.operator for w in witnesses if w.is_independent_evidence}
    assert len(operators) >= 2, f"only found {operators}"


def test_no_evidence_when_nothing_is_cached(no_network_client):
    """A source that cannot answer contributes nothing -- and never invents a date."""
    collector = WitnessCollector(client=no_network_client)
    target = Target(doc_id="unknown", url="https://example.invalid/never-archived")
    bound = collector.consensus(collector.collect(target))
    assert not bound.has_evidence
    assert bound.not_after is None
    assert bound.independent_operators == 0


def test_self_asserted_dates_never_tighten_a_bound(no_network_client):
    """The core rule: a document's own claim is not evidence about itself."""
    collector = WitnessCollector(client=no_network_client)
    text = '<html><head><meta name="date" content="1999-01-01"></head><body>hello world</body></html>'
    target = Target(doc_id="claimy", text=text)
    witnesses = collector.collect(target)

    assert any(w.kind is WitnessKind.SELF_ASSERTED for w in witnesses)
    bound = collector.consensus(witnesses)
    assert bound.not_after is None, "a self-asserted date established a bound"
    assert bound.independent_operators == 0


# --------------------------------------------------------------------------- estimator
def _w(operator: str, day: str, forgeability: float = 1e-3, reliability: float = 0.99):
    from dendro.types import Witness

    return Witness(
        source_id=operator,
        operator=operator,
        kind=WitnessKind.SNAPSHOT,
        observed_at=_dt.datetime.fromisoformat(day).replace(tzinfo=UTC),
        target="t",
        reliability=reliability,
        forgeability=forgeability,
    )


def test_repeat_captures_from_one_operator_buy_little():
    """Twenty captures from one archive are still one archive.

    This is the arithmetic behind the adversarial-robustness claim: within an
    operator, failure is correlated, so the group's failure probability is
    floored at its forgeability no matter how many records it holds.
    """
    one = [_w("ia", "2019-01-01")]
    twenty = [_w("ia", f"2019-01-{d:02d}") for d in range(1, 21)]
    p_one, ops_one, _ = combine_failure_probability(one)
    p_many, ops_many, _ = combine_failure_probability(twenty)

    assert ops_one == ops_many == 1
    assert p_many < p_one                      # some accidental-error benefit
    assert p_many > 1e-3 * 0.5                 # but floored by forgeability
    assert p_many / p_one > 1e-3               # nowhere near 20 independent draws


def test_a_second_operator_multiplies_the_difficulty():
    """Independent operators multiply exactly; that is the whole design.

    Two groups with identical parameters must give ``p_two == p_one ** 2``.  The
    assertion is on the identity rather than on a round-number ratio, because the
    ratio is just ``1 / p_one`` and pinning it to "at least 100x" would be a claim
    about the default reliability constant rather than about independence.
    """
    one = [_w("ia", "2019-01-01")]
    two = [_w("ia", "2019-01-01"), _w("cc", "2019-01-02")]
    p_one, _, _ = combine_failure_probability(one)
    p_two, ops, _ = combine_failure_probability(two)
    assert ops == 2
    assert p_two == pytest.approx(p_one**2, rel=1e-9)
    assert p_two < p_one / 50.0


def test_three_operators_beat_thirty_captures_from_one():
    """Diversity dominates volume -- the claim, as a direct comparison."""
    thirty_one_operator = [_w("ia", f"2019-01-{1 + d % 28:02d}") for d in range(30)]
    three_operators = [_w("ia", "2019-01-01"), _w("cc", "2019-01-02"), _w("arxiv", "2019-01-03")]
    p_many, ops_many, _ = combine_failure_probability(thirty_one_operator)
    p_diverse, ops_diverse, _ = combine_failure_probability(three_operators)
    assert ops_many == 1 and ops_diverse == 3
    assert p_diverse < p_many


def test_a_lone_early_witness_cannot_drag_the_bound_back():
    """Robustness of the estimator: injecting one fake early capture is not enough.

    The bound is an evidence-weighted order statistic, not a minimum.  A single
    forged 2005 record sits alone in its operator group, so the budget never
    clears there and the reported date stays where the corroborated evidence is.
    """
    honest = [
        _w("ia", "2019-06-01"),
        _w("cc", "2019-06-05"),
        _w("arxiv", "2019-06-02", forgeability=5e-4),
    ]
    attacked = [_w("evil", "2005-01-01", forgeability=0.4), *honest]

    cfg = ConsensusConfig(alpha=1e-3)
    from dendro.cache import Cache, HttpClient

    collector = WitnessCollector(sources=[], client=HttpClient(offline=True), config=cfg)
    clean = collector.consensus(honest)
    dirty = collector.consensus(attacked)

    assert clean.not_after.year == 2019
    assert dirty.not_after.year == 2019, f"a single forged witness moved the bound to {dirty.not_after}"


def test_interval_widens_when_evidence_is_thin(offline_client):
    """not_after_high demands more corroboration and therefore lands later."""
    collector = WitnessCollector(client=offline_client)
    target = Target(doc_id="pep20", url="https://www.python.org/dev/peps/pep-0020/")
    bound = collector.consensus(collector.collect(target))
    assert bound.not_after_low <= bound.not_after <= bound.not_after_high
