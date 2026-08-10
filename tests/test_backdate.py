"""Backdate forgery must be caught by witness inconsistency -- and only then.

Two properties, and the second matters more than the first:

1. A document that claims to be old, in a neighbourhood the archives were
   demonstrably crawling, with no independent record, is flagged.
2. A document with no coverage measurement behind it is **never** flagged, no
   matter how old its claim.  Most text that ever existed was never archived;
   a detector that treats obscurity as guilt is worse than none.

The likelihood ratio has the right limit built in: ``log LR = -k·log(1-c)``
goes to exactly zero as coverage ``c`` goes to zero, so the "no accusation
without measurement" property is algebraic rather than a guard clause.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from dendro.cache import HttpClient
from dendro.types import UTC, Witness, WitnessKind
from dendro.witness import Target, WitnessCollector

NOW = _dt.datetime.now(tz=UTC)


@pytest.fixture
def collector(no_network_client) -> WitnessCollector:
    return WitnessCollector(sources=[], client=no_network_client)


def _snapshot(operator: str, when: str, forgeability: float = 1e-3) -> Witness:
    return Witness(
        source_id=operator,
        operator=operator,
        kind=WitnessKind.SNAPSHOT,
        observed_at=_dt.datetime.fromisoformat(when).replace(tzinfo=UTC),
        target="t",
        forgeability=forgeability,
    )


def _claim(when: str) -> Witness:
    return Witness(
        source_id="self_asserted",
        operator="document-itself",
        kind=WitnessKind.SELF_ASSERTED,
        observed_at=_dt.datetime.fromisoformat(when).replace(tzinfo=UTC),
        target="t",
        reliability=0.6,
        forgeability=0.98,
    )


# --------------------------------------------------------------------------- detection
def test_claim_with_no_record_under_coverage_is_flagged(collector):
    """The pure forgery: claims 2019, no archive has it, archives were looking."""
    target = Target(doc_id="forged", url="https://www.python.org/dev/peps/pep-9999-fake/",
                    claimed_date=_dt.datetime(2019, 3, 11, tzinfo=UTC))
    witnesses = [_claim("2019-03-11")]
    bound = collector.consensus(witnesses)
    flags = collector.detect_inconsistencies(
        target, witnesses, bound, coverage={"wayback": 0.92, "commoncrawl": 0.80}
    )
    kinds = {f.kind for f in flags}
    assert "backdate" in kinds
    backdate = next(f for f in flags if f.kind == "backdate")
    assert backdate.log_lr > 3.0
    assert backdate.severity == "high"


def test_first_sighting_years_after_the_claim_is_flagged(collector):
    """Archives exist for the document -- but only from long after it claims."""
    target = Target(doc_id="late", url="https://example.org/x",
                    claimed_date=_dt.datetime(2018, 1, 1, tzinfo=UTC))
    witnesses = [_claim("2018-01-01"), _snapshot("internet-archive", "2025-07-02")]
    bound = collector.consensus(witnesses)
    flags = collector.detect_inconsistencies(target, witnesses, bound, coverage={"wayback": 0.9})
    assert "backdate" in {f.kind for f in flags}


def test_commit_predating_its_repository_is_flagged(collector):
    """A git date is an environment variable; the forge's creation date is not."""
    witness = Witness(
        source_id="github",
        operator="github",
        kind=WitnessKind.COMMIT,
        observed_at=_dt.datetime(2019, 4, 1, tzinfo=UTC),
        target="repo/file.py",
        forgeability=3e-2,
        raw={"repo_created_at": "2025-02-14T00:00:00Z"},
    )
    target = Target(doc_id="repo", url="https://github.com/a/b")
    bound = collector.consensus([witness])
    flags = collector.detect_inconsistencies(target, [witness], bound)
    assert "commit_predates_repo" in {f.kind for f in flags}
    assert next(f for f in flags if f.kind == "commit_predates_repo").log_lr >= 4.0


# --------------------------------------------------------------------------- restraint
def test_no_coverage_means_no_accusation(collector):
    """Absence of evidence is not evidence of absence -- enforced by the formula."""
    target = Target(doc_id="obscure", claimed_date=_dt.datetime(2005, 6, 1, tzinfo=UTC))
    witnesses = [_claim("2005-06-01")]
    bound = collector.consensus(witnesses)
    flags = collector.detect_inconsistencies(target, witnesses, bound, coverage={})
    assert "backdate" not in {f.kind for f in flags}

    # Even an explicit zero-coverage report must not trigger it.
    flags = collector.detect_inconsistencies(
        target, witnesses, bound, coverage={"wayback": 0.0, "commoncrawl": 0.01}
    )
    assert "backdate" not in {f.kind for f in flags}


def test_genuine_old_document_is_not_flagged(collector):
    """Corroborated claims must pass cleanly, or the flag is worthless."""
    target = Target(doc_id="genuine", url="https://example.org/x",
                    claimed_date=_dt.datetime(2019, 3, 1, tzinfo=UTC))
    witnesses = [
        _claim("2019-03-01"),
        _snapshot("internet-archive", "2019-03-04"),
        _snapshot("commoncrawl-org", "2019-04-10"),
    ]
    bound = collector.consensus(witnesses)
    flags = collector.detect_inconsistencies(target, witnesses, bound, coverage={"wayback": 0.9})
    assert "backdate" not in {f.kind for f in flags}
    assert bound.not_after.year == 2019


def test_recent_document_claiming_a_recent_date_is_not_flagged(collector):
    """The age guard: a 2026 document claiming 2026 is unremarkable."""
    recent = NOW - _dt.timedelta(days=30)
    target = Target(doc_id="new", url="https://example.org/new", claimed_date=recent)
    witnesses = [_claim(recent.date().isoformat())]
    bound = collector.consensus(witnesses)
    flags = collector.detect_inconsistencies(target, witnesses, bound, coverage={"wayback": 0.9})
    assert "backdate" not in {f.kind for f in flags}


def test_log_lr_scales_with_coverage_and_source_count(collector):
    """More covering sources, and deeper coverage, mean a stronger accusation."""
    target = Target(doc_id="f", url="https://example.org/f",
                    claimed_date=_dt.datetime(2018, 1, 1, tzinfo=UTC))
    witnesses = [_claim("2018-01-01")]
    bound = collector.consensus(witnesses)

    def lr(coverage):
        flags = collector.detect_inconsistencies(target, witnesses, bound, coverage=coverage)
        hits = [f for f in flags if f.kind == "backdate"]
        return hits[0].log_lr if hits else 0.0

    weak = lr({"wayback": 0.30})
    strong = lr({"wayback": 0.95})
    two = lr({"wayback": 0.95, "commoncrawl": 0.90})
    assert 0.0 < weak < strong < two


# --------------------------------------------------------------------------- end to end
def test_backdated_corpus_documents_are_flagged(small_corpus):
    """End-to-end on the generated attack split, using the offline scorer."""
    from benchmarks.corpus import archive_entries
    from benchmarks.dendro_scorer import DendroScorer

    scorer = DendroScorer(archive_entries(small_corpus), probe_forgeries=False, offline=True)
    # Supply the coverage that a live probe would have measured for these hosts;
    # offline the probe itself cannot run, but the *decision rule* is what is
    # under test here, and it is exercised with a realistic measurement.
    scorer._coverage_cache = {}
    flagged = 0
    for doc in small_corpus.backdated:
        from dendro.witness import Target as T

        target = T(doc_id=doc["doc_id"], url=doc["url"], text=doc["text"],
                   claimed_date=doc["claimed_date"])
        from dendro.sources.selfasserted import SelfAssertedSource

        witnesses = SelfAssertedSource().collect(target, scorer.client)
        bound = scorer.collector.consensus(witnesses)
        flags = scorer.collector.detect_inconsistencies(
            target, witnesses, bound, coverage={"wayback": 0.9, "commoncrawl": 0.75}
        )
        if any(f.kind == "backdate" for f in flags):
            flagged += 1
    assert flagged == len(small_corpus.backdated), (
        f"only {flagged}/{len(small_corpus.backdated)} forgeries flagged"
    )


def test_forged_dates_are_extracted_from_every_surface():
    """A forger picks whichever field the platform reads, so all four are parsed."""
    from benchmarks.generators import backdate
    from dendro.sources.selfasserted import extract_claimed_dates

    when = _dt.datetime(2019, 3, 11, tzinfo=UTC)
    for style in ("html_meta", "frontmatter", "jsonld", "prose"):
        text = backdate("Some ordinary looking prose about a topic.", when, style=style)
        claims = extract_claimed_dates(text)
        assert claims, f"no claimed date extracted from style {style}"
        assert any(c[1].date() == when.date() for c in claims), f"wrong date parsed for {style}"
