"""Reflow invariance, partial containment, and channel behaviour.

The strongest statement available for reflow invariance is not "similarity is
high" but **hash equality**: four different renderings of the same prose must
produce the identical ``normalized_sha256``.  That is what these tests assert,
because a similarity threshold can be tuned until it passes and an equality
cannot.
"""

from __future__ import annotations

import pytest

from dendro.fingerprint import (
    LshIndex,
    NormalizationConfig,
    ReflowFingerprint,
    estimate_containment,
    estimate_jaccard,
    minhash_of,
)

PROSE = """The Bergamo mortality registry recorded 1,247 excess deaths in March 2019.
Researchers at the University of Padua compared the figures with prior baselines.
They concluded that the reporting delay accounted for roughly 18 percent of the gap.
A follow-up audit in Milan reproduced the estimate using independent municipal records.
The registry has since published monthly counts alongside a revised methodology note."""

HTML = """<html><head><title>Report</title><style>.x{color:red}</style>
<script>var a=1;</script></head><body>
<nav>Home | About | Contact | Search</nav>
<div class="cookie-banner">We use cookies to improve your experience. Accept all cookies</div>
<p>The Bergamo mortality registry recorded 1,247 excess&nbsp;deaths in March 2019.<br>
Researchers at the University of Padua compared the figures with prior baselines.</p>
<p>They concluded that the reporting delay accounted for roughly 18 percent of the gap.
A follow-up audit in Milan reproduced the estimate using independent municipal records.</p>
<p>The registry has since published monthly counts alongside a revised methodology note.</p>
<footer>&copy; 2019 Example Corp. All rights reserved.</footer>
<div>Share this &middot; Tweet &middot; Print</div></body></html>"""

WET = " ".join(PROSE.split())

QUOTED = "\n".join(
    "> " + line
    for line in """The Bergamo mortality registry recorded 1,247 excess deaths in
March 2019. Researchers at the University of Padua compared the
figures with prior baselines. They concluded that the reporting
delay accounted for roughly 18 percent of the gap. A follow-up
audit in Milan reproduced the estimate using independent
municipal records. The registry has since published monthly
counts alongside a revised methodology note.""".splitlines()
)

SMART = (
    PROSE.replace("The registry", "The “registry”")
    .replace("-", "–")
    .replace("2019.", "2019…")
)


@pytest.fixture
def rf() -> ReflowFingerprint:
    return ReflowFingerprint()


def test_renderings_normalise_to_the_same_hash(rf):
    """HTML chrome, WET flattening and quote-rewrapping all vanish."""
    hashes = {
        name: rf.fingerprint(name, text).normalized_sha256
        for name, text in (("plain", PROSE), ("html", HTML), ("wet", WET), ("quoted", QUOTED))
    }
    assert len(set(hashes.values())) == 1, f"renderings diverged: {hashes}"


def test_identical_renderings_score_perfect_containment(rf):
    a = rf.fingerprint("a", HTML)
    b = rf.fingerprint("b", PROSE)
    scores = rf.channel_scores(a, b)
    for channel, value in scores.items():
        assert value == pytest.approx(1.0), f"channel {channel} lost information: {value}"


def test_typography_changes_barely_move_the_fingerprint(rf):
    """Smart quotes, en-dashes and ellipses are typography, not content."""
    a = rf.fingerprint("a", SMART)
    b = rf.fingerprint("b", PROSE)
    assert rf.channel_scores(a, b)["word"] > 0.90


def test_boilerplate_is_actually_removed(rf):
    nd = rf.normalizer.normalize(HTML)
    for junk in ("cookies", "all rights reserved", "tweet", "home | about"):
        assert junk not in nd.text, f"boilerplate survived: {junk!r}"
    assert "bergamo mortality registry" in nd.text


def test_disabling_boilerplate_stripping_breaks_the_invariance(rf):
    """The ablation confirms which stage is doing the work."""
    weak = ReflowFingerprint(normalization=NormalizationConfig(strip_boilerplate=False))
    a = weak.fingerprint("a", HTML)
    b = weak.fingerprint("b", PROSE)
    assert a.normalized_sha256 != b.normalized_sha256
    assert weak.channel_scores(a, b)["word"] < 1.0


# --------------------------------------------------------------------------- containment
def test_containment_finds_a_short_quote_inside_a_long_document(rf):
    """A 60-word excerpt has tiny Jaccard and near-total containment.

    This asymmetry is why ancestry is decided on containment: the Jaccard number
    for this pair is close to noise, and the containment number is close to 1.
    """
    long_doc = (PROSE + "\n") * 12
    excerpt = "\n".join(PROSE.splitlines()[:2])
    q = rf.fingerprint("q", excerpt)
    r = rf.fingerprint("r", long_doc)

    j = estimate_jaccard(q.channels["word"].minhash, r.channels["word"].minhash)
    containment = rf.channel_scores(q, r)["word"]
    assert containment > 0.75, f"excerpt not contained: {containment}"
    assert containment > j


def test_containment_estimator_matches_exact_sets():
    a = [f"tok{i}" for i in range(200)]
    b = [f"tok{i}" for i in range(100, 700)]
    sa, sb = minhash_of(a, 256), minhash_of(b, 256)
    j = estimate_jaccard(sa, sb)
    est = estimate_containment(j, len(set(a)), len(set(b)))
    true = len(set(a) & set(b)) / len(set(a))
    assert est == pytest.approx(true, abs=0.15), f"estimate {est} vs truth {true}"


def test_unrelated_documents_score_low(rf):
    other = (
        "Sea-surface temperature anomalies in the Tasman basin during austral summer were "
        "reconstructed from coral cores. The chronology relies on annual density banding and "
        "was cross-checked against instrumental records from three Australian stations."
    )
    scores = rf.channel_scores(rf.fingerprint("a", other), rf.fingerprint("b", PROSE))
    assert scores["word"] < 0.10
    assert scores["rare"] < 0.30


# --------------------------------------------------------------------------- windows & lsh
def test_windows_localise_a_quotation(rf):
    long_doc = "\n".join(f"Filler sentence number {i} about unrelated matters." for i in range(120))
    mixed = long_doc + "\n" + PROSE + "\n" + long_doc
    fp = rf.fingerprint("mixed", mixed)
    src = rf.fingerprint("src", PROSE)
    assert len(fp.windows) > 3
    best = max(estimate_jaccard(w.minhash, src.windows[0].minhash) for w in fp.windows)
    assert best > 0.05, "no window localised the embedded quotation"


def test_lsh_retrieves_a_reflowed_duplicate(rf):
    index = LshIndex()
    index.add(rf.fingerprint("plain", PROSE))
    for i in range(40):
        index.add(rf.fingerprint(f"noise{i}", f"Unrelated document {i}. " + "lorem ipsum dolor sit amet. " * 30))
    hits = index.query(rf.fingerprint("query", HTML))
    assert "plain" in hits, "LSH failed to retrieve the reflowed duplicate"
    assert hits["plain"] >= max(v for k, v in hits.items() if k != "plain") if len(hits) > 1 else True


def test_fingerprints_are_stable_across_instances():
    """Two processes must agree, or a committed sketch cache is worthless."""
    a = ReflowFingerprint().fingerprint("x", PROSE)
    b = ReflowFingerprint().fingerprint("x", PROSE)
    assert a.normalized_sha256 == b.normalized_sha256
    assert list(a.channels["word"].minhash) == list(b.channels["word"].minhash)


def test_cjk_text_is_tokenised_rather_than_swallowed(rf):
    text = "この文書は2019年3月に公開された記録である。ベルガモの死亡登録簿は1,247件の超過死亡を記録した。" * 3
    fp = rf.fingerprint("ja", text)
    assert fp.n_tokens > 20, "CJK collapsed into too few tokens"
    assert fp.channels["word"].cardinality > 10
