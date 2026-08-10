"""Paraphrased documents must still align to their archived ancestor.

This is the paraphrase attack, tested directly on **real arXiv abstracts**:
take a document that is in the archive layer with a genuine 2019/2020
registration date, rewrite it hard enough that exact word 5-grams are gone, and
require that :meth:`~dendro.alignment.Aligner.oldest_ancestor` still finds the
original and hands back its date.

The tests also pin the *negative* direction, which matters more.  An aligner that
found ancestors everywhere would pass the robustness test and be useless: it
would attach 2019 dates to documents written yesterday.  So the suite asserts
both that rewrites are found and that unrelated documents are not.
"""

from __future__ import annotations

import random

import pytest

from benchmarks.generators import Paraphraser
from dendro.alignment import Aligner, AlignmentConfig, ArchiveLayer, longest_increasing_run
from dendro.fingerprint import ReflowFingerprint
from dendro.types import Relation


def test_reflowed_copy_is_found_as_identical(archive_layer, small_corpus):
    layer, aligner, rf = archive_layer
    source = small_corpus.archive[0]
    html = f"<html><body><nav>Home | About</nav><p>{source['text']}</p>" \
           f"<footer>&copy; 2019 Example</footer></body></html>"
    match = aligner.oldest_ancestor(rf.fingerprint("q", html), layer)
    assert match is not None, "a reflowed copy was not recognised"
    assert match.relation in (Relation.IDENTICAL, Relation.NEAR_DUPLICATE)
    assert match.ref_doc_id == source["doc_id"]


@pytest.mark.parametrize("strength", [0.35, 0.55, 0.75])
def test_paraphrase_still_aligns_to_its_ancestor(archive_layer, small_corpus, strength):
    """The headline robustness property, swept over attack strength.

    Recall is required to stay high rather than perfect: paraphrase is a real
    attack and some documents do fall out of reach.  What must not happen is a
    collapse, and what must *never* happen is the failure mode being a confident
    wrong answer rather than a miss.
    """
    layer, aligner, rf = archive_layer
    sources = small_corpus.archive[:40]
    found = 0
    wrong = 0
    for i, src in enumerate(sources):
        para = Paraphraser(strength=strength, seed=i).paraphrase(src["text"])
        match = aligner.oldest_ancestor(rf.fingerprint(f"para{i}", para), layer)
        if match is None:
            continue
        if match.ref_doc_id == src["doc_id"]:
            found += 1
        else:
            wrong += 1

    recall = found / len(sources)
    assert recall >= 0.70, f"ancestor recall collapsed to {recall:.2f} at strength {strength}"
    assert wrong <= 2, f"{wrong} paraphrases aligned to the *wrong* ancestor"


def test_paraphrase_destroys_exact_ngrams(small_corpus):
    """Confirms the attack is real, so the test above is not vacuous.

    An earlier paraphraser only substituted synonyms and left word-channel
    containment above 0.5; robustness measured against it would have been an
    artefact.  This asserts the attack actually removes the exact-match signal.
    """
    rf = ReflowFingerprint()
    scores = []
    for i, src in enumerate(small_corpus.archive[:25]):
        para = Paraphraser(strength=0.55, seed=i).paraphrase(src["text"])
        scores.append(rf.channel_scores(rf.fingerprint("q", para), rf.fingerprint("r", src["text"]))["word"])
    mean_word = sum(scores) / len(scores)
    assert mean_word < 0.35, f"paraphrase left word-channel containment at {mean_word:.2f}"


def test_paraphrase_inherits_the_ancestor_date(archive_layer, small_corpus):
    """The whole point: a rewrite carries its source's proven existence date."""
    layer, aligner, rf = archive_layer
    src = small_corpus.archive[3]
    para = Paraphraser(strength=0.55, seed=99).paraphrase(src["text"])
    match = aligner.oldest_ancestor(rf.fingerprint("q", para), layer)
    assert match is not None
    assert match.witness_time is not None
    assert match.witness_time.year <= 2020


def test_unrelated_documents_are_not_claimed_as_ancestors(archive_layer, recent_documents):
    """The false-positive direction, which is the dangerous one."""
    layer, aligner, rf = archive_layer
    claimed = 0
    for i, doc in enumerate(recent_documents[:60]):
        match = aligner.oldest_ancestor(rf.fingerprint(f"new{i}", doc["text"]), layer)
        if match is not None:
            claimed += 1
    rate = claimed / 60
    assert rate <= 0.05, f"{rate:.0%} of unrelated recent documents were given a false ancestor"


def test_synthetic_documents_do_not_get_false_ancestors(archive_layer, small_corpus):
    """Generated text must not align to the archive layer.

    If it did, the benchmark would be measuring leakage between the generator's
    training split and the archive split rather than provenance.
    """
    layer, aligner, rf = archive_layer
    claimed = sum(
        1
        for i, d in enumerate(small_corpus.synthetic[:60])
        if aligner.oldest_ancestor(rf.fingerprint(f"syn{i}", d["text"]), layer) is not None
    )
    assert claimed <= 3, f"{claimed}/60 synthetic documents matched an archived ancestor"


def test_excerpt_is_recognised_as_contained(archive_layer, small_corpus):
    """Partial coverage: a fragment must resolve to the document it came from."""
    layer, aligner, rf = archive_layer
    src = small_corpus.archive[7]
    words = src["text"].split()
    excerpt = " ".join(words[: max(45, len(words) // 3)])
    matches = aligner.matches(rf.fingerprint("frag", excerpt), layer)
    assert matches, "an excerpt found no match at all"
    assert matches[0].ref_doc_id == src["doc_id"]
    assert matches[0].relation in (Relation.CONTAINED_BY, Relation.NEAR_DUPLICATE, Relation.IDENTICAL)


def test_longest_increasing_run():
    assert longest_increasing_run([]) == 0
    assert longest_increasing_run([(0, 1), (1, 2), (2, 3)]) == 3
    assert longest_increasing_run([(0, 3), (1, 2), (2, 1)]) == 1
    assert longest_increasing_run([(0, 1), (1, 5), (2, 2), (3, 3)]) == 3


def test_null_model_makes_confidence_corpus_relative(archive_layer):
    """Without a fitted null, confidence is a bare score; with one it is calibrated."""
    layer, _, rf = archive_layer
    naive = Aligner(rf)
    fitted = Aligner(rf).fit_null(layer)
    assert fitted._null_fitted
    assert not naive._null_fitted
    assert set(fitted._null_mean) >= {"word", "rare", "num"}
    # A corpus of same-domain abstracts has a non-trivial baseline overlap; if it
    # were zero the null model would be pointless.
    assert fitted._null_mean["rare"] > 0.0
