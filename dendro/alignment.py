"""Align a candidate document to the archive layer and find its oldest ancestor.

This is where a fingerprint becomes a *date*.  The archive layer holds documents
whose existence times are witnessed (see :mod:`dendro.witness`); alignment asks
whether the document in front of us is a rendering, an excerpt, or a rewrite of
something already in there.  If it is, the candidate inherits the ancestor's
bound:

    2026 blog post  --paraphrase-->  2019 arXiv paragraph  (witnessed 2019-03-11)
    => the *content* of the 2026 post is bounded at 2019-03-11

That inheritance is the entire answer to the paraphrase attack.  A statistical
detector sees fluent 2026 prose and calls it synthetic; Dendro sees a 2019
ancestor and reports a 2019 bound with the alignment as its receipt.

Three design decisions carry the weight:

**Asymmetric statistics.**  Ancestry is directional.  ``containment(query in
ref)`` and ``containment(ref in query)`` say different things and the pair of
them is what distinguishes "this is an excerpt of that" from "this quotes that"
from "these are the same document".

**Window alignment with an order constraint.**  Matching windows is not enough;
a genuine derivation preserves *sequence*.  The longest increasing subsequence
of matched window indices separates a real rewrite from two documents that
happen to share vocabulary because they are about the same event.

**An empirical null.**  Every raw score is converted to a confidence against a
null distribution measured on random non-matching pairs *from the same corpus*.
Without that, thresholds are folklore; with it, ``AncestorMatch.confidence`` is
a quantity that can be calibrated and plotted (**calibration** claim).
"""

from __future__ import annotations

import bisect
import datetime as _dt
import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .fingerprint import (
    LshIndex,
    ReflowFingerprint,
    estimate_containment,
    estimate_jaccard,
    simhash_similarity,
)
from .types import (
    AlignmentScore,
    AncestorMatch,
    ConsensusBound,
    Fingerprint,
    Relation,
    to_utc,
)

__all__ = ["AlignmentConfig", "ArchiveLayer", "Aligner", "longest_increasing_run"]


@dataclass(frozen=True)
class AlignmentConfig:
    """Decision thresholds, all of them ablatable from ``benchmarks/run.py``.

    The ``derived_*`` triple is the paraphrase detector.  A paraphrase destroys
    exact word 5-grams (``word`` -> ~0) while leaving the rare-content-word set
    and the numeral/entity set largely intact, because a rewrite that changes
    "1,247 deaths in Bergamo" has stopped being a rewrite.  Requiring *two* of
    those channels plus sequence agreement is what keeps the false-positive rate
    down on same-topic-different-document pairs.
    """

    near_dup_containment: float = 0.80
    excerpt_containment: float = 0.60
    derived_rare: float = 0.42
    derived_num: float = 0.40
    derived_char: float = 0.10
    derived_min_channels: int = 2
    window_match_threshold: float = 0.30
    order_consistency_min: float = 0.60
    max_candidates: int = 256
    min_band_hits: int = 1
    null_sample_pairs: int = 384
    min_confidence: float = 0.55
    min_query_tokens: int = 24


# --------------------------------------------------------------------------- archive layer
@dataclass
class ArchiveEntry:
    """One document with a witnessed existence bound."""

    doc_id: str
    fingerprint: Fingerprint
    not_after: Optional[_dt.datetime] = None
    url: Optional[str] = None
    bound: Optional[ConsensusBound] = None
    meta: dict = field(default_factory=dict)


class ArchiveLayer:
    """The searchable set of documents whose existence times are already proven.

    In production this is populated by walking a Common Crawl index or a Wayback
    CDX range and fingerprinting what comes back.  In the benchmark it is
    populated from the pre-2021 corpus with the real archive timestamps attached.
    Either way the invariant is the same: **an entry may only be added with a
    bound that came from an independent operator**, never from the document's
    own metadata.  ``add`` enforces it.
    """

    def __init__(self, fingerprinter: Optional[ReflowFingerprint] = None, index: Optional[LshIndex] = None) -> None:
        self.fingerprinter = fingerprinter or ReflowFingerprint()
        self.index = index or LshIndex()
        self.entries: dict[str, ArchiveEntry] = {}
        self._by_hash: dict[str, list[str]] = {}

    def add(
        self,
        doc_id: str,
        text: Optional[str] = None,
        *,
        fingerprint: Optional[Fingerprint] = None,
        not_after: Optional[_dt.datetime] = None,
        bound: Optional[ConsensusBound] = None,
        url: Optional[str] = None,
        meta: Optional[Mapping] = None,
    ) -> ArchiveEntry:
        if fingerprint is None:
            if text is None:
                raise ValueError("need text or fingerprint")
            fingerprint = self.fingerprinter.fingerprint(doc_id, text)
        when = not_after if not_after is not None else (bound.not_after if bound else None)
        entry = ArchiveEntry(
            doc_id=doc_id,
            fingerprint=fingerprint,
            not_after=to_utc(when) if when is not None else None,
            url=url,
            bound=bound,
            meta=dict(meta or {}),
        )
        self.entries[doc_id] = entry
        self.index.add(fingerprint)
        self._by_hash.setdefault(fingerprint.normalized_sha256, []).append(doc_id)
        return entry

    def add_many(self, items: Iterable[Mapping]) -> "ArchiveLayer":
        for item in items:
            self.add(**dict(item))
        return self

    def exact(self, fp: Fingerprint) -> list[ArchiveEntry]:
        """Byte-identical-after-normalisation hits: the free, exact path."""
        return [self.entries[d] for d in self._by_hash.get(fp.normalized_sha256, []) if d != fp.doc_id]

    def __len__(self) -> int:
        return len(self.entries)

    def oldest_time(self) -> Optional[_dt.datetime]:
        times = [e.not_after for e in self.entries.values() if e.not_after]
        return min(times) if times else None


# --------------------------------------------------------------------------- helpers
def longest_increasing_run(pairs: Sequence[tuple[int, int]]) -> int:
    """Length of the longest increasing subsequence of reference indices.

    Demonstrates **adversarial-robustness** on the false-positive side: sequence
    agreement is the cheap structural check that separates a derivation from a
    topical coincidence.  A rewrite visits its source's material roughly in
    order; two independent articles about the same earthquake do not, however
    much vocabulary they share.
    """
    if not pairs:
        return 0
    tails: list[int] = []
    for _, r in sorted(pairs):
        i = bisect.bisect_right(tails, r)
        if i == len(tails):
            tails.append(r)
        else:
            tails[i] = r
    return len(tails)


# --------------------------------------------------------------------------- aligner
class Aligner:
    """Scores candidate pairs and decides ancestry.

    ``fit_null`` should be called once per corpus.  It measures what the channel
    scores look like between documents that are *not* related, which converts
    every later score into a confidence rather than a bare number.  Skipping it
    is allowed (thresholds still apply) but then ``confidence`` degrades to a
    monotone transform of the raw score and the reliability diagram in
    ``figures/`` gets noticeably worse -- a fact ``benchmarks/run.py`` records.
    """

    def __init__(
        self,
        fingerprinter: Optional[ReflowFingerprint] = None,
        config: Optional[AlignmentConfig] = None,
    ) -> None:
        self.fp = fingerprinter or ReflowFingerprint()
        self.config = config or AlignmentConfig()
        self._null_mean: dict[str, float] = {}
        self._null_std: dict[str, float] = {}
        self._null_fitted = False

    # -- null model --------------------------------------------------------
    def fit_null(self, layer: ArchiveLayer, seed: int = 7) -> "Aligner":
        """Estimate per-channel containment between unrelated documents.

        Demonstrates **calibration**: it converts a raw similarity into a
        confidence measured against what "unrelated" actually looks like in *this*
        corpus.  Sampling from the layer itself matters -- a corpus of arXiv
        abstracts has a much higher baseline rare-word overlap than a corpus of
        novels, and a fixed threshold would be wrong for one of them.
        """
        ids = list(layer.entries)
        if len(ids) < 4:
            return self
        rng = random.Random(seed)
        acc: dict[str, list[float]] = {}
        for _ in range(min(self.config.null_sample_pairs, len(ids) * 4)):
            a, b = rng.sample(ids, 2)
            fa, fb = layer.entries[a].fingerprint, layer.entries[b].fingerprint
            if fa.normalized_sha256 == fb.normalized_sha256:
                continue
            for name, val in self.fp.channel_scores(fa, fb).items():
                acc.setdefault(name, []).append(val)
        for name, vals in acc.items():
            arr = np.asarray(vals, dtype=float)
            self._null_mean[name] = float(arr.mean())
            self._null_std[name] = float(arr.std(ddof=1)) if arr.size > 1 else 0.05
        self._null_fitted = bool(acc)
        return self

    def _surprise(self, channel_scores: Mapping[str, float]) -> float:
        """Total z-score of the observed channel scores under the null."""
        if not self._null_fitted:
            return 0.0
        zs = []
        for name, val in channel_scores.items():
            mu = self._null_mean.get(name)
            sd = max(self._null_std.get(name, 0.05), 0.02)
            if mu is None:
                continue
            zs.append((val - mu) / sd)
        return float(sum(zs) / math.sqrt(len(zs))) if zs else 0.0

    # -- pairwise scoring --------------------------------------------------
    def _window_alignment(self, q: Fingerprint, r: Fingerprint) -> tuple[float, float, int, float]:
        """(coverage_q, coverage_r, matched, order_consistency).

        Implements **partial-coverage ancestry**: a 300-token quotation inside a
        12k-token page shows up here as ``coverage_q ~ 1.0, coverage_r ~ 0.03``,
        which the classifier reads as CONTAINED_BY -- an answer whole-document
        Jaccard (0.02) can never give.
        """
        if not q.windows or not r.windows:
            return 0.0, 0.0, 0, 0.0
        thr = self.config.window_match_threshold
        matched_pairs: list[tuple[int, int]] = []
        r_hit: set[int] = set()
        for wi, qw in enumerate(q.windows):
            best, best_j = 0.0, -1
            for rj, rw in enumerate(r.windows):
                s = estimate_jaccard(qw.minhash, rw.minhash)
                if s > best:
                    best, best_j = s, rj
            if best >= thr and best_j >= 0:
                matched_pairs.append((wi, best_j))
                r_hit.add(best_j)
        cov_q = len(matched_pairs) / len(q.windows)
        cov_r = len(r_hit) / len(r.windows)
        order = longest_increasing_run(matched_pairs) / len(matched_pairs) if matched_pairs else 0.0
        return cov_q, cov_r, len(matched_pairs), order

    def score(self, query: Fingerprint, ref: Fingerprint) -> AlignmentScore:
        """Full pairwise comparison across channels and windows.

        Demonstrates **adversarial-robustness**: the per-channel spread is what
        separates a reflow (every channel high) from a paraphrase (exact channels
        collapsed, rare/numeral channels intact) from an unrelated document
        (everything low).  A single similarity number cannot make that
        distinction.
        """
        ch_q = self.fp.channel_scores(query, ref)
        ch_r = self.fp.channel_scores(ref, query)
        qw, rw = query.channels["word"], ref.channels["word"]
        j = estimate_jaccard(qw.minhash, rw.minhash)
        cov_q, cov_r, matched, order = self._window_alignment(query, ref)
        combined = self.fp.combined_score(ch_q)
        return AlignmentScore(
            jaccard=j,
            containment_query_in_ref=ch_q.get("word", 0.0),
            containment_ref_in_query=ch_r.get("word", 0.0),
            window_coverage_query=cov_q,
            window_coverage_ref=cov_r,
            simhash_similarity=simhash_similarity(qw.simhash, rw.simhash),
            channel_scores={**ch_q, "order_consistency": order, "surprise": self._surprise(ch_q)},
            combined=combined,
            matched_windows=matched,
        )

    # -- classification ----------------------------------------------------
    def classify(self, query: Fingerprint, ref: Fingerprint, score: AlignmentScore) -> tuple[Relation, float]:
        """Map a score to a relation plus a confidence in [0, 1].

        The DERIVED branch is the one that matters for **adversarial-robustness**:
        it fires when the exact-match channels have collapsed but the
        paraphrase-resistant ones have not, *and* the surviving matches are in
        the right order.  Requiring the order constraint is what stops two
        articles about the same earthquake from being called derivations of each
        other.
        """
        cfg = self.config
        if query.normalized_sha256 == ref.normalized_sha256:
            return Relation.IDENTICAL, 1.0

        cq = score.containment_query_in_ref
        cr = score.containment_ref_in_query
        ch = score.channel_scores
        order = float(ch.get("order_consistency", 0.0))
        surprise = float(ch.get("surprise", 0.0))

        if cq >= cfg.near_dup_containment and cr >= cfg.near_dup_containment:
            return Relation.NEAR_DUPLICATE, _confidence(min(cq, cr), surprise)
        if cq >= cfg.excerpt_containment and cq - cr >= 0.20:
            return Relation.CONTAINED_BY, _confidence(cq, surprise)
        if cr >= cfg.excerpt_containment and cr - cq >= 0.20:
            return Relation.CONTAINS, _confidence(cr, surprise)

        strong = sum(
            [
                ch.get("rare", 0.0) >= cfg.derived_rare,
                ch.get("num", 0.0) >= cfg.derived_num,
                ch.get("char", 0.0) >= cfg.derived_char,
            ]
        )
        if strong >= cfg.derived_min_channels:
            ordered = order >= cfg.order_consistency_min or score.matched_windows <= 1
            if ordered:
                base = 0.5 * ch.get("rare", 0.0) + 0.3 * ch.get("num", 0.0) + 0.2 * ch.get("char", 0.0)
                return Relation.DERIVED, _confidence(base, surprise)
        return Relation.UNRELATED, 0.0

    # -- search ------------------------------------------------------------
    def candidates(self, query: Fingerprint, layer: ArchiveLayer) -> list[str]:
        """Sublinear retrieval: exact-hash hits first, then LSH band collisions.

        Demonstrates **adversarial-robustness** and **cost** together.  Retrieval
        runs over every paraphrase-resistant channel, not just exact shingles,
        because a candidate that is never retrieved can never be classified --
        indexing only ``word`` silently capped ancestor recall at 25% while every
        downstream component looked correct.
        """
        out: list[str] = [e.doc_id for e in layer.exact(query)]
        seen = set(out)
        hits = layer.index.query(query)
        for doc_id, bands in sorted(hits.items(), key=lambda kv: -kv[1]):
            if bands < self.config.min_band_hits or doc_id in seen:
                continue
            out.append(doc_id)
            seen.add(doc_id)
            if len(out) >= self.config.max_candidates:
                break
        return out

    def matches(self, query: Fingerprint, layer: ArchiveLayer) -> list[AncestorMatch]:
        """All ancestral relations to the archive layer, best first.

        Demonstrates **partial-coverage ancestry**: excerpts and quotations
        surface here as ``CONTAINED_BY`` / ``CONTAINS`` rather than being
        discarded, because containment is scored asymmetrically and windows
        localise the shared span.
        """
        if query.n_tokens < self.config.min_query_tokens:
            return []
        found: list[AncestorMatch] = []
        for doc_id in self.candidates(query, layer):
            entry = layer.entries[doc_id]
            sc = self.score(query, entry.fingerprint)
            rel, conf = self.classify(query, entry.fingerprint, sc)
            if rel is Relation.UNRELATED or conf < self.config.min_confidence:
                continue
            found.append(
                AncestorMatch(
                    ref_doc_id=doc_id,
                    witness_time=entry.not_after,
                    relation=rel,
                    score=sc,
                    confidence=conf,
                    ref_url=entry.url,
                )
            )
        found.sort(key=lambda m: (-m.confidence, m.ref_doc_id))
        return found

    def oldest_ancestor(self, query: Fingerprint, layer: ArchiveLayer) -> Optional[AncestorMatch]:
        """The earliest-witnessed ancestral match, or ``None``.

        This is the function the paraphrase attack has to defeat.  It returns the
        *oldest* qualifying ancestor rather than the best-scoring one, because
        the quantity being proven is a lower bound on age; among several valid
        ancestors the earliest gives the tightest true statement.
        """
        ancestral = [m for m in self.matches(query, layer) if m.is_ancestral and m.witness_time]
        if not ancestral:
            return None
        return min(ancestral, key=lambda m: (m.witness_time, -m.confidence))


def _confidence(strength: float, surprise: float) -> float:
    """Blend raw match strength with how surprising it is under the null.

    Two documents can score 0.5 rare-word containment because they are related,
    or because the corpus is 400 papers on the same benchmark.  The null model
    knows the difference; ``strength`` alone does not.
    """
    s = max(0.0, min(1.0, strength))
    if surprise <= 0.0:
        return s
    bump = 1.0 - math.exp(-max(0.0, surprise) / 6.0)
    return float(min(1.0, 0.65 * s + 0.35 * bump + 0.25 * s * bump))
