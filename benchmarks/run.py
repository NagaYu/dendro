"""The benchmark: three conditions, five evaluation axes, everything measured.

Conditions
    **(A)** statistical / perplexity-family detector -- unsupervised, reads prose.
    **(B)** learned classifier -- supervised on classes (i) and (iii), reads prose.
    **(C)** Dendro -- archival evidence, reads nothing but metadata and fingerprints.

Axes
    1. discrimination in the ordinary condition (AUC);
    2. robustness under attack -- paraphrase sweep and backdate forgery;
    3. generalisation to generator generations not seen in training;
    4. calibration (reliability, ECE, Brier);
    5. cost per document and cache hit rate.

Two tasks are reported, not one, because collapsing them would overstate the
result:

``lowbackground``
    class (i) pre-2021 human vs class (iii) synthetic.  The task Dendro is built
    for, and the one where an existence bound is decisive.
``all_eras``
    classes (i)+(ii) human vs class (iii) synthetic.  Dendro is *weaker* here and
    the table says so plainly: proving a document existed in 2025 is not evidence
    that a human wrote it, so Dendro abstains on most of class (ii).  The
    selective-prediction curve is where that abstention is shown to be worth
    something rather than a dodge.

Run::

    python -m benchmarks.run                 # full, writes results/*.csv
    python -m benchmarks.run --quick         # smaller, for a smoke test
    python -m benchmarks.run --offline       # cache only, no network
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dendro.fingerprint import FingerprintConfig, NormalizationConfig, ReflowFingerprint
from dendro.propagate import (
    Calibrator,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)

from .baselines import LearnedDetector, StatisticalDetector, auc, risk_coverage
from .corpus import EvalCorpus, archive_entries, build_corpus, paraphrase_sweep
from .dendro_scorer import DendroScorer
from .generators import GENERATION_LADDER, NgramLM, Paraphraser

RESULTS = REPO / "results"
PARAPHRASE_STRENGTHS = (0.20, 0.35, 0.55, 0.75, 0.95)
SCRAMBLE_LEVELS = (0.0, 0.3, 0.6)


# --------------------------------------------------------------------------- io
def write_csv(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})
    print(f"  wrote {path.relative_to(REPO)}  ({len(rows)} rows)")


# --------------------------------------------------------------------------- harness
class Benchmark:
    """Holds the fitted detectors and the cached per-document scores."""

    def __init__(self, corpus: EvalCorpus, offline: bool = False, seed: int = 11) -> None:
        self.corpus = corpus
        self.rng = random.Random(seed)
        self.seed = seed

        # ---- splits ----
        # Three-way, not two-way.  Model *fitting* and probability *calibration*
        # need different data, and conflating them produced a genuinely harmful
        # bug: with a calibration set containing only pre-2021 human documents and
        # synthetic ones, the isotonic map had never seen a *recent human*
        # document and extrapolated them straight into the accusatory region --
        # Dendro reported P(human) ~ 0.03 for real 2025 abstracts. Calibration data
        # must cover the deployment distribution, and it is shared across all
        # three methods so the comparison stays fair.
        old = list(corpus.human_old)
        self.rng.shuffle(old)
        a, b = int(0.40 * len(old)), int(0.60 * len(old))
        self.old_train, self.old_cal, self.old_test = old[:a], old[a:b], old[b:]

        recent = list(corpus.human_recent)
        self.rng.shuffle(recent)
        cut = int(0.25 * len(recent))
        self.recent_cal, self.recent_test = recent[:cut], recent[cut:]

        by_gen: dict[int, list[dict]] = {}
        for d in corpus.synthetic:
            by_gen.setdefault(d["generation"], []).append(d)
        self.syn_train: list[dict] = []
        self.syn_cal: list[dict] = []
        self.syn_test_by_gen: dict[int, list[dict]] = {}
        for g, docs in sorted(by_gen.items()):
            docs = list(docs)
            self.rng.shuffle(docs)
            a, b = int(0.40 * len(docs)), int(0.60 * len(docs))
            # (B)'s classifier only ever sees generations 1-2. Everything else is
            # held out entirely, which is what axis (3) measures.
            if g <= 2:
                self.syn_train.extend(docs[:a])
            self.syn_cal.extend(docs[a:b] if g <= 2 else [])
            self.syn_test_by_gen[g] = docs[b:]
        self.syn_test = [d for docs in self.syn_test_by_gen.values() for d in docs]

        self.train_docs = [*self.old_train, *self.syn_train]
        self.train_labels = [int(d["label_human"]) for d in self.train_docs]
        self.cal_docs = [*self.old_cal, *self.recent_cal, *self.syn_cal]
        self.cal_labels = [int(d["label_human"]) for d in self.cal_docs]

        # ---- (A) ----
        t0 = time.perf_counter()
        self.stat = StatisticalDetector(variant="combined")
        self.stat.fit_reference([d["text"] for d in corpus.reference])
        self.stat.fit_calibration([d["text"] for d in self.cal_docs], self.cal_labels)
        self.t_stat_fit = time.perf_counter() - t0

        # ---- (B) ----
        t0 = time.perf_counter()
        self.learned = LearnedDetector(seed=seed).fit(
            [d["text"] for d in self.train_docs], self.train_labels
        )
        self.learned.fit_calibration([d["text"] for d in self.cal_docs], self.cal_labels)
        self.t_learned_fit = time.perf_counter() - t0

        # ---- (C) ----
        t0 = time.perf_counter()
        self.dendro = DendroScorer(archive_entries(corpus), offline=True if offline else None)
        self.dendro.fit_calibration(self.cal_docs)
        self.t_dendro_fit = time.perf_counter() - t0

        self._cache: dict[str, dict[str, tuple[float, float]]] = {"A": {}, "B": {}, "C": {}}
        self._verdicts: dict[str, Any] = {}

    # -- scoring -----------------------------------------------------------
    def scores(self, method: str, docs: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
        """(probability that the document is human, confidence) for one method."""
        cache = self._cache[method]
        todo = [d for d in docs if d["doc_id"] not in cache]
        if todo:
            if method == "A":
                ps = self.stat.predict_proba([d["text"] for d in todo])
                for d, p in zip(todo, ps):
                    cache[d["doc_id"]] = (float(p), abs(float(p) - 0.5) * 2.0)
            elif method == "B":
                ps = self.learned.predict_proba([d["text"] for d in todo])
                for d, p in zip(todo, ps):
                    cache[d["doc_id"]] = (float(p), abs(float(p) - 0.5) * 2.0)
            else:
                for d in todo:
                    v = self.dendro.score(d)
                    self._verdicts[d["doc_id"]] = v
                    cache[d["doc_id"]] = (float(v.human_origin_p), float(1.0 - (v.ci_high - v.ci_low)))
        p = np.array([cache[d["doc_id"]][0] for d in docs], dtype=float)
        c = np.array([cache[d["doc_id"]][1] for d in docs], dtype=float)
        return p, c

    def evaluate(self, name: str, positives: Sequence[Mapping], negatives: Sequence[Mapping]) -> list[dict]:
        docs = [*positives, *negatives]
        labels = [1] * len(positives) + [0] * len(negatives)
        rows: list[dict] = []
        for method, label in (("A", "statistical"), ("B", "learned"), ("C", "dendro")):
            p, conf = self.scores(method, docs)
            rows.append(
                {
                    "condition": name,
                    "method": method,
                    "method_name": label,
                    "n_pos": len(positives),
                    "n_neg": len(negatives),
                    "auc": round(auc(p, labels), 4),
                    "ece": round(expected_calibration_error(p, labels), 4),
                    "brier": round(brier_score(p, labels), 4),
                    "mean_p_pos": round(float(p[: len(positives)].mean()), 4),
                    "mean_p_neg": round(float(p[len(positives) :].mean()), 4),
                    "abstain_rate": round(self._abstain_rate(method, docs), 4),
                    # Split by class, because a pooled abstain rate is
                    # uninterpretable: on a set that is 80% synthetic, "83%
                    # abstained" is dominated by the negatives correctly
                    # declining to answer and says nothing about the positives.
                    "abstain_rate_pos": round(self._abstain_rate(method, positives), 4),
                    "abstain_rate_neg": round(self._abstain_rate(method, negatives), 4),
                }
            )
        return rows

    def backdate_flag_rate(self, docs: Sequence[Mapping]) -> float:
        """Fraction of documents Dendro explicitly flags as backdated.

        Measured on forgeries this is detection; measured on genuine pre-2021
        documents it is the false-accusation rate, and both belong in the table.
        A detector that flags everything would score perfectly on the first and
        is worthless -- which is exactly why the second column exists.
        """
        self.scores("C", docs)
        hits = 0
        for d in docs:
            v = self._verdicts.get(d["doc_id"])
            if v and any(f.kind in ("backdate", "commit_predates_repo") for f in v.flags):
                hits += 1
        return hits / max(1, len(docs))

    def _abstain_rate(self, method: str, docs: Sequence[Mapping]) -> float:
        if method != "C":
            return 0.0  # (A) and (B) always answer -- that is the point of the axis
        vs = [self._verdicts.get(d["doc_id"]) for d in docs]
        return float(np.mean([bool(v and v.abstained) for v in vs]))


# --------------------------------------------------------------------------- axes
def axis_main(bench: Benchmark) -> list[dict]:
    """Axis (1): ordinary-condition discrimination, including the honest control.

    ``recent_only`` flatters Dendro for a reason that has nothing to do with the
    method: every class (ii) document is an arXiv paper and therefore carries a
    registration witness, while every class (iii) document carries none *by
    construction*.  "Has any archival record at all" then separates the classes
    perfectly, and a 1.0 there would be a property of how the corpus was built.

    ``recent_unwitnessed`` removes the artefact by stripping class (ii) of its
    witnesses -- a recent human document that nobody archived, which is the
    common case in the wild.  Dendro should and does fall to chance there, and
    abstain on nearly all of it.  That number is the honest characterisation of
    what evidence-based dating cannot do, and it belongs next to the ones that
    look good.
    """
    c = bench.corpus
    rows: list[dict] = []
    rows += bench.evaluate("lowbackground", bench.old_test, bench.syn_test)
    rows += bench.evaluate("all_eras", [*bench.old_test, *bench.recent_test], bench.syn_test)
    rows += bench.evaluate("recent_only", bench.recent_test, bench.syn_test)

    unwitnessed = [
        {**d, "doc_id": f"{d['doc_id']}:unwitnessed", "witnesses": [], "url": None, "claimed_date": None}
        for d in bench.recent_test[:250]
    ]
    rows += bench.evaluate("recent_unwitnessed", unwitnessed, bench.syn_test)
    return rows


def axis_robustness(bench: Benchmark, sweep: Mapping[float, Sequence[Mapping]]) -> list[dict]:
    """Axis (2): how far performance falls under paraphrase and backdating.

    The paraphrase condition is scored as *human-derived* positives against
    synthetic negatives.  A method that calls a rewritten 2019 abstract synthetic
    has made a false accusation about human writing, and that is the error this
    condition is designed to expose.
    """
    c = bench.corpus
    rows: list[dict] = []
    baseline = {r["method"]: r["auc"] for r in bench.evaluate("lowbackground", bench.old_test, bench.syn_test)}

    for strength, docs in sorted(sweep.items()):
        for row in bench.evaluate(f"paraphrase@{strength:.2f}", docs, bench.syn_test):
            row["attack"] = "paraphrase"
            row["attack_strength"] = strength
            row["auc_drop_vs_clean"] = round(baseline[row["method"]] - row["auc"], 4)
            rows.append(row)

    if c.backdated:
        # AUC alone would flatter Dendro here: class (i) has strong witnesses and
        # class (iv-b) has none, so the two separate even if the forged date is
        # never noticed at all.  The quantities that actually speak to the claim
        # are the flag rates -- how often a forgery is *identified as* a forgery,
        # and how often a genuine old document is wrongly accused.
        flag_rate = bench.backdate_flag_rate(c.backdated)
        false_flag = bench.backdate_flag_rate(bench.old_test)
        for row in bench.evaluate("backdate", bench.old_test, c.backdated):
            row["attack"] = "backdate"
            row["attack_strength"] = 1.0
            row["auc_drop_vs_clean"] = round(baseline[row["method"]] - row["auc"], 4)
            # (A) and (B) read prose only; the forged metadata is invisible to
            # them by construction, so their flag rate is exactly zero.
            row["backdate_flag_rate"] = round(flag_rate, 4) if row["method"] == "C" else 0.0
            row["false_flag_rate"] = round(false_flag, 4) if row["method"] == "C" else 0.0
            rows.append(row)
    return rows


def axis_generalization(bench: Benchmark) -> list[dict]:
    """Axis (3): performance per generator generation, seen and unseen.

    (B) was trained on generations 1-2 only.  (A) is unsupervised but its signal
    shrinks as the generator improves.  (C) never reads the text, so it has no
    mechanism by which the generation could matter -- the flat line is the
    prediction, and this is the measurement of it.
    """
    ladder = {d["generation"]: d for d in bench.corpus.ladder}
    rows: list[dict] = []

    groups: list[tuple[int, list[dict]]] = sorted(bench.syn_test_by_gen.items())
    by_unseen: dict[int, list[dict]] = {}
    for d in bench.corpus.unseen_family:
        by_unseen.setdefault(d["generation"], []).append(d)
    groups += sorted(by_unseen.items())

    for g, docs in groups:
        if not docs:
            continue
        info = ladder.get(g, {})
        for row in bench.evaluate(f"generation-{g}", bench.old_test, docs):
            row["generation"] = g
            row["generator"] = info.get("generator", f"gen{g}")
            row["family"] = info.get("family", "temperature")
            row["coherence"] = info.get("coherence")
            row["logloss_gap"] = round(float(info.get("logloss_gap", float("nan"))), 4)
            row["burstiness_gap"] = round(float(info.get("burstiness_gap", float("nan"))), 4)
            row["seen_in_training"] = g <= 2
            row["seen_family"] = info.get("family", "temperature") == "temperature"
            rows.append(row)
    return rows


def axis_calibration(bench: Benchmark) -> list[dict]:
    """Axis (4): reliability diagram bins for each method on the pooled test set."""
    docs = [*bench.old_test, *bench.corpus.human_recent, *bench.syn_test]
    labels = [int(d["label_human"]) for d in docs]
    rows: list[dict] = []
    for method, name in (("A", "statistical"), ("B", "learned"), ("C", "dendro")):
        p, _ = bench.scores(method, docs)
        for b in reliability_bins(p, labels, n_bins=10):
            rows.append({"method": method, "method_name": name, **b})
    return rows


def axis_selective(bench: Benchmark) -> list[dict]:
    """Axis (2b): risk-coverage. Where abstention earns its keep."""
    docs = [*bench.old_test, *bench.corpus.human_recent, *bench.syn_test]
    labels = [int(d["label_human"]) for d in docs]
    rows: list[dict] = []
    for method, name in (("A", "statistical"), ("B", "learned"), ("C", "dendro")):
        p, conf = bench.scores(method, docs)
        for point in risk_coverage(p, labels, conf):
            rows.append({"method": method, "method_name": name, **point})
    return rows


def axis_prevalence_sweep(bench: Benchmark, corpus: EvalCorpus) -> list[dict]:
    """How much of the answer rests on the assumed prevalence curve.

    The curve -- the machine-generated share of new text over time -- is the one
    genuinely unfalsifiable input to the whole method, so it deserves a
    sensitivity analysis rather than a footnote.  Re-scoring under alternative
    curves separates what is assumed from what is measured: the *ranking* of
    methods is a property of the evidence and does not move, while the absolute
    probabilities do.  A reader who disagrees with the default curve can find
    their own row here.
    """
    from dendro.propagate import ContaminationPropagator, PrevalenceCurve, PropagationConfig

    curves = {
        "default (mid 2023.4)": PrevalenceCurve(),
        "early onset (mid 2022.5)": PrevalenceCurve(midpoint_year=2022.5),
        "late onset (mid 2024.5)": PrevalenceCurve(midpoint_year=2024.5),
        "low ceiling (0.20)": PrevalenceCurve(ceiling=0.20),
        "high ceiling (0.70)": PrevalenceCurve(ceiling=0.70),
        "shallow (steepness 0.6)": PrevalenceCurve(steepness=0.6),
    }
    docs = [*bench.old_test, *bench.syn_test]
    labels = [1] * len(bench.old_test) + [0] * len(bench.syn_test)
    recent = bench.recent_test[:200]

    rows: list[dict] = []
    original = bench.dendro.propagator
    saved_cache = dict(bench._cache["C"])
    saved_verdicts = dict(bench._verdicts)
    try:
        for name, curve in curves.items():
            # A *fresh* calibrator per curve, deliberately.  Reusing the fitted one
            # would apply a map learned for a different score distribution and
            # produce nonsense -- an earlier version did exactly that and reported
            # AUC 0.500 under a low-ceiling curve, which is an artefact of the
            # mismatched map rather than a property of the curve.  These rows show
            # the *raw* inference so the curve's effect is legible on its own.
            prop = ContaminationPropagator(PropagationConfig(prevalence=curve))
            bench.dendro.propagator = prop
            ps = [bench.dendro.score(d).human_origin_p for d in docs]
            rec = [bench.dendro.score(d) for d in recent]
            rows.append(
                {
                    "curve": name,
                    "midpoint_year": curve.midpoint_year,
                    "ceiling": curve.ceiling,
                    "steepness": curve.steepness,
                    "auc_lowbackground": round(auc(ps, labels), 4),
                    "mean_p_pre2021": round(float(np.mean(ps[: len(bench.old_test)])), 4),
                    "mean_p_synthetic": round(float(np.mean(ps[len(bench.old_test) :])), 4),
                    "mean_p_recent_human": round(float(np.mean([v.human_origin_p for v in rec])), 4),
                    "abstain_rate_recent": round(float(np.mean([v.abstained for v in rec])), 4),
                }
            )
    finally:
        # Restore rather than clear: ``axis_cost`` divides wall time by the number
        # of documents scored, and clearing made that denominator 1.
        bench.dendro.propagator = original
        bench._cache["C"] = saved_cache
        bench._verdicts.clear()
        bench._verdicts.update(saved_verdicts)
    return rows


def axis_cost(bench: Benchmark, wall: Mapping[str, float]) -> list[dict]:
    """Axis (5): per-document cost and cache hit rate."""
    stats = bench.dendro.stats
    n_docs = len(bench._cache["C"]) or 1
    return [
        {
            "method": "C",
            "method_name": "dendro",
            "documents_scored": n_docs,
            **stats.as_row(),
            "network_calls_per_doc": round(stats.network_calls / n_docs, 4),
            "seconds_per_doc": round(wall.get("C", 0.0) / n_docs, 4),
            "fit_seconds": round(bench.t_dendro_fit, 3),
            "coverage_probe_cache_entries": len(bench.dendro._coverage_cache),
        },
        {
            "method": "A",
            "method_name": "statistical",
            "documents_scored": len(bench._cache["A"]),
            "requests": 0,
            "hits": 0,
            "hit_rate": "",
            "network_calls": 0,
            "seconds_per_doc": round(wall.get("A", 0.0) / max(1, len(bench._cache["A"])), 4),
            "fit_seconds": round(bench.t_stat_fit, 3),
        },
        {
            "method": "B",
            "method_name": "learned",
            "documents_scored": len(bench._cache["B"]),
            "requests": 0,
            "hits": 0,
            "hit_rate": "",
            "network_calls": 0,
            "seconds_per_doc": round(wall.get("B", 0.0) / max(1, len(bench._cache["B"])), 4),
            "fit_seconds": round(bench.t_learned_fit, 3),
        },
    ]


def render_as_web_page(text: str, seed: int = 0) -> str:
    """Wrap plain prose in the chrome a real archive capture would carry.

    The normalisation stages exist to survive *renderings*, so measuring them on
    plain-text abstracts measures nothing -- an earlier version of this ablation
    did exactly that and reported every stage as worth ~0.03, because there was
    no markup to strip. Wrapping the text first is what makes the stage-by-stage
    ablation informative.
    """
    rng = random.Random(seed)
    lines = text.splitlines()
    body = "\n".join(
        f"<p>{l}</p>" if rng.random() < 0.7 else f"<div>{l}<br></div>" for l in lines if l.strip()
    )
    # Everything added here is *chrome* -- navigation, banners, share widgets,
    # a copyright footer -- never new prose.  An earlier version also injected an
    # "<h2>Abstract</h2>" heading, which is genuine added content, so the
    # hash-equality test could never pass for any configuration and the column
    # read 0% across the board: the test was measuring its own fixture rather
    # than the normaliser.
    return (
        "<html><head><title>Archived copy</title>"
        "<style>.nav{color:#333}</style><script>var t=1;</script></head><body>\n"
        "<nav>Home | About | Publications | Contact | Search</nav>\n"
        '<div class="cookie">We use cookies to improve your experience. Accept all cookies</div>\n'
        f"{body}\n"
        "<div>Share this &middot; Tweet &middot; Print</div>\n"
        "<footer>&copy; 2019 Example Institute. All rights reserved.</footer>\n"
        "</body></html>"
    )


def axis_ablation(corpus: EvalCorpus, n: int = 45, strength: float = 0.55) -> list[dict]:
    """What each normalisation stage and each channel actually buys.

    Two different questions, measured separately because the stages serve them
    differently:

    ``reflow``
        does a *rendered* copy of a document still hash to the same value? This
        is what the markup/boilerplate/whitespace stages exist for, so it is
        measured against HTML-wrapped text rather than plain prose.
    ``paraphrase``
        does the channel survive a rewrite, and does it stay *low* on unrelated
        documents? The second half matters: a channel that scores high on
        everything is a constant, not evidence.
    """
    docs = corpus.archive[:n]
    rng = random.Random(3)
    rows: list[dict] = []

    variants = {
        "full": NormalizationConfig(),
        "no_boilerplate_strip": NormalizationConfig(strip_boilerplate=False),
        "no_markup_strip": NormalizationConfig(strip_markup=False),
        "no_punctuation_fold": NormalizationConfig(fold_punctuation=False),
        "no_whitespace_collapse": NormalizationConfig(collapse_whitespace=False),
        "no_url_strip": NormalizationConfig(strip_urls=False),
    }
    for vname, cfg in variants.items():
        rf = ReflowFingerprint(normalization=cfg)
        matched: dict[str, list[float]] = {}
        random_pairs: dict[str, list[float]] = {}
        reflow: dict[str, list[float]] = {}
        hash_hits = 0
        for i, d in enumerate(docs):
            source = rf.fingerprint("r", d["text"])
            rendered = rf.fingerprint("html", render_as_web_page(d["text"], seed=i))
            para = rf.fingerprint("q", Paraphraser(strength=strength, seed=i).paraphrase(d["text"]))
            other = rf.fingerprint("o", docs[rng.randrange(len(docs))]["text"])

            hash_hits += int(rendered.normalized_sha256 == source.normalized_sha256)
            for k, v in rf.channel_scores(rendered, source).items():
                reflow.setdefault(k, []).append(v)
            for k, v in rf.channel_scores(para, source).items():
                matched.setdefault(k, []).append(v)
            for k, v in rf.channel_scores(para, other).items():
                random_pairs.setdefault(k, []).append(v)

        for ch in sorted(matched):
            m = float(np.mean(matched[ch]))
            b = float(np.mean(random_pairs[ch]))
            rows.append(
                {
                    "variant": vname,
                    "channel": ch,
                    "reflow_containment": round(float(np.mean(reflow[ch])), 4),
                    "reflow_hash_match_rate": round(hash_hits / max(1, len(docs)), 4),
                    "matched_containment": round(m, 4),
                    "random_containment": round(b, 4),
                    "separation": round(m - b, 4),
                }
            )
    return rows


def axis_scramble(bench: Benchmark, corpus: EvalCorpus, n: int = 40) -> list[dict]:
    """The breaking point: paraphrase that also replaces rare content words.

    This is not a faithful paraphrase -- swapping technical vocabulary changes
    what the document is about -- but it is the honest answer to "where does the
    method fail?", and leaving it out would make the robustness claim look
    stronger than it is.
    """
    import re

    vocab = sorted({w for d in corpus.archive[:120] for w in re.findall(r"[a-z]{5,}", d["text"].lower())})
    rng = random.Random(5)
    sources = list(corpus.archive)
    rng.shuffle(sources)
    rows: list[dict] = []
    for level in SCRAMBLE_LEVELS:
        docs = []
        for i, rec in enumerate(sources[:n]):
            para = Paraphraser(strength=0.55, content_scramble=level, vocabulary=vocab, seed=100 + i)
            docs.append(
                {
                    "doc_id": f"scr{level:.1f}:{rec['doc_id']}",
                    "text": para.paraphrase(rec["text"]),
                    "label_human": 1,
                    "klass": "paraphrased",
                    "url": None,
                    "claimed_date": None,
                    "witnesses": [],
                }
            )
        p, _ = bench.scores("C", docs)
        found = sum(
            1
            for d in docs
            if (v := bench._verdicts.get(d["doc_id"])) is not None
            and v.ancestor is not None
            and v.ancestor.is_ancestral
        )
        for row in bench.evaluate(f"scramble@{level:.1f}", docs, bench.syn_test):
            row["content_scramble"] = level
            row["ancestor_recall"] = round(found / max(1, len(docs)), 4)
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- main
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small corpus, for a smoke test")
    ap.add_argument("--offline", action="store_true", help="cache only; no network")
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--no-backdate-probe", action="store_true", help="skip live coverage probes")
    ap.add_argument(
        "--synthetic",
        help="JSONL replacing class (iii) — e.g. real model output from scripts/generate_synthetic.py",
    )
    args = ap.parse_args(argv)

    quick = dict(
        n_synthetic_per_generation=12, n_paraphrase=12, n_backdate=6,
        n_archive=120, n_query_old=100, n_query_recent=120, n_lm=300, n_reference=200,
    )
    full = dict(n_synthetic_per_generation=60, n_paraphrase=80, n_backdate=60)

    print("building corpus ...", flush=True)
    t0 = time.perf_counter()
    corpus = build_corpus(seed=args.seed, **(quick if args.quick else full))
    if args.synthetic:
        from .corpus import substitute_synthetic

        substitute_synthetic(corpus, args.synthetic, seed=args.seed)
        print(f"  class (iii) replaced from {args.synthetic}")
    print("  " + json.dumps(corpus.summary()))
    print(f"  {time.perf_counter() - t0:.1f}s")

    print("fitting detectors ...", flush=True)
    bench = Benchmark(corpus, offline=args.offline, seed=args.seed % 10_000)
    if args.no_backdate_probe:
        bench.dendro.probe_forgeries = False
    print(
        f"  (A) {bench.t_stat_fit:.1f}s   (B) {bench.t_learned_fit:.1f}s   (C) {bench.t_dendro_fit:.1f}s"
    )

    wall: dict[str, float] = {}

    print("axis 1: discrimination ...", flush=True)
    t0 = time.perf_counter()
    main_rows = axis_main(bench)
    wall["all"] = time.perf_counter() - t0
    write_csv(RESULTS / "main.csv", main_rows)

    print("axis 3: generalisation across generator generations ...", flush=True)
    gen_rows = axis_generalization(bench)
    write_csv(RESULTS / "generalization.csv", gen_rows)

    print("axis 2: robustness under attack ...", flush=True)
    sweep = paraphrase_sweep(corpus, PARAPHRASE_STRENGTHS, seed=args.seed, n=(15 if args.quick else 45))
    rob_rows = axis_robustness(bench, sweep)
    write_csv(RESULTS / "robustness.csv", rob_rows)

    print("axis 2c: content-scramble breaking point ...", flush=True)
    scr_rows = axis_scramble(bench, corpus, n=(12 if args.quick else 40))
    write_csv(RESULTS / "scramble.csv", scr_rows)

    print("axis 4: calibration ...", flush=True)
    cal_rows = axis_calibration(bench)
    write_csv(RESULTS / "calibration.csv", cal_rows)

    print("axis 2b: selective prediction ...", flush=True)
    sel_rows = axis_selective(bench)
    write_csv(RESULTS / "selective.csv", sel_rows)

    print("ablation: fingerprint channels ...", flush=True)
    abl_rows = axis_ablation(corpus, n=(12 if args.quick else 45))
    write_csv(RESULTS / "ablation.csv", abl_rows)

    write_csv(RESULTS / "generator_ladder.csv", corpus.ladder)

    print("sensitivity: the assumed prevalence curve ...", flush=True)
    write_csv(RESULTS / "prevalence_sweep.csv", axis_prevalence_sweep(bench, corpus))

    print("axis 5: cost ...", flush=True)
    cost_rows = axis_cost(bench, {"C": wall.get("all", 0.0)})
    write_csv(RESULTS / "cost.csv", cost_rows)

    summary = {
        "corpus": corpus.summary(),
        "seed": args.seed,
        "offline": args.offline,
        "headline": _headline(main_rows, gen_rows, rob_rows),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote results/summary.json")
    print()
    _print_headline(summary["headline"])
    return 0


def _headline(main_rows, gen_rows, rob_rows) -> dict:
    def pick(rows, **kw):
        for r in rows:
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return {}

    out: dict[str, Any] = {}
    for m in ("A", "B", "C"):
        gens = [r for r in gen_rows if r["method"] == m and r.get("seen_family", True)]
        seen = [r["auc"] for r in gens if r.get("seen_in_training")]
        unseen = [r["auc"] for r in gens if not r.get("seen_in_training")]
        other_family = [r["auc"] for r in gen_rows if r["method"] == m and not r.get("seen_family", True)]
        para = [r for r in rob_rows if r["method"] == m and r.get("attack") == "paraphrase"]
        back = pick(rob_rows, method=m, attack="backdate")
        out[m] = {
            "auc_lowbackground": pick(main_rows, condition="lowbackground", method=m).get("auc"),
            "auc_all_eras": pick(main_rows, condition="all_eras", method=m).get("auc"),
            "auc_gen_seen": round(float(np.mean(seen)), 4) if seen else None,
            "auc_gen_unseen": round(float(np.mean(unseen)), 4) if unseen else None,
            "gen_drop": round(float(np.mean(seen) - np.mean(unseen)), 4) if seen and unseen else None,
            "auc_unseen_family": round(float(np.mean(other_family)), 4) if other_family else None,
            "family_drop": (
                round(float(np.mean(seen) - np.mean(other_family)), 4) if seen and other_family else None
            ),
            "auc_paraphrase_055": next(
                (r["auc"] for r in para if abs(r.get("attack_strength", 0) - 0.55) < 1e-6), None
            ),
            "auc_backdate": back.get("auc"),
            "backdate_drop": back.get("auc_drop_vs_clean"),
            "backdate_flag_rate": back.get("backdate_flag_rate"),
            "false_flag_rate": back.get("false_flag_rate"),
        }
    return out


def _print_headline(h: dict) -> None:
    names = {"A": "(A) statistical", "B": "(B) learned", "C": "(C) Dendro"}
    cols = [
        ("auc_lowbackground", "AUC clean"),
        ("auc_gen_seen", "AUC gen 1-2"),
        ("auc_gen_unseen", "AUC gen 3-6"),
        ("auc_unseen_family", "AUC new family"),
        ("auc_paraphrase_055", "AUC paraphrase"),
        ("backdate_flag_rate", "forgery caught"),
        ("false_flag_rate", "false accusations"),
    ]
    print(f"{'method':16s}" + "".join(f"{label:>16s}" for _, label in cols))
    for m in ("A", "B", "C"):
        row = f"{names[m]:16s}"
        for key, _ in cols:
            v = h[m].get(key)
            row += f"{v if v is not None else '-':>16}"
        print(row)


if __name__ == "__main__":
    raise SystemExit(main())
