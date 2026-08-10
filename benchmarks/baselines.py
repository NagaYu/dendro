"""Baselines (A) and (B): the synthetic-text detectors Dendro is compared against.

These are *baselines*, kept deliberately faithful-but-minimal.  The project's
contribution is the evidence side; re-implementing a state-of-the-art detector
would not change the shape of the result, because the failure modes measured
here are structural rather than accuracy-related:

**(A) Statistical / perplexity family.**  The signal every such detector rides is
that generated text is *more predictable* than human text under a general
language model: lower mean per-token log-loss and lower variance ("burstiness").
Also implemented is a curvature probe in the spirit of DetectGPT -- generated
text sits near a local maximum of the model's log-probability, so perturbing it
costs more log-probability than perturbing human text.  All three read the prose
and nothing else, which is exactly why the backdate attack is invisible to them.

**(B) Learned classifier.**  Character and word n-gram TF-IDF into logistic
regression, trained on classes (i) and (iii).  This is a strong baseline in the
regime it was trained for and it is the one that reveals the generalisation
problem: its features describe *a particular generator*, so output from a
generator it never saw is out of distribution.

Both are given every fair advantage: (A)'s reference model is fitted on held-out
human text from the same corpus, and (B) is trained and evaluated with a proper
split.  Both are converted to calibrated probabilities via Platt scaling on the
training split, so the reliability comparison in axis (4) is like-for-like.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .generators import NgramLM

__all__ = ["StatisticalDetector", "LearnedDetector", "auc", "risk_coverage"]


# --------------------------------------------------------------------------- metrics
def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC AUC for "higher score means label 1", by rank statistic.

    Written out rather than imported so the benchmark has one fewer moving part,
    and because ties matter here: an abstaining method produces many identical
    scores, and the midrank treatment below is the correct handling (a tie
    contributes 0.5, i.e. a coin flip, which is what abstention deserves).
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def risk_coverage(
    scores: Sequence[float], labels: Sequence[int], confidences: Sequence[float], n_points: int = 21
) -> list[dict]:
    """Selective-prediction curve: error rate as a function of coverage.

    This is the axis on which abstention pays.  A method that is wrong loudly and
    a method that says "I don't know" can have the same AUC; they are not the
    same tool.  Sorting by confidence and reporting error at each coverage level
    separates them.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    c = np.asarray(confidences, dtype=float)
    order = np.argsort(-c, kind="mergesort")
    s, y = s[order], y[order]
    out: list[dict] = []
    for frac in np.linspace(1.0 / n_points, 1.0, n_points):
        k = max(2, int(round(frac * len(s))))
        sub_s, sub_y = s[:k], y[:k]
        if len(np.unique(sub_y)) < 2:
            out.append({"coverage": float(k / len(s)), "auc": float("nan"), "error": float("nan"), "n": k})
            continue
        thr = float(np.median(sub_s))
        pred = (sub_s > thr).astype(int)
        out.append(
            {
                "coverage": float(k / len(s)),
                "auc": auc(sub_s, sub_y),
                "error": float((pred != sub_y).mean()),
                "n": k,
            }
        )
    return out


# --------------------------------------------------------------------------- (A)
def _platt(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, float]:
    """Fit a 1-D logistic map from raw score to probability (Newton steps)."""
    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if x.size < 8 or len(np.unique(y)) < 2:
        return 1.0, 0.0
    sd = x.std() or 1.0
    x = (x - x.mean()) / sd
    a, b = 0.0, 0.0
    for _ in range(60):
        p = 1.0 / (1.0 + np.exp(-(a * x + b)))
        w = np.clip(p * (1 - p), 1e-6, None)
        g = np.array([np.sum((y - p) * x), np.sum(y - p)])
        h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
        try:
            step = np.linalg.solve(h + 1e-6 * np.eye(2), g)
        except np.linalg.LinAlgError:
            break
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-9:
            break
    return float(a), float(b)


@dataclass
class StatisticalDetector:
    """Baseline (A): unsupervised perplexity-family scoring.

    ``variant`` selects the statistic.  Raw scores are written so that higher
    *conventionally* means more human (human text is less predictable and more
    bursty), but the benchmark always scores through :meth:`predict_proba`, whose
    Platt fit learns the sign from the training split.

    That matters here for an honest reason.  Real LLM output is *lower*
    perplexity than human text; the offline coherence-limited generator used in
    this repository is *higher* perplexity, because limiting coherence adds
    surprise rather than removing it.  The magnitude of the gap -- which is what
    detectability depends on, and what the ladder sweeps -- behaves the same way
    in both cases, but the sign does not.  Letting the detector learn its own
    orientation is what keeps the comparison fair rather than accidentally
    handicapping the baseline; the README says so rather than leaving a reader to
    assume the proxy reproduces the sign as well as the magnitude.
    """

    variant: str = "combined"
    max_order: int = 4
    n_perturbations: int = 6
    perturb_rate: float = 0.12
    seed: int = 0
    reference: Optional[NgramLM] = None
    _platt_params: tuple[float, float] = (1.0, 0.0)
    _norm: tuple[float, float] = (0.0, 1.0)

    def fit_reference(self, texts: Sequence[str]) -> "StatisticalDetector":
        """Fit the reference LM on held-out *human* text -- never on generated text."""
        self.reference = NgramLM(max_order=self.max_order).fit(texts)
        return self

    # -- statistics --------------------------------------------------------
    def _features(self, text: str) -> dict[str, float]:
        assert self.reference is not None, "call fit_reference() first"
        lp = self.reference.token_logprobs(text)
        if lp.size < 5:
            return {"logloss": 0.0, "burstiness": 0.0, "curvature": 0.0}
        logloss = float(-lp.mean())
        burstiness = float(lp.std())
        curvature = self._curvature(text, lp)
        return {"logloss": logloss, "burstiness": burstiness, "curvature": curvature}

    def _curvature(self, text: str, base_lp: np.ndarray) -> float:
        """DetectGPT-style probe: how much log-probability a perturbation costs.

        Generated text sits near a local maximum of the scoring model's
        log-probability, so random perturbation costs it more than it costs
        human text.  Implemented with token deletion/duplication rather than a
        mask-filling model, which is the cheap approximation -- adequate for a
        baseline whose role is to be compared against, not to win.
        """
        assert self.reference is not None
        # A *stable* per-document seed.  Python's builtin ``hash`` for str is salted
        # per process unless PYTHONHASHSEED is fixed, so seeding from it made this
        # baseline's AUC wobble by ~0.003 between otherwise identical runs -- small,
        # but it means "re-run and get the same table" was quietly untrue.
        digest = hashlib.blake2b(text[:64].encode("utf-8"), digest_size=4).digest()
        rng = random.Random(self.seed ^ int.from_bytes(digest, "big"))
        tokens = NgramLM.tokenize(text)
        if len(tokens) < 20:
            return 0.0
        base = float(base_lp.mean())
        drops: list[float] = []
        for _ in range(self.n_perturbations):
            perturbed = list(tokens)
            k = max(1, int(self.perturb_rate * len(tokens)))
            for _ in range(k):
                i = rng.randrange(len(perturbed))
                if rng.random() < 0.5:
                    perturbed.pop(i)
                else:
                    perturbed.insert(i, perturbed[i])
            lp = self.reference.token_logprobs(" ".join(perturbed))
            if lp.size > 4:
                drops.append(base - float(lp.mean()))
        return float(np.mean(drops)) if drops else 0.0

    def raw_score(self, text: str) -> float:
        """Higher = more human.

        ``logloss``: human text is *less* predictable, so higher log-loss reads
        as more human.  ``burstiness``: human text varies more.  ``curvature``:
        generated text loses more log-probability under perturbation, so the
        negated drop reads as more human.
        """
        f = self._features(text)
        if self.variant == "perplexity":
            return f["logloss"]
        if self.variant == "burstiness":
            return f["burstiness"]
        if self.variant == "curvature":
            return -f["curvature"]
        return f["logloss"] + 0.5 * f["burstiness"] - 0.5 * f["curvature"]

    # -- calibration -------------------------------------------------------
    def fit_calibration(self, texts: Sequence[str], labels: Sequence[int]) -> "StatisticalDetector":
        raw = np.array([self.raw_score(t) for t in texts], dtype=float)
        self._norm = (float(raw.mean()), float(raw.std() or 1.0))
        self._platt_params = _platt(raw, labels)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raw = np.array([self.raw_score(t) for t in texts], dtype=float)
        mu, sd = self._norm
        z = (raw - mu) / (sd or 1.0)
        a, b = self._platt_params
        return 1.0 / (1.0 + np.exp(-(a * z + b)))


# --------------------------------------------------------------------------- (B)
@dataclass
class LearnedDetector:
    """Baseline (B): supervised TF-IDF + logistic regression.

    Trained on class (i) versus a *subset* of class (iii) -- by default only the
    first two generator generations.  That restriction is the point of the
    experiment: it mirrors reality, where a detector is trained on the outputs of
    models that exist today and then meets a model that ships next year.
    """

    seed: int = 0
    max_features: int = 60_000
    _pipe: object = field(default=None, repr=False)
    _platt_params: tuple[float, float] = (1.0, 0.0)
    _norm: tuple[float, float] = (0.0, 1.0)
    _calibrated: bool = False
    fitted: bool = False

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "LearnedDetector":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                         max_features=self.max_features, sublinear_tf=True, min_df=2)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         max_features=self.max_features, sublinear_tf=True, min_df=2)),
            ]
        )
        self._pipe = Pipeline(
            [("f", features), ("clf", LogisticRegression(max_iter=2000, C=4.0, random_state=self.seed))]
        )
        self._pipe.fit(list(texts), list(labels))
        self.fitted = True
        return self

    def fit_calibration(self, texts: Sequence[str], labels: Sequence[int]) -> "LearnedDetector":
        """Recalibrate on the shared held-out split.

        A logistic classifier's ``predict_proba`` is calibrated for the
        distribution it was *trained* on, which here is pre-2021 human versus two
        generator generations.  Deployment includes recent human text and unseen
        generators.  Recalibrating on the same held-out split the other two
        conditions use is what makes the reliability comparison like-for-like
        rather than a comparison of how convenient each method's training
        distribution happened to be.
        """
        raw = self._raw(texts)
        self._norm = (float(raw.mean()), float(raw.std() or 1.0))
        z = (raw - self._norm[0]) / (self._norm[1] or 1.0)
        self._platt_params = _platt(z, labels)
        self._calibrated = True
        return self

    def _raw(self, texts: Sequence[str]) -> np.ndarray:
        if not self.fitted or self._pipe is None:
            return np.full(len(texts), 0.5, dtype=float)
        return self._pipe.predict_proba(list(texts))[:, 1]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        raw = self._raw(texts)
        if not self._calibrated:
            return raw
        z = (raw - self._norm[0]) / (self._norm[1] or 1.0)
        a, b = self._platt_params
        return 1.0 / (1.0 + np.exp(-(a * z + b)))
