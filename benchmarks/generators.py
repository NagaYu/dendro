"""Synthetic-text generators, paraphrase attacks, and backdate forgery.

These build classes (iii) and (iv) of the evaluation corpus.  Two design
commitments make the benchmark mean something.

**The generator ladder is a measured axis, not a label.**  The headline figure
claims that statistical and learned detectors decay as generators improve while
Dendro stays flat.  If "generation" were an arbitrary index, that figure would be
decoration.  Here a generation is a sampling temperature, and what is plotted on
the x-axis is the *measured* distributional distance between generated text and
held-out human text (mean log-loss gap and burstiness gap under a reference
model).  Detector (A) decays because the quantity it measures genuinely shrinks;
nothing is stipulated.

The mechanism is the real one, scaled down.  Perplexity-family detectors work
because generated text is *more predictable* than human text under a general
language model -- lower mean log-loss, lower variance.  Sampling an n-gram model
at temperature ``T`` reproduces exactly that: low ``T`` concentrates on frequent
continuations (very detectable, like greedy decoding from an early model); at
``T -> 1`` the sample comes from the fitted approximation of the human
distribution itself and the gap closes.  An n-gram model is a weak stand-in for a
transformer in every respect except the one being tested, which is the axis the
detectors actually consume.

**The paraphrase attack is not rigged in Dendro's favour.**  It would be easy to
write a paraphraser that preserves precisely the channels Dendro relies on.  This
one substitutes *content words* -- the rare-term channel's whole substrate -- at
a controlled rate, on top of reordering, clause restructuring, and splitting.
What it preserves is what a real LLM paraphrase preserves: numerals and named
entities, because a rewrite that changes "1,247 deaths in Bergamo" has stopped
being a rewrite of that text.  Dendro degrades under it, and
``results/robustness.csv`` reports by how much.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "NgramLM",
    "GeneratorConfig",
    "SyntheticGenerator",
    "GENERATION_LADDER",
    "Paraphraser",
    "backdate",
    "measure_detectability_axis",
]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_BOS = "<s>"
_EOS = "</s>"


# --------------------------------------------------------------------------- LM
class NgramLM:
    """Back-off n-gram model: the generator *and* the reference scorer.

    Used for two different jobs, always fitted on *disjoint* human splits so the
    benchmark is not measuring self-recognition:

    * fitted on the generator split, it produces class (iii);
    * fitted on the reference split, it scores perplexity for baseline (A).

    Smoothing is stupid backoff, which is not a proper distribution but is the
    standard cheap choice and is monotone in the way scoring needs.
    """

    def __init__(self, max_order: int = 4, backoff: float = 0.4) -> None:
        self.max_order = int(max_order)
        self.backoff = float(backoff)
        self.counts: list[dict[tuple[str, ...], Counter]] = [defaultdict(Counter) for _ in range(self.max_order)]
        self.totals: list[dict[tuple[str, ...], int]] = [defaultdict(int) for _ in range(self.max_order)]
        self.vocab: set[str] = set()
        self.n_tokens = 0
        self._uni_cache = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Whitespace tokenisation -- chosen to make detokenisation lossless.

        A punctuation-splitting tokeniser would be linguistically nicer and was
        the first implementation, but it made the benchmark *invalid*: rejoining
        the pieces mangled LaTeX (``$\\infty$`` came back as ``$ {\\ infty$``) and
        shifted punctuation spacing, so generated text carried a surface
        fingerprint that had nothing to do with the generator's statistics.  The
        learned baseline then scored a perfect 1.0 on every generation by reading
        the formatting artefact, and the generalisation axis measured nothing.

        Splitting on whitespace round-trips exactly, so generated and human text
        differ only in word choice and order -- which is the difference the
        experiment is about.
        """
        return (text or "").split()

    def fit(self, texts: Iterable[str]) -> "NgramLM":
        for text in texts:
            tokens = [_BOS] * (self.max_order - 1) + self.tokenize(text) + [_EOS]
            self.vocab.update(tokens)
            self.n_tokens += len(tokens)
            for order in range(1, self.max_order + 1):
                idx = order - 1
                for i in range(self.max_order - 1, len(tokens)):
                    ctx = tuple(tokens[i - idx : i])
                    self.counts[idx][ctx][tokens[i]] += 1
                    self.totals[idx][ctx] += 1
        return self

    # -- scoring -----------------------------------------------------------
    def logprob(self, context: Sequence[str], token: str) -> float:
        """log P(token | context) with stupid backoff down to a uniform floor."""
        penalty = 0.0
        for order in range(min(self.max_order, len(context) + 1), 0, -1):
            idx = order - 1
            ctx = tuple(context[len(context) - idx :]) if idx else ()
            total = self.totals[idx].get(ctx, 0)
            if total:
                c = self.counts[idx][ctx].get(token, 0)
                if c:
                    return math.log(c / total) + penalty
            penalty += math.log(self.backoff)
        return math.log(1.0 / max(len(self.vocab), 1000)) + penalty

    def token_logprobs(self, text: str) -> np.ndarray:
        """Per-token log-probabilities -- the raw material for baseline (A)."""
        tokens = [_BOS] * (self.max_order - 1) + self.tokenize(text)
        out = [
            self.logprob(tokens[max(0, i - self.max_order + 1) : i], tokens[i])
            for i in range(self.max_order - 1, len(tokens))
        ]
        return np.asarray(out, dtype=float) if out else np.zeros(1, dtype=float)

    # -- generation --------------------------------------------------------
    def _unigram_items(self):
        if self._uni_cache is None:
            counter = self.counts[0].get((), Counter())
            items = [(t, c) for t, c in counter.items() if t not in (_BOS, _EOS)]
            self._uni_cache = (items, [float(c) for _, c in items], float(sum(c for _, c in items)))
        return self._uni_cache

    def generate(
        self,
        n_tokens: int,
        order: int = 3,
        temperature: float = 1.0,
        coherence: float = 1.0,
        top_k: Optional[int] = None,
        rng: Optional[random.Random] = None,
        seed_context: Optional[Sequence[str]] = None,
    ) -> str:
        """Sample text at a given order, temperature, and *coherence*.

        ``coherence`` is the ladder's real knob and the one with a clean
        interpretation: it is the probability that a token is drawn from the
        conditional model rather than from the marginal (unigram) distribution.
        It interpolates between a bag of words and the fitted approximation of
        the human process::

            P_g(w | ctx) = coherence * P_ngram(w | ctx) + (1 - coherence) * P_unigram(w)

        So "a better generator" means "captures more of the true conditional
        structure", which is both the honest description of what improved
        language models do and a quantity a reference model can actually see.
        At ``coherence = 1`` the sample comes from the model fitted on human text
        and the gap to human statistics is as small as this family allows.

        Temperature is applied to the count distribution (``p ∝ c^(1/T)``) and is
        held fixed across the ladder so that ``coherence`` is the only moving
        part and the measured axis is uncontaminated.
        """
        rng = rng or random.Random(0)
        order = max(1, min(order, self.max_order))
        history: list[str] = list(seed_context or [_BOS] * (order - 1))
        out: list[str] = []
        inv_t = 1.0 / max(1e-3, temperature)
        uni_items, uni_weights, uni_total = self._unigram_items()

        for _ in range(n_tokens):
            token = None
            if coherence < 1.0 and rng.random() > coherence and uni_total > 0:
                token = _weighted_choice(uni_items, uni_weights, uni_total, rng)
            else:
                for o in range(order, 0, -1):
                    idx = o - 1
                    ctx = tuple(history[len(history) - idx :]) if idx else ()
                    counter = self.counts[idx].get(ctx)
                    if counter:
                        items = list(counter.items())
                        if top_k:
                            # Truncated sampling: a structurally different decoding
                            # rule, not merely a different temperature.  Its
                            # artefacts differ in kind (a hard support cut rather
                            # than a reweighting), which is what makes it a usable
                            # stand-in for an unseen *model family*.
                            items = sorted(items, key=lambda kv: -kv[1])[:top_k]
                        weights = [c**inv_t for _, c in items]
                        total = sum(weights)
                        if total > 0:
                            token = _weighted_choice(items, weights, total, rng)
                            break
            if token is None or token == _EOS:
                history = [_BOS] * (order - 1)
                if out and out[-1] not in ".!?":
                    out.append(".")
                continue
            out.append(token)
            history.append(token)
        return " ".join(out)


def _weighted_choice(items, weights, total, rng: random.Random):
    r = rng.random() * total
    upto = 0.0
    for (tok, _), w in zip(items, weights):
        upto += w
        if upto >= r:
            return tok
    return items[-1][0]


def _join_words(words: Sequence[str]) -> str:
    """Re-join word/punctuation tokens with conventional spacing."""
    out: list[str] = []
    for tok in words:
        if not tok[:1].isalnum() and tok not in "([{":
            out.append(tok)
        else:
            out.append((" " if out else "") + tok)
    return "".join(out)


def detokenize(tokens: Sequence[str]) -> str:
    """Re-join tokens into prose with plausible spacing and capitalisation."""
    out: list[str] = []
    capitalise = True
    for tok in tokens:
        if re.fullmatch(r"[^\w\s]", tok):
            if tok in "([{\"'":
                out.append(" " + tok)
            else:
                out.append(tok)
            if tok in ".!?":
                capitalise = True
            continue
        word = tok.capitalize() if capitalise and tok[:1].islower() else tok
        out.append((" " if out else "") + word)
        capitalise = False
    text = "".join(out).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


# --------------------------------------------------------------------------- ladder
@dataclass(frozen=True)
class GeneratorConfig:
    """One rung of the generator ladder."""

    name: str
    order: int
    temperature: float
    generation: int
    coherence: float = 1.0
    top_k: Optional[int] = None
    family: str = "temperature"


#: Six "generations", ordered by how much of the human conditional structure the
#: generator reproduces.  Only ``coherence`` moves, so the ladder has exactly one
#: degree of freedom and ``results/generator_ladder.csv`` reports the *measured*
#: distance to human text for each rung -- the axis is confirmed, not stipulated.
GENERATION_LADDER: tuple[GeneratorConfig, ...] = (
    GeneratorConfig("gen1", order=3, temperature=1.0, generation=1, coherence=0.30),
    GeneratorConfig("gen2", order=3, temperature=1.0, generation=2, coherence=0.50),
    GeneratorConfig("gen3", order=3, temperature=1.0, generation=3, coherence=0.68),
    GeneratorConfig("gen4", order=3, temperature=1.0, generation=4, coherence=0.82),
    GeneratorConfig("gen5", order=3, temperature=1.0, generation=5, coherence=0.93),
    GeneratorConfig("gen6-frontier", order=3, temperature=1.0, generation=6, coherence=1.00),
)

#: A generator family the detectors never see during training.  It differs in
#: *decoding rule* (hard top-k truncation) and *order*, not merely in a scalar
#: setting, which is the analogue of meeting a different model rather than the
#: same model at a different temperature.  Axis (3b) evaluates on this, and it is
#: the condition where a learned detector's features stop describing reality --
#: while Dendro, which never reads the text, cannot be affected in principle.
UNSEEN_FAMILY: tuple[GeneratorConfig, ...] = (
    GeneratorConfig("unseen-topk8", order=4, temperature=0.9, generation=7,
                    coherence=0.97, top_k=8, family="top-k"),
    GeneratorConfig("unseen-topk40", order=2, temperature=1.05, generation=8,
                    coherence=1.00, top_k=40, family="top-k"),
)


class SyntheticGenerator:
    """Produces class (iii): synthetic documents across generator generations."""

    def __init__(self, lm: NgramLM) -> None:
        self.lm = lm

    def generate(
        self,
        config: GeneratorConfig,
        n_docs: int,
        tokens_per_doc: int = 220,
        seed: int = 0,
    ) -> list[dict]:
        rng = random.Random(seed * 1000 + config.generation)
        out: list[dict] = []
        for i in range(n_docs):
            text = self.lm.generate(
                tokens_per_doc + rng.randint(-40, 60),
                order=config.order,
                temperature=config.temperature,
                coherence=config.coherence,
                top_k=config.top_k,
                rng=rng,
            )
            out.append(
                {
                    "doc_id": f"syn:{config.name}:{i:04d}",
                    "text": text,
                    "generator": config.name,
                    "generation": config.generation,
                    "temperature": config.temperature,
                    "family": config.family,
                    "label_human": 0,
                    "klass": "synthetic",
                    "era": "recent",
                }
            )
        return out


def measure_detectability_axis(
    reference: NgramLM, human_texts: Sequence[str], synthetic_texts: Sequence[str]
) -> dict[str, float]:
    """Measure how far a generation actually sits from human text.

    This is the honest x-axis for the headline figure.  Two statistics, both of
    them the ones perplexity-family detectors consume:

    ``logloss_gap``
        difference in mean per-token negative log-probability under a reference
        model fitted on a *disjoint* human split.
    ``burstiness_gap``
        difference in the standard deviation of those same per-token values.
        Human text is bursty; sampled-at-low-temperature text is not.

    A generation whose gaps are near zero is, by construction, one that a
    perplexity detector cannot see -- and Dendro's performance on it should be
    unchanged, because Dendro never computes either number.
    """

    def stats(texts: Sequence[str]) -> tuple[float, float]:
        means, sds = [], []
        for t in texts:
            lp = reference.token_logprobs(t)
            if lp.size > 4:
                means.append(-float(lp.mean()))
                sds.append(float(lp.std()))
        return (float(np.mean(means)) if means else float("nan"),
                float(np.mean(sds)) if sds else float("nan"))

    h_mean, h_sd = stats(human_texts)
    s_mean, s_sd = stats(synthetic_texts)
    return {
        "human_logloss": h_mean,
        "synthetic_logloss": s_mean,
        "logloss_gap": abs(s_mean - h_mean),
        "human_burstiness": h_sd,
        "synthetic_burstiness": s_sd,
        "burstiness_gap": abs(s_sd - h_sd),
    }


# --------------------------------------------------------------------------- paraphrase
#: Content-word substitutions.  Deliberately targets the rare-term channel that
#: Dendro's paraphrase resistance depends on -- an attack that spared it would
#: prove nothing.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "approach": ("method", "strategy", "technique"), "method": ("approach", "procedure", "technique"),
    "results": ("findings", "outcomes", "measurements"), "show": ("demonstrate", "indicate", "reveal"),
    "shows": ("demonstrates", "indicates", "reveals"), "propose": ("introduce", "put forward", "present"),
    "significant": ("substantial", "considerable", "marked"), "improve": ("enhance", "boost", "strengthen"),
    "improves": ("enhances", "boosts", "strengthens"), "performance": ("effectiveness", "quality", "capability"),
    "model": ("system", "framework", "formulation"), "models": ("systems", "frameworks", "formulations"),
    "data": ("observations", "records", "measurements"), "study": ("investigation", "analysis", "examination"),
    "analysis": ("examination", "assessment", "evaluation"), "increase": ("rise", "growth", "gain"),
    "decrease": ("decline", "reduction", "drop"), "large": ("substantial", "extensive", "sizeable"),
    "small": ("modest", "limited", "minor"), "important": ("crucial", "notable", "consequential"),
    "difficult": ("challenging", "demanding", "hard"), "problem": ("issue", "difficulty", "challenge"),
    "solution": ("remedy", "answer", "resolution"), "based": ("grounded", "founded", "built"),
    "using": ("employing", "applying", "leveraging"), "used": ("employed", "applied", "adopted"),
    "provide": ("offer", "supply", "deliver"), "provides": ("offers", "supplies", "delivers"),
    "obtain": ("acquire", "derive", "secure"), "observed": ("recorded", "noted", "detected"),
    "consider": ("examine", "regard", "treat"), "suggests": ("implies", "points to", "signals"),
    "suggest": ("imply", "point to", "signal"), "measure": ("quantify", "gauge", "assess"),
    "estimate": ("approximate", "gauge", "compute"), "framework": ("architecture", "scheme", "structure"),
    "evidence": ("indication", "support", "grounds"), "effect": ("influence", "impact", "consequence"),
    "effects": ("influences", "impacts", "consequences"), "change": ("shift", "variation", "alteration"),
    "changes": ("shifts", "variations", "alterations"), "reduce": ("lower", "diminish", "curtail"),
    "reduces": ("lowers", "diminishes", "curtails"), "produce": ("yield", "generate", "create"),
    "report": ("describe", "document", "record"), "reports": ("describes", "documents", "records"),
    "recent": ("current", "latest", "contemporary"), "previous": ("earlier", "prior", "preceding"),
    "several": ("a number of", "various", "multiple"), "various": ("assorted", "diverse", "several"),
    "however": ("nevertheless", "even so", "that said"), "therefore": ("consequently", "as a result", "hence"),
    "additionally": ("moreover", "in addition", "further"), "finally": ("lastly", "in closing", "at last"),
    "task": ("problem", "objective", "job"), "tasks": ("problems", "objectives", "jobs"),
    "training": ("fitting", "optimisation", "learning"), "learning": ("training", "acquisition", "adaptation"),
    "network": ("architecture", "graph", "system"), "networks": ("architectures", "graphs", "systems"),
    "algorithm": ("procedure", "routine", "scheme"), "algorithms": ("procedures", "routines", "schemes"),
    "experiments": ("trials", "tests", "evaluations"), "experiment": ("trial", "test", "evaluation"),
    "accurate": ("precise", "exact", "faithful"), "accuracy": ("precision", "correctness", "fidelity"),
    "compare": ("contrast", "benchmark", "juxtapose"), "compared": ("contrasted", "benchmarked", "set against"),
    "existing": ("current", "established", "available"), "novel": ("new", "original", "fresh"),
    "efficient": ("economical", "streamlined", "lean"), "robust": ("resilient", "stable", "durable"),
    "general": ("broad", "wide-ranging", "overarching"), "specific": ("particular", "distinct", "targeted"),
    "system": ("platform", "apparatus", "setup"), "systems": ("platforms", "apparatuses", "setups"),
    "process": ("procedure", "mechanism", "operation"), "structure": ("organisation", "arrangement", "layout"),
    "function": ("role", "operation", "purpose"), "value": ("magnitude", "quantity", "figure"),
    "values": ("magnitudes", "quantities", "figures"), "region": ("area", "zone", "district"),
    "sample": ("specimen", "draw", "selection"), "samples": ("specimens", "draws", "selections"),
    "condition": ("state", "circumstance", "setting"), "conditions": ("states", "circumstances", "settings"),
    "relationship": ("association", "link", "correspondence"), "distribution": ("spread", "profile", "allocation"),
    "parameters": ("coefficients", "settings", "quantities"), "parameter": ("coefficient", "setting", "quantity"),
    "control": ("regulate", "govern", "manage"), "predict": ("forecast", "anticipate", "project"),
    "prediction": ("forecast", "projection", "anticipation"), "identify": ("pinpoint", "detect", "single out"),
    "develop": ("build", "construct", "devise"), "developed": ("built", "constructed", "devised"),
    "apply": ("deploy", "utilise", "bring to bear"), "applied": ("deployed", "utilised", "brought to bear"),
    "require": ("demand", "call for", "necessitate"), "requires": ("demands", "calls for", "necessitates"),
    "achieve": ("attain", "reach", "realise"), "achieves": ("attains", "reaches", "realises"),
    "limited": ("constrained", "restricted", "bounded"), "complex": ("intricate", "involved", "elaborate"),
    "simple": ("straightforward", "plain", "uncomplicated"), "different": ("distinct", "dissimilar", "separate"),
    "similar": ("comparable", "alike", "analogous"), "high": ("elevated", "steep", "considerable"),
    "low": ("reduced", "modest", "slight"), "quality": ("calibre", "standard", "grade"),
    "information": ("detail", "material", "content"), "knowledge": ("understanding", "expertise", "insight"),
    "research": ("investigation", "inquiry", "scholarship"), "work": ("effort", "undertaking", "labour"),
    "paper": ("article", "manuscript", "report"), "understanding": ("comprehension", "grasp", "insight"),
    "theory": ("account", "hypothesis", "doctrine"), "empirical": ("observational", "experimental", "data-driven"),
    "demonstrate": ("establish", "illustrate", "make clear"), "represent": ("depict", "stand for", "encode"),
    "generate": ("produce", "create", "yield"), "estimated": ("approximated", "computed", "gauged"),
    "observed": ("recorded", "detected", "witnessed"), "described": ("characterised", "outlined", "set out"),
}

#: Multi-word rewrites -- the constructions an LLM paraphrase reliably flattens.
PHRASES: tuple[tuple[str, str], ...] = (
    (r"\bin order to\b", "to"), (r"\bdue to the fact that\b", "because"),
    (r"\ba number of\b", "several"), (r"\bit is possible that\b", "possibly"),
    (r"\bwith respect to\b", "regarding"), (r"\bin terms of\b", "as regards"),
    (r"\bas well as\b", "and also"), (r"\bin addition to\b", "besides"),
    (r"\bmake use of\b", "use"), (r"\btake into account\b", "account for"),
    (r"\bis able to\b", "can"), (r"\bare able to\b", "can"),
    (r"\bplays? an important role\b", "matters substantially"),
    (r"\bthe majority of\b", "most"), (r"\bat the same time\b", "simultaneously"),
    (r"\bon the other hand\b", "conversely"), (r"\bin this paper\b", "here"),
    (r"\bwe show that\b", "our results establish that"),
    (r"\bwe propose\b", "we put forward"), (r"\bwe present\b", "we set out"),
    (r"\bhas been shown\b", "was found"), (r"\bcan be used to\b", "serves to"),
    (r"\bit should be noted that\b", "note that"), (r"\bfor the purpose of\b", "for"),
)

_PRESERVE_RE = re.compile(r"^[A-Z]|^\d|\d")

#: Adverbials and hedges a rewrite drops in at clause boundaries.  Inserting one
#: token every few positions is what actually destroys word n-grams -- far more
#: effectively than synonym substitution, and it is what real paraphrase does.
_FILLERS: tuple[str, ...] = (
    "notably", "in practice", "broadly", "here", "indeed", "in turn", "as such",
    "by contrast", "more precisely", "in effect", "crucially", "typically",
    "in particular", "arguably", "overall", "that is", "correspondingly",
)
#: Function words a rewrite frequently elides without changing meaning.
_DROPPABLE: frozenset[str] = frozenset(
    {"the", "a", "an", "that", "which", "very", "quite", "rather", "own", "also", "then"}
)


@dataclass
class Paraphraser:
    """LLM-style rewrite: class (iv-a), the paraphrase attack.

    Three edit families, because only together do they constitute a real attack:

    ``strength``
        probability that an eligible word is swapped for a synonym.
    ``structural_rate``
        per-position probability of an *insertion, deletion, or local
        transposition*.  This is the part that matters.  Exact word 5-grams
        require five consecutive tokens to survive intact, so perturbing one
        position in four removes essentially all of them -- which is precisely
        what a genuine rewrite does and what a naive synonym-only "paraphraser"
        fails to do.  An earlier version of this class had only synonym
        substitution and left word-channel containment at 0.54 even at maximum
        strength; it was not an attack at all, and any robustness measured
        against it would have been an artefact.
    ``content_scramble``
        the *harsh* ablation: replace rare content words with unrelated ones
        drawn from the corpus.  Off by default because it is not faithful --
        a rewrite that turns "quantum annealer" into an unrelated term is not a
        paraphrase of that document.  It is swept anyway in
        ``results/robustness.csv`` so the breaking point is on the record rather
        than left as an unexamined assumption.

    Numerals and capitalised entities are preserved, mirroring real paraphrase.
    That preservation is not a convenience: it is the mechanism Dendro exploits,
    and stating it plainly is the difference between a claim and a trick.
    """

    strength: float = 0.5
    structural_rate: Optional[float] = None
    content_scramble: float = 0.0
    preserve_entities: bool = True
    seed: int = 0
    vocabulary: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.structural_rate is None:
            # Scale with strength: at strength 0.55 roughly one position in four
            # is perturbed, which is enough to remove nearly every 5-gram.
            object.__setattr__(self, "structural_rate", 0.45 * self.strength)

    def paraphrase(self, text: str) -> str:
        rng = self._rng
        work = text
        for pattern, replacement in PHRASES:
            if rng.random() < self.strength:
                work = re.sub(pattern, replacement, work, flags=re.IGNORECASE)

        sentences = [s for s in _SENT_RE.split(work) if s.strip()]
        rewritten = [self._rewrite_sentence(s, rng) for s in sentences]

        # Local reordering: real rewrites move material around a little.
        i = 0
        while i + 1 < len(rewritten):
            if rng.random() < self.strength * 0.30:
                rewritten[i], rewritten[i + 1] = rewritten[i + 1], rewritten[i]
                i += 2
            else:
                i += 1
        return " ".join(s.strip() for s in rewritten if s.strip())

    # -- token-level ------------------------------------------------------
    def _is_protected(self, tok: str, position: int) -> bool:
        if not self.preserve_entities:
            return False
        if any(ch.isdigit() for ch in tok):
            return True
        return bool(tok[:1].isupper() and position > 0)

    def _substitute(self, tok: str, rng: random.Random) -> str:
        lower = tok.lower()
        options = SYNONYMS.get(lower)
        if options and rng.random() < self.strength:
            choice = options[rng.randrange(len(options))]
            return choice.capitalize() if tok[:1].isupper() else choice
        if (
            self.content_scramble > 0
            and self.vocabulary
            and len(lower) >= 4
            and lower not in _DROPPABLE
            and rng.random() < self.content_scramble
        ):
            return self.vocabulary[rng.randrange(len(self.vocabulary))]
        return tok

    def _rewrite_sentence(self, sentence: str, rng: random.Random) -> str:
        words = re.findall(r"\w+|[^\w\s]", sentence)
        out: list[str] = []
        i = 0
        rate = float(self.structural_rate or 0.0)
        while i < len(words):
            tok = words[i]
            if not tok[:1].isalnum():
                out.append(tok)
                i += 1
                continue

            protected = self._is_protected(tok, i)
            roll = rng.random()

            # Disjoint bands, weighted toward the edits that break n-grams
            # *without* wrecking grammaticality.  Transposition is the most
            # destructive per edit and also the most damaging to fluency, so it
            # gets the smallest share: paraphrased text that reads as broken
            # would confound baseline (A), whose whole signal is fluency, and the
            # comparison would then measure this class's artefacts rather than
            # the effect of paraphrase.
            if roll < rate * 0.45:
                out.append(_FILLERS[rng.randrange(len(_FILLERS))])   # insert an adverbial
                out.append(tok if protected else self._substitute(tok, rng))
                i += 1
                continue
            if roll < rate * 0.78:
                if not protected and tok.lower() in _DROPPABLE:
                    i += 1                                           # elide a function word
                    continue
                out.append(_FILLERS[rng.randrange(len(_FILLERS))])
                out.append(tok if protected else self._substitute(tok, rng))
                i += 1
                continue
            if (
                roll < rate
                and not protected
                and i + 1 < len(words)
                and words[i + 1][:1].isalnum()
                and not self._is_protected(words[i + 1], i + 1)
            ):
                out.append(self._substitute(words[i + 1], rng))       # transpose a pair
                out.append(self._substitute(tok, rng))
                i += 2
                continue

            out.append(tok if protected else self._substitute(tok, rng))
            i += 1

        result = _join_words(out)
        if rng.random() < self.strength * 0.35:
            result = self._restructure(result, rng)
        return result

    @staticmethod
    def _restructure(sentence: str, rng: random.Random) -> str:
        """Clause-level surgery: split on a connective and swap the halves."""
        for connective, template in (
            (" because ", "Since {b}, {a}"),
            (" although ", "Even though {b}, {a}"),
            (" while ", "Whereas {b}, {a}"),
            (" and ", "{a}. Additionally, {b}"),
            (" but ", "{a}. Yet {b}"),
        ):
            if connective in sentence.lower():
                idx = sentence.lower().index(connective)
                a, b = sentence[:idx].strip(), sentence[idx + len(connective) :].strip()
                if len(a.split()) >= 4 and len(b.split()) >= 4:
                    b = b.rstrip(".")
                    a = a.rstrip(".")
                    return template.format(a=a[0].lower() + a[1:] if template.startswith("Since") or
                                           template.startswith("Even") or template.startswith("Whereas") else a,
                                           b=b) + "."
        return sentence


# --------------------------------------------------------------------------- backdating
_BACKDATE_STYLES = ("html_meta", "frontmatter", "jsonld", "prose")


def backdate(text: str, when, style: Optional[str] = None, seed: int = 0) -> str:
    """Class (iv-b): wrap a document in forged "this is old" metadata.

    Four realistic surfaces, because a forger picks whichever the target platform
    reads.  Every one of them is invisible to a statistical or learned detector --
    those consume prose, and the prose is unchanged.  Dendro catches all four,
    not by being cleverer about the text, but by asking archives that were
    demonstrably crawling that neighbourhood whether they ever saw it.
    """
    rng = random.Random(seed)
    style = style or _BACKDATE_STYLES[rng.randrange(len(_BACKDATE_STYLES))]
    iso = when.date().isoformat() if hasattr(when, "date") else str(when)[:10]
    pretty = when.strftime("%B %d, %Y") if hasattr(when, "strftime") else iso

    if style == "html_meta":
        return (
            f'<html><head><meta name="date" content="{iso}">\n'
            f'<meta property="article:published_time" content="{iso}T09:14:00Z">\n'
            f"</head><body>\n<p>{text}</p>\n</body></html>"
        )
    if style == "frontmatter":
        return f"---\ntitle: archived note\ndate: {iso}\nauthor: staff\n---\n\n{text}"
    if style == "jsonld":
        return (
            '<html><head><script type="application/ld+json">'
            f'{{"@context":"https://schema.org","@type":"Article","datePublished":"{iso}",'
            f'"dateModified":"{iso}"}}</script></head><body>{text}</body></html>'
        )
    return f"Published on {pretty}\n\n{text}"
