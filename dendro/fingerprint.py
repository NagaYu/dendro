"""ReflowFingerprint: format-blind, paraphrase-resistant document sketches.

The problem this module solves is the reason evidence-based dating is hard.
The *same text* reaches you through many pipes and none of them preserve bytes:

    a 2019 blog post            ->  HTML with a nav bar and a cookie banner
    the Wayback capture of it   ->  the same HTML plus an injected toolbar
    the Common Crawl WET record ->  de-tagged, re-wrapped at 80 columns
    a 2026 Markdown mirror      ->  smart quotes, different heading syntax
    an LLM paraphrase of it     ->  same facts, different words

A fingerprint that survives the first four and *partially* survives the fifth is
what lets a 2026 document inherit a 2019 existence bound.  Four ideas do the
work:

**1. Aggressive normalisation.**  Markup, boilerplate lines, quote markers, and
whitespace are removed before anything is hashed, so rows 1-4 above collapse to
one identical string.  ``tests/test_fingerprint.py`` asserts the *hash equality*,
not merely a high similarity -- that is the strongest available statement of the
**reflow-invariance** claim.

**2. Multiple channels.**  Exact word shingles die under paraphrase; character
n-grams die a bit slower; the set of rare content words and the set of numerals
and named entities barely move, because a paraphrase that changes "1,247 deaths
in Bergamo" is no longer a paraphrase.  Scoring several channels and combining
them is what buys graceful degradation instead of a cliff, and the per-channel
contribution is measured in ``results/ablation.csv``.

**3. Containment, not similarity.**  A 200-word excerpt inside a 20-page paper
has Jaccard 0.01 and containment 1.0.  Ancestry is an asymmetric question and
needs the asymmetric statistic.

**4. Windows.**  Whole-document sketches cannot localise a quotation.  Sliding
window sketches can, which turns "unrelated" into "12% of this document is
verbatim from that 2019 page" -- the **partial-coverage ancestry** claim.

Nothing here looks at a language model, a perplexity, or a token distribution.
That is the structural reason Dendro's accuracy does not move when a new
generator ships (**generator-independence**).
"""

from __future__ import annotations

import hashlib
import html as _html
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .types import ChannelSketch, Fingerprint, WindowSketch

__all__ = [
    "NormalizationConfig",
    "FingerprintConfig",
    "NormalizedDoc",
    "Normalizer",
    "BoilerplateModel",
    "ReflowFingerprint",
    "LshIndex",
    "estimate_jaccard",
    "estimate_containment",
    "simhash_similarity",
    "minhash_of",
]

_U64 = np.uint64
_MASK64 = (1 << 64) - 1

# --------------------------------------------------------------------------- lexicon
#: High-frequency English function/filler words.  Used for two things: deciding
#: which tokens are "rare" (the paraphrase-resistant channel) and recognising
#: prose-like lines during boilerplate detection.
STOPWORDS: frozenset[str] = frozenset(
    """
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further had hadn't has hasn't have haven't having he her here hers herself him
himself his how i if in into is isn't it its itself just let's me more most mustn't my myself no nor
not of off on once only or other ought our ours ourselves out over own same shan't she should
shouldn't so some such than that the their theirs them themselves then there these they this those
through to too under until up very was wasn't we were weren't what when where which while who whom
why will with won't would wouldn't you your yours yourself yourselves also may might must shall upon
one two three new use used using make made many much well back even still way take take get go know
see come think look want give first last long great little other another good new old right big high
different small large next early young important few public bad same able
""".split()
)

#: Lines matching these are page furniture, not content.  Removing them is what
#: makes a Wayback capture and a Common Crawl WET record hash identically.
BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*(skip to (main )?content|jump to (navigation|content))\s*$",
        r"^\s*(share (this)?|tweet|print|email this|save for later)\s*$",
        r"^\s*(cookie|we use cookies|accept (all )?cookies|privacy (policy|notice)|terms of (use|service))\b",
        r"^\s*(all rights reserved|copyright\b|\(c\)\s*\d{4}|©)",
        r"^\s*(sign (in|up)|log ?in|log ?out|subscribe( now)?|newsletter)\s*$",
        r"^\s*(home|about|contact|search|menu|navigation|next|previous|back to top)\s*$",
        r"^\s*(posted (in|on|by)|filed under|tags?:|categor(y|ies):)\b",
        r"^\s*(advertisement|sponsored|related (articles|posts|stories))\s*$",
        r"^\s*(loading|javascript is (required|disabled)|enable javascript)\b",
        r"^\s*\d+\s*(comments?|shares?|likes?|views?|min read)\s*$",
        r"^\s*(the wayback machine|internet archive|archived from the original)\b",
        r"^\s*(follow us|connect with us)\b",
        r"^\s*(read more|continue reading|view all|see also)\s*$",
        r"^\s*(edit|history|discussion|talk|permalink|reply|quote)\s*$",
    )
)

#: Separators used by navigation bars and share widgets.  A short line built out
#: of these is furniture even when no individual segment matches a pattern.
_NAV_SPLIT_RE = re.compile(r"\s*[|·•‧⋅/›»—–]\s*|\s{3,}")
_SENTENCE_END_RE = re.compile(r"[.!?。！？]\s*$")

_TAG_RE = re.compile(r"<[^>]{1,2000}>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
#: The whole document head is metadata, not body text.  Leaving it in meant the
#: contents of <title> were prepended to the normalised text, so an HTML
#: rendering and a plain-text copy of the same prose hashed differently -- the
#: reflow-invariance guarantee, broken by one tag.  Extractors like Common
#: Crawl's WET pipeline drop the head, and so does this.
_HEAD_RE = re.compile(r"<head\b.*?</head>", re.IGNORECASE | re.DOTALL)
_TITLE_RE = re.compile(r"<title\b.*?</title>", re.IGNORECASE | re.DOTALL)
_BLOCK_TAGS = (
    "p div br hr li ul ol dl dt dd tr td th table section article aside nav header footer "
    "main figure figcaption blockquote pre form fieldset h1 h2 h3 h4 h5 h6"
).split()
_BLOCK_RE = re.compile(r"</?\s*(?:%s)\b[^>]*>" % "|".join(_BLOCK_TAGS), re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_REF_RE = re.compile(r"!?\[([^\]]*)\]\[[^\]]*\]")
_QUOTE_PREFIX_RE = re.compile(r"^\s*(?:[>|:]+\s*)+")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+•·]|\(?\d{1,3}[.)])\s+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*|\s*#+\s*$")
_EMPHASIS_RE = re.compile(r"[*_`~]{1,3}")
_FOOTNOTE_RE = re.compile(r"\[\s*\d{1,3}\s*\]|\(\s*\d{1,3}\s*\)")
_CITATION_RE = re.compile(r"\[(?:[A-Za-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}[a-z]?)\]")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]+")
_NUM_RE = re.compile(r"\d[\d,.:/%-]*\d|\d")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-ZÀ-ɏ]{2,}(?:\s+[A-Z][a-zA-ZÀ-ɏ]{2,}){0,3}")

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "–": "-", "—": "-", "‒": "-", "−": "-", "―": "-",
    "…": "...", " ": " ", "​": "", "‌": "", "‍": "",
    "﻿": "", "­": "", "′": "'", "″": '"',
}
_PUNCT_TABLE = str.maketrans(_PUNCT_MAP)


# --------------------------------------------------------------------------- hashing
def _splitmix64(x: np.ndarray, salt: int) -> np.ndarray:
    """A 64-bit hash family indexed by ``salt``.

    Each salt gives an (empirically) independent permutation of the 64-bit
    space, which is exactly what MinHash needs.  Written with explicit
    ``np.uint64`` constants because ``np.uint64 * python_int`` silently promotes
    to float64 and would destroy the low bits.
    """
    z = (x ^ _U64(salt & _MASK64)).astype(np.uint64, copy=True)
    z ^= z >> _U64(30)
    z *= _U64(0xBF58476D1CE4E5B9)
    z ^= z >> _U64(27)
    z *= _U64(0x94D049BB133111EB)
    z ^= z >> _U64(31)
    return z


def _hash_tokens(items: Sequence[str]) -> np.ndarray:
    """Stable 64-bit base hashes for a list of strings.

    Uses blake2b rather than Python's ``hash`` so that fingerprints are
    identical across processes and machines -- a hard requirement for a cache of
    archive sketches that is committed to the repository.
    """
    if not items:
        return np.zeros(0, dtype=np.uint64)
    out = np.empty(len(items), dtype=np.uint64)
    for i, item in enumerate(items):
        out[i] = int.from_bytes(
            hashlib.blake2b(item.encode("utf-8"), digest_size=8).digest(), "big"
        )
    return out


def minhash_of(items: Sequence[str], num_perm: int = 128, salts: Optional[np.ndarray] = None) -> np.ndarray:
    """MinHash signature of a set of strings.

    ``O(|S| * num_perm)`` in numpy; a 10k-shingle document with 128 permutations
    is about 1.3M uint64 operations, i.e. milliseconds.  Cost matters because
    benchmark axis (5) counts per-document expense.
    """
    if salts is None:
        salts = _perm_salts(num_perm)
    base = np.unique(_hash_tokens(list(items)))
    if base.size == 0:
        return np.full(num_perm, np.uint64(_MASK64), dtype=np.uint64)
    sig = np.empty(num_perm, dtype=np.uint64)
    for j in range(num_perm):
        sig[j] = _splitmix64(base, int(salts[j])).min()
    return sig


def _perm_salts(num_perm: int, seed: int = 0xD3ADB33F) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, 1 << 62, size=num_perm, dtype=np.uint64)


def _simhash(items: Sequence[str], weights: Optional[Sequence[float]] = None) -> int:
    """64-bit SimHash.  Cheap locality signal, used as a fast pre-filter."""
    if not items:
        return 0
    base = _hash_tokens(list(items))
    w = np.ones(len(items), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    bits = ((base[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & _U64(1)).astype(np.int8)
    acc = ((bits * 2 - 1).astype(np.float64) * w[:, None]).sum(axis=0)
    value = 0
    for i in range(64):
        if acc[i] > 0:
            value |= 1 << i
    return value


def estimate_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Unbiased MinHash estimate of ``|A n B| / |A u B|``.

    Supports **reflow-invariance** and **cost**: the estimate is a single
    vectorised comparison of two fixed-width signatures, so comparing a
    document against a whole archive layer stays affordable.
    """
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    return float(np.count_nonzero(a == b) / a.size)


def estimate_containment(jaccard: float, card_a: int, card_b: int) -> float:
    """Recover ``|A n B| / |A|`` from a Jaccard estimate plus both cardinalities.

    ``J = I / (a + b - I)``  =>  ``I = J(a + b) / (1 + J)``.  This is the standard
    cardinality-corrected containment and it is what makes "this 2026 article
    quotes 40% of a 2019 page" a computable statement.  Variance grows when the
    two sets differ wildly in size, which is precisely when the window-level
    alignment in :mod:`dendro.alignment` takes over.
    """
    if card_a <= 0 or jaccard <= 0.0:
        return 0.0
    inter = jaccard * (card_a + card_b) / (1.0 + jaccard)
    return float(min(1.0, max(0.0, inter / card_a)))


def simhash_similarity(a: int, b: int) -> float:
    """``1 - hamming/64``.  Chance level is 0.5, not 0.

    A cheap pre-filter only. Reported for diagnostics rather than used for
    ancestry decisions, because its floor at 0.5 makes it a poor **calibration**
    substrate compared with containment.
    """
    return 1.0 - (bin((int(a) ^ int(b)) & _MASK64).count("1") / 64.0)


# --------------------------------------------------------------------------- normalisation
@dataclass(frozen=True)
class NormalizationConfig:
    """Which invariances the fingerprint is asked to have.

    Every switch here is an *ablation knob*: turning one off and re-running
    ``benchmarks/run.py --ablation`` shows how much of the reflow invariance it
    was responsible for.
    """

    strip_markup: bool = True
    strip_boilerplate: bool = True
    strip_quote_prefixes: bool = True
    strip_urls: bool = True
    strip_footnote_markers: bool = True
    casefold: bool = True
    fold_punctuation: bool = True
    collapse_whitespace: bool = True
    drop_short_lines: int = 0      # 0 disables; otherwise drop lines with < N words
    keep_numbers: bool = True      # numbers are a paraphrase-resistant signal


@dataclass(frozen=True)
class NormalizedDoc:
    """Result of normalisation: canonical text plus the token views built from it."""

    text: str
    tokens: tuple[str, ...]
    rare_terms: tuple[str, ...]
    numerals: tuple[str, ...]
    entities: tuple[str, ...]
    dropped_lines: int
    sha256: str


class BoilerplateModel:
    """Corpus-fitted line-level boilerplate detector.

    Pattern matching catches the universal furniture ("skip to content").  Site
    furniture is idiosyncratic, so the remaining boilerplate is found
    empirically: any normalised line that appears in more than ``min_doc_frac``
    of a document collection is chrome, not content.  Fitting is optional --
    Dendro works without it, just with a slightly noisier fingerprint.
    """

    def __init__(self, min_doc_frac: float = 0.15, min_docs: int = 3, max_words: int = 25) -> None:
        self.min_doc_frac = float(min_doc_frac)
        self.min_docs = int(min_docs)
        self.max_words = int(max_words)
        self.blocked: set[str] = set()
        self.n_docs = 0

    def fit(self, texts: Iterable[str]) -> "BoilerplateModel":
        counts: Counter[str] = Counter()
        n = 0
        for text in texts:
            n += 1
            seen = {
                _light_line_key(line)
                for line in text.splitlines()
                if line.strip() and len(line.split()) <= self.max_words
            }
            counts.update(seen)
        self.n_docs = n
        if n >= self.min_docs:
            threshold = max(self.min_docs, math.ceil(self.min_doc_frac * n))
            self.blocked = {k for k, c in counts.items() if c >= threshold and k}
        return self

    def is_boilerplate(self, line: str) -> bool:
        return _light_line_key(line) in self.blocked


def _is_nav_line(line: str) -> bool:
    """True for menu / share / breadcrumb strips.

    A Wayback capture keeps the site's navigation; a Common Crawl WET record
    keeps a differently-wrapped version of it; a Markdown mirror has none.  Left
    in, that furniture is 10-20% of a short page's shingles and it is *shared
    across every page of the site*, which manufactures spurious near-duplicates.
    Removing it is a precondition for the **reflow-invariance** claim.
    """
    stripped = line.strip()
    if len(stripped.split()) > 14 or _SENTENCE_END_RE.search(stripped):
        return False
    segments = [s.strip() for s in _NAV_SPLIT_RE.split(stripped) if s.strip()]
    if len(segments) < 2:
        return False
    furniture = 0
    for seg in segments:
        words = seg.split()
        if any(p.search(seg) for p in BOILERPLATE_PATTERNS):
            furniture += 1
        elif len(words) <= 3 and not _SENTENCE_END_RE.search(seg):
            furniture += 1
    return furniture >= len(segments) - 0  # every segment must look like furniture


def _light_line_key(line: str) -> str:
    s = unicodedata.normalize("NFKC", line).translate(_PUNCT_TABLE).casefold()
    s = _WS_RE.sub(" ", s).strip()
    return s


class Normalizer:
    """Turns any rendering of a document into one canonical string.

    The order of operations is load-bearing.  Markup goes first (so boilerplate
    patterns see prose, not tags), boilerplate second (so line structure still
    exists), whitespace last (so line-based rules had something to work with).
    """

    def __init__(
        self,
        config: Optional[NormalizationConfig] = None,
        boilerplate: Optional[BoilerplateModel] = None,
    ) -> None:
        self.config = config or NormalizationConfig()
        self.boilerplate = boilerplate

    # -- stages ------------------------------------------------------------
    def _demarkup(self, text: str) -> str:
        if not self.config.strip_markup:
            return text
        text = _SCRIPT_RE.sub(" ", text)
        text = _COMMENT_RE.sub(" ", text)
        text = _HEAD_RE.sub(" ", text)
        text = _TITLE_RE.sub(" ", text)
        # Turn block-level boundaries into newlines so the line-level rules still
        # see structure.  Both opening *and* closing tags count: a page written as
        # one long line (`<nav>…</nav><p>…</p><footer>…</footer>`) would otherwise
        # collapse its navigation and its body into a single line, and the nav
        # detector -- which only fires on short, sentence-free lines -- would never
        # see anything to remove.
        text = _BLOCK_RE.sub("\n", text)
        text = _TAG_RE.sub(" ", text)
        text = _html.unescape(text)
        text = _MD_LINK_RE.sub(r"\1", text)
        text = _MD_REF_RE.sub(r"\1", text)
        return text

    def _clean_line(self, line: str) -> Optional[str]:
        cfg = self.config
        if cfg.strip_quote_prefixes:
            line = _QUOTE_PREFIX_RE.sub("", line)
            line = _LIST_PREFIX_RE.sub("", line)
        if cfg.strip_markup:
            line = _HEADING_RE.sub("", line)
            line = _EMPHASIS_RE.sub("", line)
        stripped = line.strip()
        if not stripped:
            return None
        if cfg.strip_boilerplate:
            if any(p.search(stripped) for p in BOILERPLATE_PATTERNS):
                return None
            if _is_nav_line(stripped):
                return None
            if self.boilerplate is not None and self.boilerplate.is_boilerplate(stripped):
                return None
        if cfg.drop_short_lines and len(stripped.split()) < cfg.drop_short_lines:
            return None
        return stripped

    def normalize(self, text: str) -> NormalizedDoc:
        """Canonicalise, then derive the four token views the channels need."""
        cfg = self.config
        raw = text or ""
        work = self._demarkup(raw)
        work = unicodedata.normalize("NFKC", work)
        if cfg.fold_punctuation:
            work = work.translate(_PUNCT_TABLE)
        if cfg.strip_urls:
            work = _URL_RE.sub(" ", work)
            work = _EMAIL_RE.sub(" ", work)
        if cfg.strip_footnote_markers:
            work = _CITATION_RE.sub(" ", work)
            work = _FOOTNOTE_RE.sub(" ", work)

        kept, dropped = [], 0
        for line in work.splitlines():
            cleaned = self._clean_line(line)
            if cleaned is None:
                if line.strip():
                    dropped += 1
            else:
                kept.append(cleaned)
        joined = "\n".join(kept)

        # Entities and numerals are read off the *cleaned but not yet casefolded*
        # text.  Capitalisation is the only signal for entities, and reading them
        # after boilerplate removal keeps nav words ("Home", "Contact") out of a
        # channel whose whole value is that paraphrase leaves it alone.
        numerals = tuple(sorted({m.group(0) for m in _NUM_RE.finditer(joined)})) if cfg.keep_numbers else ()
        entities = tuple(sorted({m.group(0).strip() for m in _ENTITY_RE.finditer(joined)}))

        flat = joined
        if cfg.casefold:
            flat = flat.casefold()
        if cfg.collapse_whitespace:
            flat = _WS_RE.sub(" ", flat).strip()

        tokens = _tokenize(flat)
        rare = tuple(t for t in tokens if len(t) >= 4 and t not in STOPWORDS and not t.isdigit())
        digest = hashlib.sha256(flat.encode("utf-8")).hexdigest()
        return NormalizedDoc(
            text=flat,
            tokens=tokens,
            rare_terms=rare,
            numerals=numerals,
            entities=entities,
            dropped_lines=dropped,
            sha256=digest,
        )


def _tokenize(text: str) -> tuple[str, ...]:
    """Word tokens, with CJK runs emitted as character bigrams.

    CJK has no spaces, so word shingling would degenerate to one giant token.
    Character bigrams give a comparable granularity and keep the module usable
    on Japanese and Chinese sources without a segmenter dependency.
    """
    out: list[str] = []
    pos = 0
    for m in _CJK_RE.finditer(text):
        out.extend(_WORD_RE.findall(text[pos : m.start()]))
        run = m.group(0)
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
        pos = m.end()
    out.extend(_WORD_RE.findall(text[pos:]))
    return tuple(out)


# --------------------------------------------------------------------------- fingerprinting
@dataclass(frozen=True)
class FingerprintConfig:
    """Sketch geometry.

    ``num_perm`` trades accuracy for speed: the standard error of a Jaccard
    estimate is ``sqrt(J(1-J)/k)``, so 128 permutations give +-0.04 at J=0.5.
    ``window_tokens``/``window_stride`` set the smallest quotation that can be
    localised -- 128/64 means any verbatim run of ~128 tokens lands wholly
    inside at least one window.
    """

    num_perm: int = 128
    word_shingle: int = 5
    char_shingle: int = 5
    window_tokens: int = 128
    window_stride: int = 64
    window_perm: int = 64
    min_window_tokens: int = 48
    channel_weights: Mapping[str, float] = field(
        default_factory=lambda: {"word": 0.40, "char": 0.25, "rare": 0.25, "num": 0.10}
    )


class ReflowFingerprint:
    """Builds :class:`~dendro.types.Fingerprint` objects and scores them.

    A single instance is reusable and cheap to share: the permutation salts are
    derived deterministically from a fixed seed, so two processes -- or a
    committed sketch cache and a live run -- produce comparable signatures.
    """

    def __init__(
        self,
        config: Optional[FingerprintConfig] = None,
        normalization: Optional[NormalizationConfig] = None,
        boilerplate: Optional[BoilerplateModel] = None,
    ) -> None:
        self.config = config or FingerprintConfig()
        self.normalizer = Normalizer(normalization, boilerplate)
        self._salts = _perm_salts(self.config.num_perm)
        self._window_salts = _perm_salts(self.config.window_perm, seed=0x5EEDBEEF)

    # -- channel construction ---------------------------------------------
    def _word_shingles(self, tokens: Sequence[str]) -> list[str]:
        w = self.config.word_shingle
        if len(tokens) < w:
            return [" ".join(tokens)] if tokens else []
        return [" ".join(tokens[i : i + w]) for i in range(len(tokens) - w + 1)]

    def _char_shingles(self, text: str) -> list[str]:
        n = self.config.char_shingle
        squeezed = text.replace(" ", "")
        if len(squeezed) < n:
            return [squeezed] if squeezed else []
        return [squeezed[i : i + n] for i in range(len(squeezed) - n + 1)]

    def _channel(self, name: str, items: Sequence[str], weights=None) -> ChannelSketch:
        uniq = list(dict.fromkeys(items))
        sig = minhash_of(uniq, self.config.num_perm, self._salts)
        return ChannelSketch(
            name=name,
            minhash=sig,
            simhash=_simhash(items, weights),
            cardinality=len(uniq),
            weight=float(self.config.channel_weights.get(name, 0.0)),
        )

    def _windows(self, tokens: Sequence[str]) -> tuple[WindowSketch, ...]:
        cfg = self.config
        n = len(tokens)
        if n == 0:
            return ()
        size, stride = cfg.window_tokens, cfg.window_stride
        starts = list(range(0, max(1, n - size + 1), stride)) if n > size else [0]
        if n > size and starts[-1] + size < n:
            starts.append(n - size)
        out: list[WindowSketch] = []
        for idx, s in enumerate(starts):
            e = min(n, s + size)
            if e - s < min(cfg.min_window_tokens, n):
                continue
            shingles = self._word_shingles(tokens[s:e])
            out.append(
                WindowSketch(
                    index=idx,
                    start_token=s,
                    end_token=e,
                    minhash=minhash_of(shingles, cfg.window_perm, self._window_salts),
                    simhash=_simhash(shingles),
                )
            )
        return tuple(out)

    # -- public ------------------------------------------------------------
    def fingerprint(self, doc_id: str, text: str) -> Fingerprint:
        """Sketch a document through every channel.

        Demonstrates **reflow-invariance**: two renderings of the same prose
        yield the same ``normalized_sha256`` and therefore identical sketches,
        and **generator-independence**: no step consults a language model.
        """
        nd = self.normalizer.normalize(text)
        word_sh = self._word_shingles(nd.tokens)
        idf_w = _local_idf_weights(word_sh)
        channels = {
            "word": self._channel("word", word_sh, idf_w),
            "char": self._channel("char", self._char_shingles(nd.text)),
            "rare": self._channel("rare", list(dict.fromkeys(nd.rare_terms))),
            "num": self._channel(
                "num", list(dict.fromkeys(list(nd.numerals) + [e.casefold() for e in nd.entities]))
            ),
        }
        return Fingerprint(
            doc_id=doc_id,
            n_tokens=len(nd.tokens),
            normalized_sha256=nd.sha256,
            channels=channels,
            windows=self._windows(nd.tokens),
            meta={
                "dropped_lines": nd.dropped_lines,
                "n_rare": len(set(nd.rare_terms)),
                "n_numerals": len(nd.numerals),
                "n_entities": len(nd.entities),
                "normalized_chars": len(nd.text),
            },
        )

    def channel_scores(self, query: Fingerprint, ref: Fingerprint) -> dict[str, float]:
        """Per-channel containment of the query inside the reference.

        Containment rather than Jaccard because the interesting relation is
        asymmetric.  The spread across channels is the diagnostic that separates
        *reflow* (all channels high) from *paraphrase* (``word`` collapses,
        ``rare``/``num`` hold) -- the core of the **adversarial-robustness** claim.
        """
        out: dict[str, float] = {}
        for name, qch in query.channels.items():
            rch = ref.channels.get(name)
            if rch is None or qch.cardinality == 0 or rch.cardinality == 0:
                out[name] = 0.0
                continue
            j = estimate_jaccard(qch.minhash, rch.minhash)
            out[name] = estimate_containment(j, qch.cardinality, rch.cardinality)
        return out

    def combined_score(self, channel_scores: Mapping[str, float]) -> float:
        w = self.config.channel_weights
        total = sum(w.get(k, 0.0) for k in channel_scores) or 1.0
        return float(sum(channel_scores.get(k, 0.0) * w.get(k, 0.0) for k in channel_scores) / total)


def _local_idf_weights(shingles: Sequence[str]) -> list[float]:
    """Down-weight repeated shingles inside one document.

    A page that repeats its own tagline 40 times should not have that tagline
    dominate its SimHash.
    """
    counts = Counter(shingles)
    return [1.0 / math.log2(1.0 + counts[s]) if counts[s] > 1 else 1.0 for s in shingles]


# --------------------------------------------------------------------------- LSH
class LshIndex:
    """Banded LSH over MinHash signatures, for documents and for windows.

    Alignment against an archive layer must be sublinear or the whole approach
    is impractical at dataset scale; benchmark axis (5) reports the candidate
    count this achieves.  Band geometry ``(bands, rows)`` sets the approximate
    similarity threshold ``s ~ (1/bands)^(1/rows)``.

    **Retrieval indexes every paraphrase-resistant channel, not just ``word``.**
    That is not an optimisation, it is a correctness requirement, and getting it
    wrong is subtle: an earlier version indexed only exact word shingles, so a
    paraphrase whose word-channel containment had fallen to 0.18 produced no band
    collision and its true source was never even *considered* as a candidate.
    Classification was working perfectly and ancestor recall was still 25%,
    because the answer never reached the classifier.  Retrieval has to survive
    whatever the attack is, or the rest of the pipeline never runs.
    """

    DEFAULT_CHANNELS = ("word", "rare", "num")

    def __init__(
        self,
        bands: int = 32,
        rows: int = 4,
        channel: str = "word",
        channels: Optional[Sequence[str]] = None,
    ) -> None:
        self.bands = int(bands)
        self.rows = int(rows)
        self.channel = channel
        self.channels = tuple(channels) if channels is not None else self.DEFAULT_CHANNELS
        self._doc_buckets: dict[tuple[str, int, bytes], set[str]] = defaultdict(set)
        self._win_buckets: dict[tuple[int, bytes], set[tuple[str, int]]] = defaultdict(set)
        self.fingerprints: dict[str, Fingerprint] = {}

    @property
    def threshold(self) -> float:
        return float((1.0 / self.bands) ** (1.0 / self.rows))

    def _band_keys(self, sig: np.ndarray, bands: int, rows: int) -> list[tuple[int, bytes]]:
        usable = min(bands * rows, sig.size)
        bands = max(1, usable // rows)
        return [
            (b, sig[b * rows : (b + 1) * rows].tobytes())
            for b in range(bands)
        ]

    def add(self, fp: Fingerprint) -> None:
        self.fingerprints[fp.doc_id] = fp
        for name in self.channels:
            ch = fp.channels.get(name)
            if ch is None or not ch.cardinality:
                continue
            for band, sig in self._band_keys(ch.minhash, self.bands, self.rows):
                self._doc_buckets[(name, band, sig)].add(fp.doc_id)
        for w in fp.windows:
            for key in self._band_keys(w.minhash, self.bands, 2):
                self._win_buckets[key].add((fp.doc_id, w.index))

    def add_all(self, fps: Iterable[Fingerprint]) -> "LshIndex":
        for fp in fps:
            self.add(fp)
        return self

    def query(self, fp: Fingerprint, include_windows: bool = True) -> dict[str, int]:
        """Candidate reference ids with the number of colliding bands.

        Window collisions count too, which is how a short quotation retrieves a
        long source document that whole-document banding would never surface --
        the retrieval half of **partial-coverage ancestry**.
        """
        hits: Counter[str] = Counter()
        for name in self.channels:
            ch = fp.channels.get(name)
            if ch is None or not ch.cardinality:
                continue
            for band, sig in self._band_keys(ch.minhash, self.bands, self.rows):
                for doc_id in self._doc_buckets.get((name, band, sig), ()):
                    if doc_id != fp.doc_id:
                        hits[doc_id] += 1
        if include_windows:
            for w in fp.windows:
                for key in self._band_keys(w.minhash, self.bands, 2):
                    for doc_id, _ in self._win_buckets.get(key, ()):
                        if doc_id != fp.doc_id:
                            hits[doc_id] += 1
        return dict(hits)

    def __len__(self) -> int:
        return len(self.fingerprints)
