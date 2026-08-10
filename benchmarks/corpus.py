"""Assemble the labelled evaluation corpus, classes (i) through (iv).

The splitting discipline here is what keeps the benchmark from being circular,
so it is worth stating before the code.

The pre-2021 human documents are cut into **four disjoint roles**:

``archive``   indexed into the alignment layer with their real arXiv registration
              dates.  These are "what the archives hold".
``query``     class (i) test documents.  *Not* in the archive layer -- they are
              dated by their own witnesses, exactly as a real query would be.
``lm_train``  trains the synthetic generator.  Kept out of the archive layer so
              that high-temperature n-gram output cannot regurgitate archived
              text and hand Dendro a free (and fake) ancestor.
``reference`` trains the perplexity baseline's reference model.  Disjoint from
              ``lm_train`` so baseline (A) is not scoring text against the very
              model that produced it.

Paraphrase attacks are built from the ``archive`` half, because that is the real
situation: the original is archived and the rewrite is new.  Backdate attacks are
given URLs on **real hosts with real deep Wayback coverage but paths that were
never archived** -- so the coverage probe that drives the flag is a measurement,
not a stipulation.

Two honesty notes that the README repeats:

* Class (ii) is *recent, human-attributed*, not *verified human*.  By 2025 an
  unknown share of abstracts have passed through a model.  No clean recent human
  corpus exists -- which is the argument for dating by archive rather than by
  inspection, and the reason the primary task is defined on class (i).
* The arXiv ``<created>`` date used as a witness is genuine independent evidence
  fetched from arXiv's own API and cached, not a simulated timestamp.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dendro.types import UTC, Witness, WitnessKind, to_utc  # noqa: E402

from .generators import (  # noqa: E402
    GENERATION_LADDER,
    UNSEEN_FAMILY,
    GeneratorConfig,
    NgramLM,
    Paraphraser,
    SyntheticGenerator,
    backdate,
    measure_detectability_axis,
)

CORPUS_DIR = REPO / "data" / "corpus"

#: Real hosts with deep pre-2021 Wayback coverage.  Backdated forgeries are given
#: URLs here on paths that never existed, so "the archive was looking and saw
#: nothing" is a fact the CDX API confirms rather than an assumption.
FORGERY_HOSTS: tuple[str, ...] = (
    "https://www.python.org/dev/peps/pep-{n}-draft/",
    "https://www.gnu.org/philosophy/{n}-essay.html",
    "https://docs.python.org/3/tutorial/{n}-appendix.html",
    "https://www.w3.org/Provider/Style/{n}-note.html",
    "https://curl.se/docs/{n}-guide.html",
    "https://www.kernel.org/doc/html/latest/process/{n}-notes.html",
)


# --------------------------------------------------------------------------- containers
@dataclass
class EvalCorpus:
    """Everything the benchmark needs, with labels and provenance attached."""

    archive: list[dict] = field(default_factory=list)
    human_old: list[dict] = field(default_factory=list)
    human_recent: list[dict] = field(default_factory=list)
    synthetic: list[dict] = field(default_factory=list)
    unseen_family: list[dict] = field(default_factory=list)
    paraphrased: list[dict] = field(default_factory=list)
    backdated: list[dict] = field(default_factory=list)
    lm_train: list[dict] = field(default_factory=list)
    reference: list[dict] = field(default_factory=list)
    ladder: list[dict] = field(default_factory=list)

    def all_queries(self) -> list[dict]:
        return [*self.human_old, *self.human_recent, *self.synthetic, *self.paraphrased, *self.backdated]

    def summary(self) -> dict[str, int]:
        return {
            "archive": len(self.archive),
            "human_old(i)": len(self.human_old),
            "human_recent(ii)": len(self.human_recent),
            "synthetic(iii)": len(self.synthetic),
            "unseen_family(iii-b)": len(self.unseen_family),
            "paraphrased(iv-a)": len(self.paraphrased),
            "backdated(iv-b)": len(self.backdated),
            "lm_train": len(self.lm_train),
            "reference": len(self.reference),
        }


def _load(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_real_documents() -> tuple[list[dict], list[dict], list[dict]]:
    """(pre-2021 arXiv, recent arXiv, pre-2021 web) as fetched by ``scripts/fetch_corpus``."""
    return (
        _load(CORPUS_DIR / "arxiv_pre2021.jsonl"),
        _load(CORPUS_DIR / "arxiv_recent.jsonl"),
        _load(CORPUS_DIR.parent / "local" / "web_pre2021.jsonl"),
    )


# --------------------------------------------------------------------------- witnesses
def arxiv_witness(record: dict) -> Witness:
    """The real arXiv registration record, as a witness.

    This is not a simulated timestamp.  arXiv's OAI-PMH interface reported
    ``<created>`` for this identifier, the response is cached in
    ``data/fixtures/cache``, and the same value is what
    :class:`dendro.sources.scholarly.ArxivSource` would return from a live query.
    Using it directly keeps a 500-document benchmark from making 500 redundant
    API calls while leaving the evidence genuinely archival.
    """
    return Witness(
        source_id="arxiv",
        operator="arxiv-cornell",
        kind=WitnessKind.REGISTRATION,
        observed_at=to_utc(record["published"]),
        target=record.get("arxiv_id") or record["doc_id"],
        reliability=0.999,
        forgeability=5e-4,
        coverage=0.98,
        url=record.get("url"),
        cached=True,
        raw={"source": "arxiv-oai", "id": record.get("arxiv_id")},
    )


def web_witness(record: dict) -> Witness:
    """A real Internet Archive capture of a pre-2021 page."""
    return Witness(
        source_id="wayback",
        operator="internet-archive",
        kind=WitnessKind.SNAPSHOT,
        observed_at=to_utc(record["published"]),
        target=record.get("url") or record["doc_id"],
        reliability=0.995,
        forgeability=1e-3,
        coverage=0.9,
        url=record.get("wayback_url"),
        cached=True,
        raw={"source": "wayback-cdx"},
    )


def witnesses_for(record: dict) -> list[Witness]:
    if record.get("genre") == "web":
        return [web_witness(record)]
    if record.get("published"):
        return [arxiv_witness(record)]
    return []


# --------------------------------------------------------------------------- build
def _allocator(pool: list[dict]):
    """Hand out disjoint slices of a shuffled pool, in call order."""
    state = {"i": 0}

    def take(n: int) -> list[dict]:
        start = state["i"]
        end = min(len(pool), start + max(0, int(n)))
        state["i"] = end
        return pool[start:end]

    return take


def build_corpus(
    seed: int = 20260810,
    n_synthetic_per_generation: int = 45,
    n_paraphrase: int = 60,
    n_backdate: int = 45,
    paraphrase_strength: float = 0.55,
    ladder: Sequence[GeneratorConfig] = GENERATION_LADDER,
    tokens_per_doc: int = 230,
    n_archive: int = 450,
    n_query_old: int = 400,
    n_query_recent: int = 500,
    n_lm: int = 900,
    n_reference: int = 500,
) -> EvalCorpus:
    """Construct classes (i)-(iv) with disjoint roles and reproducible seeding."""
    rng = random.Random(seed)
    arxiv_old, arxiv_recent, web_old = load_real_documents()
    if not arxiv_old:
        raise RuntimeError(
            "no corpus found -- run `python -m scripts.fetch_corpus` first "
            "(it caches to data/fixtures/cache so it only has to happen once)"
        )

    # Roles are sized independently rather than as fractions of one pool, because
    # they have genuinely different appetites.  The *archive* and *query* splits
    # drive runtime (every query document is aligned against every archived one),
    # so they are capped.  The *generator* split drives corpus validity and wants
    # everything it can get: an early version trained the generator on 41
    # abstracts and the learned baseline then hit AUC 1.0 on every generation by
    # recognising the vocabulary rather than the generator.  All four roles remain
    # strictly disjoint.
    pre = list(arxiv_old)
    rng.shuffle(pre)
    take = _allocator(pre)
    archive_split = take(n_archive)
    query_split = take(n_query_old)
    lm_split = take(n_lm)
    ref_old = take(n_reference // 2)

    recent = list(arxiv_recent)
    rng.shuffle(recent)
    take_recent = _allocator(recent)
    recent_query = take_recent(n_query_recent)
    ref_recent = take_recent(n_reference // 2)

    # The reference model for baseline (A) is fitted on human text from *both*
    # eras.  Fitting it on one era only would let the baseline separate class (i)
    # from class (ii) on vocabulary drift alone -- a real effect, but not the one
    # under test, and it would quietly contaminate every comparison downstream.
    ref_split = [*ref_old, *ref_recent]

    corpus = EvalCorpus()

    # ---- archive layer: pre-2021 documents with real witnesses ----
    for rec in [*archive_split, *web_old]:
        corpus.archive.append(
            {
                **rec,
                "klass": "archive",
                "witnesses": witnesses_for(rec),
                "not_after": to_utc(rec["published"]),
            }
        )

    # ---- (i) pre-2021 human queries ----
    for rec in query_split:
        corpus.human_old.append(
            _query_doc(rec, klass="human_old", label_human=1, era="pre2021", claimed=to_utc(rec["published"]))
        )

    # ---- (ii) recent human-attributed queries ----
    for rec in recent_query:
        corpus.human_recent.append(
            _query_doc(rec, klass="human_recent", label_human=1, era="recent", claimed=to_utc(rec["published"]))
        )

    corpus.lm_train = list(lm_split)
    corpus.reference = list(ref_split)

    # ---- (iii) synthetic across generator generations ----
    lm = NgramLM(max_order=4).fit(d["text"] for d in lm_split)
    reference_lm = NgramLM(max_order=4).fit(d["text"] for d in ref_split)
    gen = SyntheticGenerator(lm)
    human_probe = [d["text"] for d in query_split[:120]]

    for cfg in ladder:
        docs = gen.generate(cfg, n_synthetic_per_generation, tokens_per_doc=tokens_per_doc, seed=seed)
        for d in docs:
            d.update({"witnesses": [], "url": None, "claimed_date": None, "label_human": 0})
        corpus.synthetic.extend(docs)
        axis = measure_detectability_axis(reference_lm, human_probe, [d["text"] for d in docs])
        corpus.ladder.append({"generator": cfg.name, "generation": cfg.generation,
                              "coherence": cfg.coherence, "temperature": cfg.temperature,
                              "order": cfg.order, "family": cfg.family, **axis})

    # ---- (iii-b) a generator family no detector is trained on ----
    for cfg in UNSEEN_FAMILY:
        docs = gen.generate(cfg, n_synthetic_per_generation, tokens_per_doc=tokens_per_doc, seed=seed + 77)
        for d in docs:
            d.update({"witnesses": [], "url": None, "claimed_date": None, "label_human": 0})
        corpus.unseen_family.extend(docs)
        axis = measure_detectability_axis(reference_lm, human_probe, [d["text"] for d in docs])
        corpus.ladder.append({"generator": cfg.name, "generation": cfg.generation,
                              "coherence": cfg.coherence, "temperature": cfg.temperature,
                              "order": cfg.order, "family": cfg.family, **axis})

    # ---- (iv-a) paraphrase attack on archived originals ----
    sources = list(archive_split)
    rng.shuffle(sources)
    for i, rec in enumerate(sources[:n_paraphrase]):
        para = Paraphraser(strength=paraphrase_strength, seed=seed + i)
        corpus.paraphrased.append(
            {
                "doc_id": f"para:{rec['doc_id']}",
                "text": para.paraphrase(rec["text"]),
                # Human-derived content: the ground truth is that this text
                # descends from a 2019 human document.  A detector that calls it
                # synthetic has made a false accusation about human writing,
                # which is the error class that matters most here.
                "label_human": 1,
                "klass": "paraphrased",
                "era": "pre2021",
                "url": None,
                "claimed_date": None,
                "witnesses": [],
                "source_doc_id": rec["doc_id"],
                "source_published": rec["published"],
                "paraphrase_strength": paraphrase_strength,
            }
        )

    # ---- (iv-b) backdated synthetic ----
    synth_pool = [d for d in corpus.synthetic if d["generation"] >= 3]
    rng.shuffle(synth_pool)
    for i, d in enumerate(synth_pool[:n_backdate]):
        forged = _forged_date(rng)
        template = FORGERY_HOSTS[i % len(FORGERY_HOSTS)]
        slug = hashlib.blake2b(d["doc_id"].encode(), digest_size=4).hexdigest()
        corpus.backdated.append(
            {
                "doc_id": f"backdate:{d['doc_id']}",
                "text": backdate(d["text"], forged, seed=seed + i),
                "label_human": 0,
                "klass": "backdated",
                "era": "recent",
                "url": template.format(n=slug),
                "claimed_date": forged.isoformat(),
                "witnesses": [],
                "generation": d["generation"],
                "generator": d["generator"],
                "forged_date": forged.isoformat(),
            }
        )
    return corpus


def _query_doc(rec: dict, klass: str, label_human: int, era: str, claimed: _dt.datetime) -> dict:
    return {
        "doc_id": rec["doc_id"],
        "text": rec["text"],
        "url": rec.get("url"),
        "label_human": label_human,
        "klass": klass,
        "era": era,
        "claimed_date": claimed.isoformat(),
        "published": rec["published"],
        "witnesses": witnesses_for(rec),
        "category": rec.get("category"),
        "genre": rec.get("genre"),
    }


def _forged_date(rng: random.Random) -> _dt.datetime:
    """A plausible pre-LLM date: uniform over 2017-2020."""
    start = _dt.datetime(2017, 1, 1, tzinfo=UTC)
    return start + _dt.timedelta(days=rng.randrange(0, 4 * 365))


# --------------------------------------------------------------------------- variants
def paraphrase_sweep(
    corpus: EvalCorpus, strengths: Sequence[float], seed: int = 7, n: int = 45
) -> dict[float, list[dict]]:
    """Rebuild the paraphrase attack at several strengths.

    The degradation *curve* is the deliverable, not a single number.  A method
    that survives strength 0.2 and collapses at 0.4 is not robust, and only a
    sweep shows that.
    """
    rng = random.Random(seed)
    sources = list(corpus.archive)
    rng.shuffle(sources)
    picked = sources[:n]
    out: dict[float, list[dict]] = {}
    for s in strengths:
        docs = []
        for i, rec in enumerate(picked):
            para = Paraphraser(strength=s, seed=seed + i)
            docs.append(
                {
                    "doc_id": f"para{s:.2f}:{rec['doc_id']}",
                    "text": para.paraphrase(rec["text"]),
                    "label_human": 1,
                    "klass": "paraphrased",
                    "era": "pre2021",
                    "url": None,
                    "claimed_date": None,
                    "witnesses": [],
                    "source_doc_id": rec["doc_id"],
                    "paraphrase_strength": s,
                }
            )
        out[s] = docs
    return out


def substitute_synthetic(corpus: EvalCorpus, path: str | pathlib.Path, seed: int = 20260810) -> EvalCorpus:
    """Replace class (iii) with documents from a JSONL file.

    This is the escape hatch for the offline generator's main limitation: it
    varies decoding, not architecture.  Point this at real model output from
    ``scripts/generate_synthetic.py --backend anthropic`` and every condition
    re-runs against it, including the backdate forgeries, which are rebuilt from
    the substituted documents so the attack is applied to the new text rather
    than to stale n-gram output.

    Condition (C) is expected to be *unchanged* by the substitution.  If it is
    not, the generator-independence claim is false and this is the experiment
    that shows it.
    """
    rng = random.Random(seed)
    rows = [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"no documents in {path}")

    docs: list[dict] = []
    for i, r in enumerate(rows):
        docs.append(
            {
                "doc_id": r.get("doc_id") or f"syn:external:{i:04d}",
                "text": r["text"],
                "generator": r.get("generator", "external"),
                "generation": int(r.get("generation", 9)),
                "family": r.get("family", "external"),
                "label_human": 0,
                "klass": "synthetic",
                "era": "recent",
                "witnesses": [],
                "url": None,
                "claimed_date": None,
            }
        )
    corpus.synthetic = docs

    n_backdate = len(corpus.backdated)
    pool = [d for d in docs]
    rng.shuffle(pool)
    corpus.backdated = []
    for i, d in enumerate(pool[:n_backdate]):
        forged = _forged_date(rng)
        template = FORGERY_HOSTS[i % len(FORGERY_HOSTS)]
        slug = hashlib.blake2b(d["doc_id"].encode(), digest_size=4).hexdigest()
        corpus.backdated.append(
            {
                "doc_id": f"backdate:{d['doc_id']}",
                "text": backdate(d["text"], forged, seed=seed + i),
                "label_human": 0,
                "klass": "backdated",
                "era": "recent",
                "url": template.format(n=slug),
                "claimed_date": forged.isoformat(),
                "witnesses": [],
                "generation": d["generation"],
                "generator": d["generator"],
                "forged_date": forged.isoformat(),
            }
        )
    return corpus


def archive_entries(corpus: EvalCorpus) -> list[dict]:
    """Shape the archive split for :meth:`dendro.alignment.ArchiveLayer.add`."""
    return [
        {
            "doc_id": d["doc_id"],
            "text": d["text"],
            "not_after": d["not_after"],
            "url": d.get("url"),
            "meta": {"category": d.get("category"), "genre": d.get("genre")},
        }
        for d in corpus.archive
    ]
