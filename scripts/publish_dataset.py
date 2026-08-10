"""Publish the Dendro-annotated evaluation corpus as a Hugging Face dataset.

    python -m scripts.publish_dataset --out NagaYu/dendro-lowbackground --push

What gets published is the real corpus from ``data/corpus/*.jsonl`` — 5,034 arXiv
records — with Dendro's provenance columns attached, plus a ``low_background``
flag marking the subset whose *pre-2021 existence is proven by an independent
registration record*.

Two decisions worth stating, because they change how the numbers should be read:

**The calibrator is left unfitted, on purpose.** A fitted isotonic map encodes the
base rate of whatever corpus it was fitted on, and baking that into a published
artefact would make every row's probability depend on a benchmark split nobody
downstream can see. The columns here are therefore the *raw* inference: evidence
strength and the prevalence curve, nothing else.

**Witnesses are real, not simulated.** ``dendro_not_after`` comes from arXiv's own
OAI-PMH ``<created>`` field — the v1 submission date, which is exactly what
:class:`dendro.sources.scholarly.ArxivSource` returns from a live query. The
committed cache in ``data/fixtures/cache`` holds the responses.

A single-operator caveat travels with the data: every row is witnessed by arXiv
and arXiv alone, so ``dendro_operators`` is 1 throughout and ``dendro_flags``
contains ``single_operator``. That is a real limitation of this corpus, not a bug,
and the card says so.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
from typing import Optional

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.corpus import arxiv_witness, load_real_documents  # noqa: E402
from dendro import __version__  # noqa: E402
from dendro.corpus_report import CorpusReport  # noqa: E402
from dendro.propagate import ContaminationPropagator, PrevalenceCurve  # noqa: E402
from dendro.sources.selfasserted import SelfAssertedSource  # noqa: E402
from dendro.witness import ConsensusConfig, Target, WitnessCollector  # noqa: E402

from .annotate_dataset import verdict_columns  # noqa: E402

CUTOFF = _dt.datetime(2021, 1, 1, tzinfo=_dt.timezone.utc)

CARD = """---
license: cc0-1.0
task_categories:
  - text-classification
  - text-generation
language:
  - en
size_categories:
  - 1K<n<10K
tags:
  - provenance
  - low-background
  - data-curation
  - dataset-contamination
  - dendro
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
---

# Dendro low-background corpus

{n_total} arXiv records annotated with **archival evidence of when they existed**, produced by
[Dendro](https://github.com/NagaYu/dendro) v{version}.

{n_low} of them ({low_pct:.1%}) are **low-background**: an independent registration record
places them before {cutoff}, i.e. before large-scale text generation. The name is from
metallurgy — low-background steel is steel smelted before the 1945 atmospheric tests: not
special steel, just ordinary steel that happens to predate the contamination, and valuable
because no amount of later care can reproduce it.

## Read this before you use it

- **This is not an AI-writing detector, and these columns must not be used as one.** They
  report *evidence that content existed before a date*. Nothing here supports a conclusion
  that a particular person did or did not write something.
- **Absence of evidence is not evidence.** Most text that has ever existed was never archived.
  `dendro_abstained` marks rows where the evidence is too thin to act on — those are
  *unknown*, not *generated*.
- **Every row is witnessed by arXiv and arXiv alone.** `dendro_operators` is 1 throughout, and
  `dendro_flags` contains `single_operator`. Compromising one organisation would be sufficient
  to move every date in this dataset. Dendro's whole argument is that confidence should come
  from *independent operators*; this corpus has one, and the column says so rather than
  hiding it.
- **The recent split is "human-attributed", not "verified human".** By 2025 an unknown share
  of abstracts have been through a model. There is no clean recent human corpus to be had —
  which is the argument for dating by archive rather than by inspection.
- **Selection is not neutral.** Archived text over-represents what crawlers reached: English,
  institutional, indexed, long-lived. A low-background subset concentrates every one of those
  biases.

## Columns

| column | meaning |
|---|---|
| `text` | title + abstract, as fingerprinted |
| `title`, `doi` | the record's own metadata |
| `url`, `arxiv_id`, `category`, `era` | identifiers and provenance of the record |
| `published` | arXiv v1 submission date — the witness |
| `dendro_not_after` | date the content is **proven to have existed by** |
| `dendro_not_after_year` | integer year, convenient for filtering |
| `dendro_human_origin_p` | calibrated probability — **not** a label |
| `dendro_ci_low` / `dendro_ci_high` | 90% credible interval, the honest width of the claim |
| `dendro_abstained` | true when the evidence is too thin to act on |
| `dendro_operators` | count of *independent* operators behind the bound |
| `dendro_evidence_logodds` | how hard the bound would be to forge |
| `dendro_flags` | detected inconsistencies |
| `dendro_explanation` | human-readable receipt naming the witness sources |
| `low_background` | proven pre-{cutoff_year} existence **and** `dendro_ci_low` ≥ {min_p} |

### Why the threshold is {min_p} and not 0.95

Because that is what the evidence in *this* corpus can actually support, and moving the
threshold to flatter the number would be the whole problem with provenance tooling.

Every row here rests on **one operator**. Under Dendro's model a single arXiv registration
gives an evidence log-odds of ~6.5, which puts the *lower* end of the 90% credible interval at
about 0.85 no matter how trustworthy arXiv is — the interval is wide because corroboration is
absent, not because the record is doubtful. At `ci_low ≥ 0.90` this dataset would contain
**zero** low-background rows, which is a true statement about single-operator evidence and a
useless artefact.

So the flag is set where single-operator registration evidence genuinely reaches, and the
number that would tighten it is `dendro_operators` — not the threshold. A document witnessed
by arXiv *and* the Internet Archive *and* Common Crawl clears 0.95 comfortably; filter on
`dendro_ci_low` yourself if you want a different line.

There is deliberately **no** `is_synthetic` column and no boolean verdict. Code that wants a
filter must pick a threshold explicitly, and should prefer `dendro_ci_low` so the choice is
conservative.

## Three lines to use it

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", split="train")
clean = ds.filter(lambda r: r["low_background"])          # {n_low} rows
```

## Purity / retention trade-off

Selecting to bound the *expected* synthetic fraction (provably maximum-cardinality — see
`dendro.corpus_report.build_low_background_subset`):

{curve}

## Method, in one paragraph

Dendro asks independent archives — the Internet Archive, Common Crawl, arXiv, Crossref, public
posting archives — what they observed and when, groups witnesses by **operator** rather than by
count (twenty captures from one archive are one archive), and reports the earliest time whose
surviving evidence clears a failure-probability budget. It never reads the prose, which is why
its accuracy does not move when a new generator ships. Probabilities are produced by a mixture
that collapses to the base rate when evidence is absent, so "no evidence" cannot be reported as
"synthetic".

The calibrator is **left unfitted here on purpose**: a fitted map encodes the base rate of the
corpus it was fitted on, and baking that into a published artefact would make every row depend
on a split nobody downstream can inspect. These are the raw numbers.

## Provenance of the data itself

arXiv metadata, including abstracts, is offered under
[CC0 1.0](https://info.arxiv.org/help/api/tou.html) through the public API and OAI-PMH, which
is how it was harvested. Thank you to arXiv for use of its open access interoperability.

Generated {today} by `scripts/publish_dataset.py` · Apache-2.0 code, CC0 data.
"""


def build_rows(min_p: float = 0.90) -> tuple[list[dict], CorpusReport]:
    """Annotate the committed corpus with real arXiv registration evidence."""
    old, recent, _ = load_real_documents()
    if not old:
        raise SystemExit("no corpus — run `python -m scripts.fetch_corpus` first")

    collector = WitnessCollector(sources=[], client=_offline_client(), config=ConsensusConfig())
    propagator = ContaminationPropagator()          # unfitted calibrator, by design
    self_asserted = SelfAssertedSource()

    rows: list[dict] = []
    verdicts = []
    for era, records in (("pre2021", old), ("recent", recent)):
        for rec in records:
            target = Target(
                doc_id=rec["doc_id"],
                url=rec.get("url"),
                text=rec.get("text"),
                arxiv_id=rec.get("arxiv_id"),
            )
            witnesses = [arxiv_witness(rec), *self_asserted.collect(target, collector.client)]
            bound = collector.consensus(witnesses)
            flags = collector.detect_inconsistencies(target, witnesses, bound, coverage={})
            verdict = propagator.verdict(rec["doc_id"], bound, None, flags)
            verdicts.append(verdict)

            cols = verdict_columns(verdict)
            na = verdict.not_after
            rows.append(
                {
                    "doc_id": rec["doc_id"],
                    "arxiv_id": rec.get("arxiv_id"),
                    "title": rec.get("title"),
                    "text": rec.get("text"),
                    "url": rec.get("url"),
                    "doi": rec.get("doi"),
                    "category": rec.get("category"),
                    "published": rec.get("published"),
                    "era": era,
                    **cols,
                    "low_background": bool(na is not None and na < CUTOFF and verdict.ci_low >= min_p),
                }
            )
    return rows, CorpusReport.from_verdicts(verdicts)


def _offline_client():
    from dendro.cache import Cache, HttpClient

    return HttpClient(cache=Cache(root=REPO / "data" / "fixtures" / "cache"), offline=True)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="NagaYu/dendro-lowbackground")
    ap.add_argument("--min-p", type=float, default=0.80)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--local", help="also write the rows to a local JSONL")
    args = ap.parse_args(argv)

    print("annotating corpus ...", flush=True)
    rows, report = build_rows(args.min_p)
    n_low = sum(r["low_background"] for r in rows)
    print(f"  {len(rows)} rows, {n_low} low-background ({n_low / len(rows):.1%})")

    curve_rows = report.subset_curve(mode="conservative")
    curve = "\n".join(
        ["| max expected synthetic | rows kept | retention |", "|---|---|---|"]
        + [f"| ≤ {r['constraint']:.0%} | {r['n_selected']} | {r['retention']:.1%} |" for r in curve_rows]
    )

    if args.local:
        pathlib.Path(args.local).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )
        print(f"  wrote {args.local}")

    from datasets import Dataset

    ds = Dataset.from_list(rows)
    prevalence = PrevalenceCurve()
    card = CARD.format(
        repo_id=args.out,
        version=__version__,
        n_total=len(rows),
        n_low=n_low,
        low_pct=n_low / len(rows),
        cutoff=CUTOFF.date().isoformat(),
        cutoff_year=CUTOFF.year,
        min_p=args.min_p,
        curve=curve,
        today=_dt.date.today().isoformat(),
    )

    if args.push:
        ds.push_to_hub(args.out, private=args.private)
        from huggingface_hub import DatasetCard

        DatasetCard(card).push_to_hub(args.out, repo_type="dataset")
        print(f"pushed https://huggingface.co/datasets/{args.out}")
    else:
        out = REPO / "data" / "local" / "dataset_card.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(card, encoding="utf-8")
        print(f"dry run — card written to {out}; pass --push to publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
