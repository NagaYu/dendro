"""Carve a low-background subset out of a dataset and publish it.

    python -m scripts.build_low_background NagaYu/some-corpus \\
        --max-synthetic 0.05 --mode conservative \\
        --out NagaYu/some-corpus-lowbackground --push

The selection is a *maximum-cardinality* solution, not a heuristic: see
:func:`dendro.corpus_report.build_low_background_subset` for the exchange
argument that makes sorting by probability and taking the longest feasible
prefix provably optimal.

``--mode conservative`` selects on the lower end of each document's credible
interval rather than the point estimate, so the purity constraint holds even if
every document sits at the pessimistic end of what its evidence supports. That is
the mode to use when publishing, because an interval is a statement about what is
*not* known and a published dataset should not quietly spend that budget on the
optimistic reading.

The generated dataset card records the constraint, the mode, the retention, the
prevalence curve in force, and the intended-use limits — so a downstream user who
reads only the card still learns that the subset is defined by *archival evidence
of prior existence* and not by a judgement about authorship.
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

from dendro import __version__  # noqa: E402
from dendro.corpus_report import CorpusReport  # noqa: E402
from dendro.propagate import PrevalenceCurve  # noqa: E402
from dendro.types import Verdict  # noqa: E402

from .annotate_dataset import build_dendro, dendro_map_fn, verdict_columns  # noqa: E402

CARD = """---
license: unknown
task_categories:
  - text-generation
tags:
  - provenance
  - low-background
  - data-curation
  - dendro
---

# {name}

A **low-background** subset of `{source}`, selected by archival evidence of prior
existence rather than by inspecting the writing.

The name is from metallurgy. Low-background steel is steel smelted before the 1945
atmospheric tests: not special steel, just ordinary steel that happens to predate the
contamination, and valuable because no amount of later care can reproduce it. Text
archived before large-scale generation has the same property and the same scarcity.

## How it was selected

Every document was dated by [Dendro](https://github.com/NagaYu/dendro) v{version}, which
asks independent archives — the Internet Archive, Common Crawl, arXiv, Crossref, public
posting archives — what they observed and when. Documents were then chosen to **maximise
subset size subject to an expected synthetic fraction below {constraint:.1%}**. The
selection is provably maximum-cardinality, not greedy-approximate.

| | |
|---|---|
| source dataset | `{source}` |
| documents considered | {n_total} |
| documents selected | {n_selected} |
| retention | {retention:.1%} |
| purity constraint | ≤ {constraint:.1%} expected synthetic |
| achieved (expected) | {achieved:.2%} |
| selection mode | `{mode}` |
| probability threshold | {threshold:.3f} |
| prevalence curve | logistic, midpoint {midpoint}, floor {floor}, ceiling {ceiling} |

## Columns

Dendro columns are prefixed `dendro_`. `dendro_not_after` is the date the content is
**proven to have existed by**; `dendro_human_origin_p` is a calibrated probability with
`dendro_ci_low` / `dendro_ci_high` bounding it; `dendro_abstained` marks documents whose
evidence was too thin to act on.

## Limitations — please read before using

- **This is not an AI-writing detector.** Selection is by *evidence that content existed
  before a date*. It says nothing about who or what wrote any particular document.
- **Absence of evidence is not evidence.** Most text that has ever existed was never
  archived. Documents were excluded for lack of a record, which is not a finding against
  them.
- **The probability depends on an assumed prevalence curve** — the machine-generated share
  of new text over time. That curve is an assumption, stated above, and it moves the
  absolute numbers.
- **Selection is not neutral.** Archived text over-represents the web that crawlers
  reached: English, institutional, indexed, long-lived. A low-background subset inherits
  every one of those biases and concentrates them.
- **Do not use these columns to make judgements about individual people.**

Generated {today} by `scripts/build_low_background.py`.
"""


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="HF dataset id or local .jsonl")
    ap.add_argument("--config")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--url-column")
    ap.add_argument("--date-column")
    ap.add_argument("--id-column")
    ap.add_argument("--max-synthetic", type=float, default=0.05)
    ap.add_argument("--mode", choices=("expected", "conservative"), default="conservative")
    ap.add_argument("--probe-coverage", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cache")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--archive", help="JSONL of dated documents to align against")
    ap.add_argument("--out", help="HF repo id or local path")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--report", help="write the full provenance report as JSON")
    args = ap.parse_args(argv)

    from datasets import Dataset, load_dataset

    path = pathlib.Path(args.dataset)
    if path.is_file():
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        ds = Dataset.from_list(rows[: args.limit] if args.limit else rows)
    else:
        ds = load_dataset(args.dataset, args.config, split=args.split)
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
    print(f"loaded {len(ds)} rows")

    archive = None
    if args.archive:
        from dendro.cli import _load_archive

        archive = _load_archive(pathlib.Path(args.archive))

    engine = build_dendro(args.cache, True if args.offline else None, archive)

    # Score every row, keeping the Verdict objects so the subset solver sees the
    # intervals rather than only the point estimates.
    verdicts: list[Verdict] = []
    for i in range(0, len(ds), args.batch_size):
        batch = ds[i : i + args.batch_size]
        n = len(batch[args.text_column])
        for j in range(n):
            verdicts.append(
                engine.date(
                    text=batch[args.text_column][j],
                    url=batch[args.url_column][j] if args.url_column else None,
                    doc_id=str(batch[args.id_column][j]) if args.id_column else f"row{i + j}",
                    claimed_date=batch[args.date_column][j] if args.date_column else None,
                    probe_coverage=args.probe_coverage,
                )
            )
        print(f"  scored {min(i + args.batch_size, len(ds))}/{len(ds)}", end="\r", flush=True)
    print()

    report = CorpusReport.from_verdicts(verdicts)
    plan = report.subset(args.max_synthetic, mode=args.mode)
    print(json.dumps(plan.as_row(), indent=2))
    print()
    print("purity / retention trade-off:")
    for row in report.subset_curve(mode=args.mode):
        print(f"  <= {row['constraint']:<6} keeps {row['n_selected']:>7} ({row['retention']:.1%})")

    keep = {doc_id for doc_id in plan.doc_ids}
    index = {v.doc_id: i for i, v in enumerate(verdicts)}
    selected_rows = sorted(index[d] for d in keep if d in index)
    subset = ds.select(selected_rows)

    cols = [verdict_columns(verdicts[i]) for i in selected_rows]
    for name in cols[0] if cols else []:
        subset = subset.add_column(name, [c[name] for c in cols])

    if args.report:
        pathlib.Path(args.report).write_text(
            json.dumps({"summary": report.summary(), "plan": plan.as_row(),
                        "curve": report.subset_curve(mode=args.mode)}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"wrote report to {args.report}")

    if args.out:
        curve = PrevalenceCurve()
        card = CARD.format(
            name=args.out.split("/")[-1],
            source=args.dataset,
            version=__version__,
            n_total=plan.n_total,
            n_selected=plan.n_selected,
            retention=plan.retention,
            constraint=plan.constraint,
            achieved=plan.expected_synthetic_fraction,
            mode=plan.mode,
            threshold=plan.threshold_p,
            midpoint=curve.midpoint_year,
            floor=curve.floor,
            ceiling=curve.ceiling,
            today=_dt.date.today().isoformat(),
        )
        if args.push:
            subset.push_to_hub(args.out, private=args.private)
            try:
                from huggingface_hub import DatasetCard

                DatasetCard(card).push_to_hub(args.out, repo_type="dataset")
            except Exception as exc:  # card is a nicety; the data is the deliverable
                print(f"(card upload skipped: {exc})")
            print(f"pushed to https://huggingface.co/datasets/{args.out}")
        else:
            out = pathlib.Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            subset.to_json(str(out), orient="records", lines=True)
            out.with_suffix(".README.md").write_text(card, encoding="utf-8")
            print(f"wrote {out} and {out.with_suffix('.README.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
