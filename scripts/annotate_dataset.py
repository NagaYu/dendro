"""Attach Dendro provenance columns to any Hugging Face dataset.

Two entry points:

* :func:`dendro_map_fn` returns a callable for ``dataset.map(..., batched=True)``,
  so annotation composes with the rest of a `datasets` pipeline;
* the CLI does the whole job and optionally pushes the result.

::

    python -m scripts.annotate_dataset wikitext --config wikitext-2-raw-v1 \\
        --split train --limit 2000 --out NagaYu/wikitext-dendro --push

    from scripts.annotate_dataset import dendro_map_fn
    ds = ds.map(dendro_map_fn(url_column="url"), batched=True, batch_size=32)

**Columns written** (prefix configurable):

===========================  ==========================================================
``dendro_not_after``         ISO date the content is proven to have existed by, or null
``dendro_not_after_year``    integer year, convenient for filtering
``dendro_human_origin_p``    calibrated probability, *not* a label
``dendro_ci_low/​ci_high``    90% credible interval — the honest width of the claim
``dendro_abstained``         true when the evidence is too thin to act on
``dendro_operators``         count of *independent* operators behind the bound
``dendro_evidence_logodds``  how hard the bound would be to forge
``dendro_flags``             inconsistencies, e.g. ``backdate``
``dendro_explanation``       human-readable receipt, with witness sources named
===========================  ==========================================================

Note what is *not* written: there is no ``is_synthetic`` column and no boolean
verdict. Downstream code that wants a filter must choose a threshold explicitly
and, ideally, use ``dendro_ci_low`` so the choice is conservative. Shipping a
label would let a probability with a wide interval travel as a fact.

Cost note: annotation is I/O against archives, so it is dominated by cache hit
rate rather than compute. Rows sharing a host are nearly free after the first —
``dendro_cache_hit_rate`` is printed at the end so the marginal cost is visible.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dendro.cache import Cache, HttpClient, RateLimiter  # noqa: E402
from dendro.pipeline import Dendro  # noqa: E402
from dendro.types import Verdict  # noqa: E402

COLUMNS = (
    "not_after",
    "not_after_year",
    "human_origin_p",
    "ci_low",
    "ci_high",
    "abstained",
    "operators",
    "evidence_logodds",
    "flags",
    "ancestor",
    "explanation",
)


def verdict_columns(verdict: Verdict, prefix: str = "dendro_") -> dict[str, Any]:
    """One verdict -> the flat columns written to the dataset."""
    na = verdict.not_after
    return {
        f"{prefix}not_after": na.date().isoformat() if na else None,
        f"{prefix}not_after_year": na.year if na else None,
        f"{prefix}human_origin_p": round(float(verdict.human_origin_p), 6),
        f"{prefix}ci_low": round(float(verdict.ci_low), 6),
        f"{prefix}ci_high": round(float(verdict.ci_high), 6),
        f"{prefix}abstained": bool(verdict.abstained),
        f"{prefix}operators": int(verdict.bound.independent_operators),
        f"{prefix}evidence_logodds": round(float(verdict.bound.forgery_logodds), 4),
        f"{prefix}flags": [f.kind for f in verdict.flags],
        f"{prefix}ancestor": verdict.ancestor.ref_doc_id if verdict.ancestor else None,
        f"{prefix}explanation": verdict.explanation,
    }


def build_dendro(
    cache_dir: Optional[str] = None,
    offline: Optional[bool] = None,
    archive: Optional[Sequence[Mapping]] = None,
) -> Dendro:
    """Construct a Dendro engine sharing one cache and one rate-limit budget.

    ``offline=None`` defers to ``DENDRO_OFFLINE``; only an explicit ``True``
    forces offline mode. Passing ``False`` unconditionally would override the
    environment variable the docs advertise, which is how a "cached-only" run
    quietly turns into a few hundred live archive requests.
    """
    client = HttpClient(
        cache=Cache(root=cache_dir) if cache_dir else Cache(),
        rate_limiter=RateLimiter(),
        offline=offline,
    )
    d = Dendro(client=client)
    if archive:
        d.index_archive(archive)
    return d


def dendro_map_fn(
    text_column: str = "text",
    url_column: Optional[str] = None,
    date_column: Optional[str] = None,
    id_column: Optional[str] = None,
    prefix: str = "dendro_",
    probe_coverage: bool = False,
    dendro: Optional[Dendro] = None,
) -> Callable[[dict[str, list]], dict[str, list]]:
    """Build a ``batched=True`` map function.

    The :class:`~dendro.pipeline.Dendro` instance is created once and closed
    over, so the HTTP cache and the rate-limit budget are shared across the whole
    ``map`` — which is the difference between annotating a 50k-row dataset in
    minutes and being throttled off an archive's API.

    ``probe_coverage`` defaults to *off*: it roughly doubles the request count and
    only matters when rows carry claimed dates you want checked for backdating.
    """
    engine = dendro if dendro is not None else build_dendro()

    def _fn(batch: dict[str, list]) -> dict[str, list]:
        n = len(batch[text_column]) if text_column in batch else len(next(iter(batch.values())))
        out: dict[str, list] = {f"{prefix}{c}": [] for c in COLUMNS}
        for i in range(n):
            text = (batch.get(text_column) or [None] * n)[i]
            url = (batch.get(url_column) or [None] * n)[i] if url_column else None
            claimed = (batch.get(date_column) or [None] * n)[i] if date_column else None
            doc_id = (batch.get(id_column) or [None] * n)[i] if id_column else None
            verdict = engine.date(
                text=text,
                url=url,
                doc_id=str(doc_id) if doc_id else (url or f"row{i}"),
                claimed_date=claimed or None,
                probe_coverage=probe_coverage,
            )
            for k, v in verdict_columns(verdict, prefix).items():
                out[k].append(v)
        return out

    _fn.dendro = engine  # type: ignore[attr-defined]
    return _fn


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="HF dataset id, or a local .jsonl path")
    ap.add_argument("--config", help="dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--url-column")
    ap.add_argument("--date-column")
    ap.add_argument("--id-column")
    ap.add_argument("--prefix", default="dendro_")
    ap.add_argument("--probe-coverage", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-proc", type=int, default=1,
                    help="keep at 1: parallel workers would each open their own cache and "
                         "multiply the request rate against public archives")
    ap.add_argument("--cache", help="cache directory")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--archive", help="JSONL of dated documents to align against")
    ap.add_argument("--out", help="HF repo id or local path to write")
    ap.add_argument("--push", action="store_true", help="push_to_hub the annotated dataset")
    ap.add_argument("--private", action="store_true")
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
    print(f"loaded {len(ds)} rows; columns: {list(ds.features)}")

    archive = None
    if args.archive:
        from dendro.cli import _load_archive

        archive = _load_archive(pathlib.Path(args.archive))
        print(f"archive layer: {len(archive)} dated documents")

    engine = build_dendro(args.cache, True if args.offline else None, archive)
    fn = dendro_map_fn(
        text_column=args.text_column,
        url_column=args.url_column,
        date_column=args.date_column,
        id_column=args.id_column,
        prefix=args.prefix,
        probe_coverage=args.probe_coverage,
        dendro=engine,
    )
    ds = ds.map(fn, batched=True, batch_size=args.batch_size, num_proc=args.num_proc,
                desc="dendro provenance")

    stats = engine.stats.as_row()
    print(json.dumps(stats, indent=2))
    print(f"cache hit rate: {stats['hit_rate']:.1%}   "
          f"network calls per document: {stats['network_calls'] / max(1, len(ds)):.2f}")

    if args.out:
        if args.push:
            ds.push_to_hub(args.out, private=args.private)
            print(f"pushed to https://huggingface.co/datasets/{args.out}")
        else:
            ds.to_json(args.out, orient="records", lines=True)
            print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
