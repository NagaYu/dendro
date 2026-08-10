"""`dendro` command line.

    dendro date https://example.org/post-from-2019
    dendro date paper.txt --claimed 2019-03-11
    dendro report NagaYu/some-dataset --split train --limit 500
    dendro subset NagaYu/some-dataset --max-synthetic 0.05 --out clean.jsonl
    dendro fingerprint page.html --show-normalized
    dendro sources
    dendro cache

Every command prints the *evidence* before the probability.  That ordering is
deliberate: the evidence is checkable by the reader (each witness carries a URL
you can open), while the probability additionally depends on an assumed
prevalence curve.  A user who trusts the number without reading the witnesses
has been handed the least reliable part of the output.

``date`` exits 0 when a bound was established, 3 when Dendro abstains for lack of
evidence, and 4 when an inconsistency was flagged, so it composes in a script
without parsing the text.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Iterable, Optional, Sequence

from . import __version__

EXIT_OK, EXIT_ERROR, EXIT_ABSTAIN, EXIT_FLAGGED = 0, 2, 3, 4

DISCLAIMER = (
    "Dendro reports archival evidence that content existed before a date. It is not an "
    "authorship test. Do not use it to conclude that a particular person did or did not "
    "write something."
)


# --------------------------------------------------------------------------- helpers
def _read_input(target: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (text, url, path) for a CLI target that may be either."""
    if target.startswith(("http://", "https://")):
        return None, target, None
    path = pathlib.Path(target)
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace"), None, str(path)
    if target == "-":
        return sys.stdin.read(), None, None
    raise SystemExit(f"not a URL and not a readable file: {target!r}")


def _table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "  (none)"
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    head = "  " + "  ".join(c.ljust(widths[c]) for c in columns)
    rule = "  " + "  ".join("-" * widths[c] for c in columns)
    body = [
        "  " + "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows
    ]
    return "\n".join([head, rule, *body])


def _build_pipeline(args: argparse.Namespace):
    from .cache import Cache, HttpClient, RateLimiter
    from .pipeline import Dendro
    from .sources import build_sources
    from .witness import ConsensusConfig

    client = HttpClient(
        cache=Cache(root=args.cache) if args.cache else Cache(),
        rate_limiter=RateLimiter(),
        # ``None`` means "consult DENDRO_OFFLINE".  Passing ``False`` here would
        # override the environment variable that the README and --help both
        # advertise as the way to force offline mode, so the flag can only ever
        # turn offline *on*, never silently off.
        offline=True if args.offline else None,
    )
    sources = build_sources(args.sources.split(",")) if args.sources else None
    return Dendro(
        sources=sources, client=client, consensus_config=ConsensusConfig(alpha=args.alpha)
    )


# --------------------------------------------------------------------------- date
def cmd_date(args: argparse.Namespace) -> int:
    """Date one document and print its evidence.

    Demonstrates **adversarial-robustness** and **generator-independence** as a
    workflow rather than a claim: the output is a witness table with clickable
    archive URLs, and nothing in the pipeline consulted a language model.
    """
    from .types import to_utc

    text, url, path = _read_input(args.target)
    dendro = _build_pipeline(args)

    if args.archive:
        dendro.index_archive(_load_archive(pathlib.Path(args.archive)))

    verdict = dendro.date(
        text=text,
        url=url,
        path=path,
        doc_id=args.target,
        claimed_date=to_utc(args.claimed) if args.claimed else None,
        probe_coverage=not args.no_coverage,
    )

    if args.json:
        print(json.dumps({**verdict.as_row(), "explanation": verdict.explanation}, indent=2))
    else:
        print(f"target: {args.target}")
        print()
        print("witnesses")
        print(_table(
            [w.as_row() for w in verdict.bound.all_witnesses],
            ["source", "operator", "kind", "observed_at", "reliability", "forgeability", "cached"],
        ))
        print()
        na = verdict.not_after
        # Report both counts.  Only witnesses at or before the bound *support* it,
        # so a table showing three operators next to "independent ops: 1" reads as
        # a bug unless the distinction is spelled out -- and the supporting count
        # is the one the confidence is computed from.
        all_ops = {w.operator for w in verdict.bound.all_witnesses if w.is_independent_evidence}
        print(f"existence bound : {na.date().isoformat() if na else 'none'}")
        print(
            f"independent ops : {verdict.bound.independent_operators} supporting the bound"
            f"  ({len(all_ops)} distinct operator(s) seen in total)"
        )
        print(f"evidence log-odds: {verdict.bound.forgery_logodds:.2f}")
        print(f"P(human-origin) : {verdict.human_origin_p:.3f}  "
              f"(90% interval {verdict.ci_low:.2f}-{verdict.ci_high:.2f})")
        if verdict.abstained:
            print("verdict         : ABSTAIN - insufficient archival evidence")
        print()
        print(verdict.explanation)
        print()
        print(f"cache: {json.dumps(dendro.stats.as_row())}")
    print(f"\n{DISCLAIMER}", file=sys.stderr)

    if verdict.flags and any(f.severity in ("high", "medium") for f in verdict.flags):
        return EXIT_FLAGGED
    return EXIT_ABSTAIN if verdict.abstained else EXIT_OK


def _load_archive(path: pathlib.Path) -> list[dict]:
    """Load an archive layer from JSONL (``doc_id``, ``text``, ``not_after``)."""
    if not path.exists():
        raise SystemExit(f"archive not found: {path}")
    out: list[dict] = []
    files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            when = rec.get("not_after") or rec.get("published")
            if not (rec.get("text") and when):
                continue
            out.append(
                {
                    "doc_id": rec.get("doc_id") or rec.get("id") or rec.get("url"),
                    "text": rec["text"],
                    "not_after": when,
                    "url": rec.get("url"),
                }
            )
    return out


# --------------------------------------------------------------------------- report
def cmd_report(args: argparse.Namespace) -> int:
    """Provenance report over a dataset -- distribution and low-background ratio."""
    from .corpus_report import CorpusReport

    records = _load_dataset(args.dataset, args.split, args.limit, args.text_column)
    if not records:
        print("no records loaded", file=sys.stderr)
        return EXIT_ERROR

    dendro = _build_pipeline(args)
    if args.archive:
        dendro.index_archive(_load_archive(pathlib.Path(args.archive)))

    verdicts = list(dendro.date_many(records, probe_coverage=args.probe_coverage).values())
    report = CorpusReport.from_verdicts(verdicts)
    summary = report.summary(min_p=args.min_p)

    if args.json:
        print(json.dumps({"summary": summary, "subset_curve": report.subset_curve()}, indent=2, default=str))
    else:
        print(f"dataset: {args.dataset}  ({summary['n']} documents)")
        print()
        print(f"  low-background (proven pre-{summary['lowbackground_cutoff'][:4]}, "
              f"p>={args.min_p}): {summary['lowbackground_low_background']:.1%}")
        print(f"  unknown (no archival evidence)          : {summary['lowbackground_unknown']:.1%}")
        print(f"  post-cutoff existence                   : {summary['lowbackground_contaminated']:.1%}")
        print()
        print(f"  mean P(human-origin) : {summary['mean_human_origin_p']:.3f}")
        print(f"  mean interval width  : {summary['mean_ci_width']:.3f}")
        print(f"  abstain rate         : {summary['abstain_rate']:.1%}")
        print(f"  flagged              : {summary['flagged_rate']:.1%}  {summary['flag_kinds'] or ''}")
        print()
        print("  evidence by operator:")
        for op, n in list(summary["evidence"]["operators"].items())[:8]:
            print(f"    {op:24s} {n}")
        print()
        print("  purity / retention trade-off:")
        print(_table(report.subset_curve(), ["constraint", "n_selected", "retention", "threshold_p"]))
        print()
        print("  existence-year profile:")
        print(_table(report.year_histogram(), ["year", "count"]))
        print()
        print(f"cache: {json.dumps(dendro.stats.as_row())}")

    if args.out:
        pathlib.Path(args.out).write_text(
            "\n".join(json.dumps(r) for r in report.to_rows()), encoding="utf-8"
        )
        print(f"wrote per-document verdicts to {args.out}")
    print(f"\n{DISCLAIMER}", file=sys.stderr)
    return EXIT_OK


def cmd_subset(args: argparse.Namespace) -> int:
    """Build a subset whose expected synthetic fraction stays under a bound."""
    from .corpus_report import CorpusReport

    records = _load_dataset(args.dataset, args.split, args.limit, args.text_column)
    dendro = _build_pipeline(args)
    if args.archive:
        dendro.index_archive(_load_archive(pathlib.Path(args.archive)))
    by_id = {r.get("doc_id") or r.get("id") or r.get("url"): r for r in records}
    verdicts = list(dendro.date_many(records).values())

    report = CorpusReport.from_verdicts(verdicts)
    plan = report.subset(args.max_synthetic, mode=args.mode)
    print(json.dumps(plan.as_row(), indent=2))

    if args.out:
        keep = set(plan.doc_ids)
        with pathlib.Path(args.out).open("w", encoding="utf-8") as fh:
            for doc_id in plan.doc_ids:
                rec = by_id.get(doc_id)
                if rec:
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"wrote {len(keep)} records to {args.out}")
    print(f"\n{DISCLAIMER}", file=sys.stderr)
    return EXIT_OK


def _load_dataset(name: str, split: str, limit: Optional[int], text_column: str) -> list[dict]:
    """Load from a local JSONL path or the Hugging Face Hub."""
    path = pathlib.Path(name)
    records: list[dict] = []
    if path.is_file():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if limit and i >= limit:
                break
            if line.strip():
                records.append(json.loads(line))
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit("install `datasets` to read from the Hugging Face Hub")
        ds = load_dataset(name, split=split)
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        records = [dict(r) for r in ds]

    out: list[dict] = []
    for i, r in enumerate(records):
        text = r.get(text_column) or r.get("text") or r.get("content") or ""
        out.append(
            {
                "doc_id": str(r.get("doc_id") or r.get("id") or r.get("url") or f"row{i}"),
                "text": text,
                "url": r.get("url"),
                "claimed_date": r.get("date") or r.get("published") or r.get("timestamp"),
            }
        )
    return out


# --------------------------------------------------------------------------- inspect
def cmd_fingerprint(args: argparse.Namespace) -> int:
    """Show the normalised form and channel sketch of a document.

    Useful for seeing **reflow-invariance** directly: run it on an HTML page and
    on a plain-text copy of the same prose and compare ``normalized_sha256``.
    """
    from .fingerprint import ReflowFingerprint

    text, _, _ = _read_input(args.target)
    if text is None:
        raise SystemExit("fingerprint needs a file or stdin, not a URL")
    rf = ReflowFingerprint()
    nd = rf.normalizer.normalize(text)
    fp = rf.fingerprint(args.target, text)
    print(f"normalized sha256 : {fp.normalized_sha256}")
    print(f"tokens            : {fp.n_tokens}")
    print(f"windows           : {len(fp.windows)}")
    print(f"dropped lines     : {fp.meta['dropped_lines']}")
    print("channels:")
    for name, ch in fp.channels.items():
        print(f"  {name:6s} cardinality={ch.cardinality:6d} simhash={ch.simhash:#018x}")
    if args.show_normalized:
        print()
        print("--- normalized ---")
        print(nd.text[: args.chars])
    return EXIT_OK


def cmd_sources(args: argparse.Namespace) -> int:
    """List witness sources with the independence assumptions they carry."""
    from .sources import default_sources

    rows = [
        {
            "source": s.source_id,
            "operator": s.operator,
            "kind": s.kind.value,
            "reliability": s.reliability,
            "forgeability": s.forgeability,
            "evidence": "yes" if s.kind.is_independent_evidence else "NO (claim only)",
        }
        for s in default_sources()
    ]
    print(_table(rows, ["source", "operator", "kind", "reliability", "forgeability", "evidence"]))
    print()
    print("Witnesses are grouped by OPERATOR, not by source: twenty captures from one")
    print("archive count as one operator. Adding a second operator multiplies the")
    print("difficulty of forging a bound; adding a second capture barely moves it.")
    return EXIT_OK


def cmd_cache(args: argparse.Namespace) -> int:
    """Report on the on-disk cache that makes offline replay possible."""
    from .cache import Cache, default_cache_dir, repo_fixture_cache

    root = pathlib.Path(args.cache) if args.cache else default_cache_dir()
    fixture = repo_fixture_cache()
    for label, path in (("writable", root), ("fixtures", fixture)):
        n = sum(1 for _ in path.rglob("*.json")) if path.exists() else 0
        size = sum(f.stat().st_size for f in path.rglob("*.json")) if path.exists() else 0
        print(f"{label:10s} {str(path):60s} {n:6d} entries  {size/1e6:8.2f} MB")
    print()
    print("Set DENDRO_OFFLINE=1 to replay from cache only; an uncached request then")
    print("degrades that source to 'unavailable' instead of reaching the network.")
    return EXIT_OK


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dendro",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"dendro {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache", help="cache directory (default: $DENDRO_CACHE or ./.dendro-cache)")
    common.add_argument("--offline", action="store_true", help="never touch the network")
    common.add_argument("--sources", help="comma-separated source ids (default: all)")
    common.add_argument("--alpha", type=float, default=1e-2,
                        help="failure-probability budget for the bound (default 0.01)")
    common.add_argument("--archive", help="JSONL file/dir of dated documents for alignment")
    common.add_argument("--json", action="store_true", help="machine-readable output")

    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("date", parents=[common], help="date a URL or file")
    p.add_argument("target")
    p.add_argument("--claimed", help="date the document claims (ISO); default: read from the text")
    p.add_argument("--no-coverage", action="store_true", help="skip coverage probes (cheaper, no backdate detection)")
    p.set_defaults(func=cmd_date)

    p = sub.add_parser("report", parents=[common], help="provenance report for a dataset")
    p.add_argument("dataset", help="HF dataset id or local .jsonl")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int)
    p.add_argument("--text-column", default="text")
    p.add_argument("--min-p", type=float, default=0.90)
    p.add_argument("--probe-coverage", action="store_true")
    p.add_argument("--out", help="write per-document verdicts to JSONL")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("subset", parents=[common], help="build a low-background subset")
    p.add_argument("dataset")
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int)
    p.add_argument("--text-column", default="text")
    p.add_argument("--max-synthetic", type=float, default=0.05)
    p.add_argument("--mode", choices=("expected", "conservative"), default="expected")
    p.add_argument("--out", help="write the selected records to JSONL")
    p.set_defaults(func=cmd_subset)

    p = sub.add_parser("fingerprint", help="show a document's normalised form and sketch")
    p.add_argument("target")
    p.add_argument("--show-normalized", action="store_true")
    p.add_argument("--chars", type=int, default=1200)
    p.set_defaults(func=cmd_fingerprint)

    p = sub.add_parser("sources", help="list witness sources and their trust parameters")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("cache", help="cache statistics")
    p.add_argument("--cache")
    p.set_defaults(func=cmd_cache)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
