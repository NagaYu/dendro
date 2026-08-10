"""Render the README's tables from ``results/*.csv``.

Every number quoted in the README is produced here, from a committed CSV, so a
claim in the prose and a claim in the data cannot drift apart.  Re-run after
``benchmarks.run``::

    python -m benchmarks.tables          # writes figures/benchmark_table.md
    python -m benchmarks.tables --print  # to stdout
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Any, Optional, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "figures" / "benchmark_table.md"

NAME = {"A": "(A) statistical", "B": "(B) learned", "C": "**(C) Dendro**"}


def read(name: str) -> list[dict[str, Any]]:
    path = RESULTS / name
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict, key: str, default=float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def fmt(v: float, nd: int = 3) -> str:
    return "—" if v != v else f"{v:.{nd}f}"


def pct(v: float) -> str:
    return "—" if v != v else f"{v:.0%}"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------- sections
def headline_table() -> str:
    main = read("main.csv")
    gen = read("generalization.csv")
    rob = read("robustness.csv")
    if not main:
        return "_(run `python -m benchmarks.run` first)_"

    rows = []
    for m in ("A", "B", "C"):
        clean = next((f(r, "auc") for r in main if r["condition"] == "lowbackground" and r["method"] == m), float("nan"))
        unwit = next(
            (f(r, "auc") for r in main if r["condition"] == "recent_unwitnessed" and r["method"] == m),
            float("nan"),
        )
        fam = [f(r, "auc") for r in gen if r["method"] == m and r.get("seen_family") == "False"]
        last_gen = [f(r, "auc") for r in gen if r["method"] == m and r.get("generation") == "6"]
        para = next(
            (f(r, "auc") for r in rob
             if r["method"] == m and r.get("attack") == "paraphrase"
             and abs(f(r, "attack_strength") - 0.55) < 1e-6),
            float("nan"),
        )
        back = next((r for r in rob if r["method"] == m and r.get("attack") == "backdate"), {})
        rows.append([
            NAME[m],
            fmt(clean),
            fmt(last_gen[0] if last_gen else float("nan")),
            fmt(sum(fam) / len(fam) if fam else float("nan")),
            fmt(para),
            pct(f(back, "backdate_flag_rate", 0.0)),
            pct(f(back, "false_flag_rate", 0.0)),
            fmt(unwit),
        ])
    return table(
        ["method", "AUC clean", "AUC newest gen", "AUC unseen family",
         "AUC paraphrased", "forgeries caught", "false accusations",
         "AUC unwitnessed recent"],
        rows,
    )


def generalization_table() -> str:
    rows_csv = read("generalization.csv")
    ladder = {r["generator"]: r for r in read("generator_ladder.csv")}
    if not rows_csv:
        return ""
    gens = sorted({r["generation"] for r in rows_csv}, key=int)
    out = []
    for g in gens:
        sub = {r["method"]: r for r in rows_csv if r["generation"] == g}
        any_row = next(iter(sub.values()))
        info = ladder.get(any_row.get("generator", ""), {})
        out.append([
            any_row.get("generator", g),
            any_row.get("family", "—"),
            fmt(f(info, "logloss_gap"), 3),
            fmt(f(sub.get("A", {}), "auc")),
            fmt(f(sub.get("B", {}), "auc")),
            fmt(f(sub.get("C", {}), "auc")),
        ])
    return table(
        ["generator", "family", "measured gap to human", "(A) AUC", "(B) AUC", "**(C) AUC**"], out
    )


def robustness_table() -> str:
    rows_csv = [r for r in read("robustness.csv") if r.get("attack") == "paraphrase"]
    if not rows_csv:
        return ""
    strengths = sorted({f(r, "attack_strength") for r in rows_csv})
    out = []
    for s in strengths:
        sub = {r["method"]: r for r in rows_csv if abs(f(r, "attack_strength") - s) < 1e-9}
        out.append([
            f"{s:.2f}",
            fmt(f(sub.get("A", {}), "auc")),
            fmt(f(sub.get("B", {}), "auc")),
            fmt(f(sub.get("C", {}), "auc")),
            pct(f(sub.get("C", {}), "abstain_rate_pos")),
        ])
    return table(
        ["paraphrase strength", "(A) AUC", "(B) AUC", "**(C) AUC**",
         "(C) abstains on the rewrite"], out
    )


def scramble_table() -> str:
    rows_csv = read("scramble.csv")
    if not rows_csv:
        return ""
    levels = sorted({f(r, "content_scramble") for r in rows_csv})
    out = []
    for lv in levels:
        sub = {r["method"]: r for r in rows_csv if abs(f(r, "content_scramble") - lv) < 1e-9}
        c = sub.get("C", {})
        out.append([
            f"{lv:.1f}",
            pct(f(c, "ancestor_recall")),
            fmt(f(sub.get("A", {}), "auc")),
            fmt(f(sub.get("B", {}), "auc")),
            fmt(f(c, "auc")),
        ])
    return table(
        ["content scramble", "(C) ancestor recall", "(A) AUC", "(B) AUC", "**(C) AUC**"], out
    )


def calibration_table() -> str:
    main = read("main.csv")
    if not main:
        return ""
    out = []
    for m in ("A", "B", "C"):
        row = next((r for r in main if r["condition"] == "all_eras" and r["method"] == m), {})
        low = next((r for r in main if r["condition"] == "lowbackground" and r["method"] == m), {})
        out.append([NAME[m], fmt(f(low, "ece")), fmt(f(low, "brier")),
                    fmt(f(row, "ece")), fmt(f(row, "brier"))])
    return table(
        ["method", "ECE (low-background)", "Brier (low-background)", "ECE (all eras)", "Brier (all eras)"],
        out,
    )


def cost_table() -> str:
    rows_csv = read("cost.csv")
    if not rows_csv:
        return ""
    out = []
    for r in rows_csv:
        out.append([
            NAME[r["method"]],
            r.get("documents_scored", "—"),
            r.get("network_calls", "0"),
            r.get("hit_rate", "—") or "—",
            fmt(f(r, "seconds_per_doc") * 1000, 1) + " ms",
        ])
    return table(["method", "documents scored", "archive requests", "cache hit rate", "per document"], out)


def ablation_table() -> str:
    rows_csv = [r for r in read("ablation.csv") if r["variant"] == "full"]
    if not rows_csv:
        return ""
    out = [
        [r["channel"], fmt(f(r, "matched_containment")), fmt(f(r, "random_containment")),
         fmt(f(r, "separation")), fmt(f(r, "reflow_containment"))]
        for r in sorted(rows_csv, key=lambda r: -f(r, "separation"))
    ]
    return table(
        ["channel", "paraphrase vs source", "vs unrelated doc", "separation", "rendered HTML vs source"],
        out,
    )


def normalisation_table() -> str:
    rows_csv = read("ablation.csv")
    if not rows_csv or "reflow_hash_match_rate" not in rows_csv[0]:
        return ""
    variants = sorted({r["variant"] for r in rows_csv})
    order = ["full", *[v for v in variants if v != "full"]]
    out = []
    for v in order:
        sub = [r for r in rows_csv if r["variant"] == v]
        if not sub:
            continue
        mean_reflow = sum(f(r, "reflow_containment") for r in sub) / len(sub)
        out.append([v.replace("_", " "), pct(f(sub[0], "reflow_hash_match_rate")), fmt(mean_reflow)])
    return table(
        ["normalisation", "rendered copy hashes identically", "mean containment of the rendering"], out
    )


def prevalence_table() -> str:
    rows_csv = read("prevalence_sweep.csv")
    if not rows_csv:
        return ""
    out = [
        [r["curve"], fmt(f(r, "auc_lowbackground")), fmt(f(r, "mean_p_pre2021")),
         fmt(f(r, "mean_p_synthetic")), fmt(f(r, "mean_p_recent_human"))]
        for r in rows_csv
    ]
    return table(
        ["prevalence curve", "AUC (low-background)", "mean P(human) pre-2021",
         "mean P(human) synthetic", "mean P(human) recent"], out
    )


# --------------------------------------------------------------------------- main
def render() -> str:
    summary_path = RESULTS / "summary.json"
    corpus = {}
    if summary_path.is_file():
        corpus = json.loads(summary_path.read_text(encoding="utf-8")).get("corpus", {})

    parts = [
        "<!-- generated by `python -m benchmarks.tables` — do not edit by hand -->",
        "## Benchmark results",
        "",
        f"Corpus: `{json.dumps(corpus)}`",
        "",
        "### Headline",
        "",
        headline_table(),
        "",
        "### Axis 3 — generalisation across generator generations",
        "",
        generalization_table(),
        "",
        "### Axis 2 — paraphrase attack",
        "",
        robustness_table(),
        "",
        "### Axis 2c — the breaking point: paraphrase that also replaces rare content words",
        "",
        scramble_table(),
        "",
        "### Axis 4 — calibration",
        "",
        calibration_table(),
        "",
        "_Read Dendro's ECE with care. On a task where the evidence separates the classes",
        "completely, an isotonic map collapses to a step function and ECE goes to ~0 — that",
        "reflects the separability of the task, not a virtue of the method. The informative",
        "calibration numbers here are the baselines', and the `recent_unwitnessed` row of",
        "`results/main.csv`, where Dendro has no evidence and abstains rather than guessing._",
        "",
        "### Axis 5 — cost",
        "",
        cost_table(),
        "",
        "### Fingerprint channels: what survives a rewrite, and what is just a constant",
        "",
        ablation_table(),
        "",
        "### Normalisation ablation, measured on rendered HTML",
        "",
        normalisation_table(),
        "",
        "### Sensitivity to the assumed prevalence curve",
        "",
        prevalence_table(),
        "",
        "_The ranking is a property of the evidence and does not move; the absolute",
        "probabilities do. That is the honest scope of the prevalence assumption._",
        "",
    ]
    return "\n".join(parts)


README = REPO / "README.md"


def inject_readme() -> bool:
    """Rewrite the README's results blocks from the CSVs.

    Idempotent, marker-delimited substitution.  The point is that no number in
    the prose can drift away from ``results/`` -- transcribing a benchmark table
    by hand is the single most common way a claim in a README stops being true
    after a re-run.
    """
    if not README.is_file():
        return False
    text = README.read_text(encoding="utf-8")
    blocks = {
        "RESULTS_HEADLINE": headline_table(),
        "RESULTS_BODY": "\n\n".join(
            [
                "### Generalisation across generator generations",
                generalization_table(),
                "### Robustness — paraphrase attack",
                robustness_table(),
                "### The breaking point — paraphrase that also replaces rare content words",
                scramble_table(),
                "### Calibration",
                calibration_table(),
                "_Read Dendro's ECE with care: where the evidence separates the classes completely, "
                "an isotonic map collapses to a step and ECE goes to ~0. That reflects the task, not "
                "a virtue of the method. The informative row is `recent_unwitnessed`, where Dendro "
                "has no evidence and abstains rather than guessing._",
                "### Cost per document",
                cost_table(),
                "### Fingerprint channels under paraphrase",
                ablation_table(),
                "### Normalisation ablation, on rendered HTML",
                normalisation_table(),
                "### Sensitivity to the assumed prevalence curve",
                prevalence_table(),
                "_The ranking is a property of the evidence and does not move; the absolute "
                "probabilities do. That is the honest scope of the prevalence assumption._",
            ]
        ),
    }
    for marker, body in blocks.items():
        begin, end = f"<!--{marker}-->", f"<!--/{marker}-->"
        payload = f"{begin}\n\n{body}\n\n{end}"
        if begin in text and end in text:
            head, rest = text.split(begin, 1)
            _, tail = rest.split(end, 1)
            text = head + payload + tail
        elif begin in text:
            text = text.replace(begin, payload, 1)
    README.write_text(text, encoding="utf-8")
    return True


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", dest="to_stdout")
    ap.add_argument("--no-readme", action="store_true", help="skip the README injection")
    args = ap.parse_args(argv)
    text = render()
    if args.to_stdout:
        print(text)
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    if not args.no_readme and inject_readme():
        print("updated README.md results blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
