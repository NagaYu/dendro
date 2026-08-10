"""Render the benchmark figures from ``results/*.csv``.

Kept separate from ``run.py`` so plots can be restyled without re-running the
experiments, and so every figure is provably a function of a committed CSV -- no
number appears in a figure that is not also in a table a reader can check.

    python -m benchmarks.figures
"""

from __future__ import annotations

import csv
import pathlib
import re
import sys
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

METHODS = ("A", "B", "C")
LABEL = {"A": "(A) statistical / perplexity", "B": "(B) learned classifier", "C": "(C) Dendro — evidence"}
COLOUR = {"A": "#d1495b", "B": "#edae49", "C": "#00798c"}
MARKER = {"A": "o", "B": "s", "C": "D"}

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def read(name: str) -> list[dict[str, Any]]:
    path = RESULTS / name
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


# --------------------------------------------------------------------------- fig 1
def fig_headline() -> None:
    """The extrapolation figure: AUC against generator generation.

    The claim it carries is structural, not empirical luck. (A) and (B) read the
    prose, so their signal is a property of the generator and shrinks as
    generators improve. (C) never reads the prose, so there is no channel through
    which the generator could matter -- the flat line is what "generator-independent"
    means, drawn.
    """
    rows = read("generalization.csv")
    if not rows:
        return

    # The two generator families must not share a continuous x-axis.  The ladder
    # is monotone in distance-from-human; the held-out top-k family is not (one
    # of its rungs is *further* from human than any ladder step), so plotting
    # them as one line would label a jump backwards as "closer to human".
    ladder_rows = [r for r in rows if r.get("seen_family") != "False"]
    family_rows = [r for r in rows if r.get("seen_family") == "False"]

    fig, axes = plt.subplots(
        1, 3, figsize=(13.0, 4.0), gridspec_kw={"width_ratios": [2.0, 1.0, 1.0]}
    )
    ax, ax_fam, ax2 = axes

    gens = sorted({int(_f(r, "generation")) for r in ladder_rows})
    for m in METHODS:
        ys = []
        for g in gens:
            hit = [r for r in ladder_rows if r["method"] == m and int(_f(r, "generation")) == g]
            ys.append(_f(hit[0], "auc") if hit else float("nan"))
        ax.plot(gens, ys, marker=MARKER[m], color=COLOUR[m], lw=2.0, ms=6, label=LABEL[m])

    seen = [
        g for g in gens
        if any(r.get("seen_in_training") == "True" and int(_f(r, "generation")) == g for r in ladder_rows)
    ]
    if seen:
        ax.axvspan(min(gens) - 0.3, max(seen) + 0.5, color="#888", alpha=0.09, lw=0)
        ax.annotate("trained on", xy=(max(seen) + 0.35, 0.06), fontsize=7.5, color="#666",
                    ha="left", va="bottom")
    ax.axhline(0.5, color="#999", lw=0.9, ls=":")
    ax.annotate("chance", xy=(gens[-1] + 0.08, 0.505), fontsize=7.5, color="#999",
                va="bottom", ha="right")
    ax.set_xlabel("generator generation  (→ closer to human text)")
    ax.set_ylabel("AUC   pre-2021 human vs synthetic")
    ax.set_ylim(0.0, 1.06)
    ax.set_xticks(gens)
    ax.set_title("Detectors decay as generators improve.\nEvidence does not.", loc="left")
    ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0.0, 0.10))

    # Middle: a generator family nobody trained on.
    if family_rows:
        names = sorted(
            {r["generator"] for r in family_rows},
            key=lambda n: int(re.sub(r"\D", "", n) or 0),
        )
        x = np.arange(len(names))
        width = 0.26
        for i, m in enumerate(METHODS):
            ys = [
                next((_f(r, "auc") for r in family_rows if r["method"] == m and r["generator"] == n), np.nan)
                for n in names
            ]
            pos = x + (i - 1) * width
            ax_fam.bar(pos, ys, width, color=COLOUR[m], label=LABEL[m])
            if m == "A":
                # (A)'s bars are near zero here, which renders as *absent* rather
                # than as "inverted". Label them so a reader does not read a
                # catastrophic result as missing data.
                for px, v in zip(pos, ys):
                    if v == v:
                        ax_fam.annotate(f"{v:.2f}", xy=(px, v + 0.02), ha="center",
                                        fontsize=7, color="#a33")
        ax_fam.axhline(0.5, color="#999", lw=0.9, ls=":")
        ax_fam.set_xticks(x)
        ax_fam.set_xticklabels([n.replace("unseen-", "") for n in names], fontsize=8)
        ax_fam.set_ylim(0.0, 1.18)
        ax_fam.set_ylabel("AUC")
        ax_fam.set_title("A generator family\nnobody trained on", loc="left")
        ax_fam.annotate("(A) inverts — confidently wrong", xy=(0.5, 1.10), xycoords=("axes fraction", "data"),
                        fontsize=7.5, color="#a33", ha="center")
        ax_fam.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=1)

    # Right: the ladder's x-axis is a measured quantity, not a label.
    ladder = [r for r in read("generator_ladder.csv") if r.get("family") == "temperature"]
    if ladder:
        ladder.sort(key=lambda r: int(_f(r, "generation")))
        g = [int(_f(r, "generation")) for r in ladder]
        ax2.plot(g, [_f(r, "logloss_gap") for r in ladder], marker="o", color="#5c4d7d",
                 lw=1.8, ms=5, label="mean log-loss gap")
        ax2.plot(g, [_f(r, "burstiness_gap") for r in ladder], marker="^", color="#a37bc4",
                 lw=1.8, ms=5, label="burstiness gap")
        ax2.set_xlabel("generator generation")
        ax2.set_ylabel("measured distance from human text")
        ax2.set_xticks(g)
        ax2.set_ylim(bottom=0)
        ax2.set_title("The ladder's x-axis is\nmeasured, not assumed", loc="left")
        ax2.legend(fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig1_headline.png")


# --------------------------------------------------------------------------- fig 2
def fig_robustness() -> None:
    """Adversarial conditions: paraphrase sweep and backdate forgery."""
    rows = read("robustness.csv")
    main = read("main.csv")
    if not rows:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.5, 1.0]})

    strengths = sorted({_f(r, "attack_strength") for r in rows if r.get("attack") == "paraphrase"})
    clean = {r["method"]: _f(r, "auc") for r in main if r["condition"] == "lowbackground"}
    for m in METHODS:
        ys = []
        for s in strengths:
            hit = [r for r in rows if r["method"] == m and r.get("attack") == "paraphrase"
                   and abs(_f(r, "attack_strength") - s) < 1e-9]
            ys.append(_f(hit[0], "auc") if hit else float("nan"))
        ax.plot([0.0, *strengths], [clean.get(m, float("nan")), *ys],
                marker=MARKER[m], color=COLOUR[m], lw=2.0, ms=6, label=LABEL[m])
    ax.axhline(0.5, color="#999", lw=0.9, ls=":")
    ax.set_xlabel("paraphrase strength  (fraction of positions edited)")
    ax.set_ylabel("AUC   paraphrased human vs synthetic")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Paraphrase attack: rewritten human text", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    ax.annotate(
        "below chance = human writing\nsystematically called synthetic",
        xy=(strengths[-1], 0.14), fontsize=7.5, color="#a33", ha="right",
    )

    back = [r for r in rows if r.get("attack") == "backdate"]
    if back:
        xs = np.arange(len(METHODS))
        caught = [_f([r for r in back if r["method"] == m][0], "backdate_flag_rate", 0.0) for m in METHODS]
        false = [_f([r for r in back if r["method"] == m][0], "false_flag_rate", 0.0) for m in METHODS]
        ax2.bar(xs - 0.19, caught, 0.36, color="#00798c", label="forgeries identified")
        ax2.bar(xs + 0.19, false, 0.36, color="#bbb", label="genuine docs wrongly flagged")
        ax2.set_xticks(xs)
        ax2.set_xticklabels(["(A)", "(B)", "(C)"])
        ax2.set_ylabel("rate")
        ax2.set_ylim(0, 1.08)
        ax2.set_title("Backdate forgery: fake old metadata", loc="left")
        ax2.legend(fontsize=7.5, loc="upper left")
        for x, v in zip(xs, caught):
            ax2.annotate(f"{v:.0%}", xy=(x - 0.19, v + 0.02), ha="center", fontsize=8)
        ax2.annotate(
            "(A) and (B) read prose only;\nforged metadata is invisible to them",
            xy=(0.5, 0.62), fontsize=7.5, color="#555", ha="center",
        )
    fig.tight_layout()
    _save(fig, "fig2_robustness.png")


# --------------------------------------------------------------------------- fig 3
def fig_calibration() -> None:
    """Reliability diagrams -- does a stated probability mean what it says?"""
    rows = read("calibration.csv")
    main = read("main.csv")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    ece = {
        r["method"]: _f(r, "ece")
        for r in main
        if r["condition"] == "all_eras"
    }
    for ax, m in zip(axes, METHODS):
        sub = [r for r in rows if r["method"] == m]
        xs, ys, ns = [], [], []
        for r in sub:
            p, o, n = _f(r, "mean_predicted"), _f(r, "observed_frequency"), _f(r, "n", 0.0)
            if n > 0 and not (np.isnan(p) or np.isnan(o)):
                xs.append(p)
                ys.append(o)
                ns.append(n)
        ax.plot([0, 1], [0, 1], color="#999", ls=":", lw=1.0)
        if xs:
            sizes = 18 + 220 * np.asarray(ns) / max(ns)
            ax.scatter(xs, ys, s=sizes, color=COLOUR[m], alpha=0.75, edgecolor="white", lw=0.6, zorder=3)
            ax.plot(xs, ys, color=COLOUR[m], lw=1.4, alpha=0.65)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("stated probability")
        ax.set_title(f"{LABEL[m]}\nECE = {ece.get(m, float('nan')):.3f}", loc="left", fontsize=9)
    axes[0].set_ylabel("observed human-origin frequency")
    fig.suptitle("Reliability: a claimed 0.9 should be right 90% of the time (marker area = bin count)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig3_calibration.png")


# --------------------------------------------------------------------------- fig 4
def fig_selective() -> None:
    """Risk-coverage: the value of saying "I don't know"."""
    rows = read("selective.csv")
    if not rows:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in METHODS:
        sub = sorted((r for r in rows if r["method"] == m), key=lambda r: _f(r, "coverage"))
        cov = [_f(r, "coverage") for r in sub]
        err = [_f(r, "error") for r in sub]
        a = [_f(r, "auc") for r in sub]
        ax.plot(cov, err, marker=MARKER[m], color=COLOUR[m], lw=1.8, ms=4, label=LABEL[m])
        ax2.plot(cov, a, marker=MARKER[m], color=COLOUR[m], lw=1.8, ms=4, label=LABEL[m])
    ax.set_xlabel("coverage  (fraction of documents answered)")
    ax.set_ylabel("error rate on answered documents")
    ax.set_title("Risk–coverage", loc="left")
    ax.legend(fontsize=7.5)
    ax2.set_xlabel("coverage")
    ax2.set_ylabel("AUC on answered documents")
    ax2.axhline(0.5, color="#999", lw=0.9, ls=":")
    ax2.set_title("Selective AUC", loc="left")
    fig.suptitle(
        "A method that abstains and a method that is confidently wrong can share an AUC. They are not the same tool.",
        x=0.01, ha="left", fontsize=9.5, color="#444",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig4_selective.png")


# --------------------------------------------------------------------------- fig 5
def fig_ablation() -> None:
    """Which fingerprint channels survive paraphrase, and which are just constants."""
    rows = read("ablation.csv")
    if not rows:
        return
    full = [r for r in rows if r["variant"] == "full"]
    if not full:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.2, 1.0]})

    channels = [r["channel"] for r in full]
    matched = [_f(r, "matched_containment") for r in full]
    random_ = [_f(r, "random_containment") for r in full]
    x = np.arange(len(channels))
    ax.bar(x - 0.2, matched, 0.4, color="#00798c", label="paraphrase vs its own source")
    ax.bar(x + 0.2, random_, 0.4, color="#c9c9c9", label="paraphrase vs an unrelated document")
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("containment")
    ax.set_title("Channel survival under paraphrase", loc="left")
    ax.legend(fontsize=7.5)
    ax.annotate(
        "the gap is the evidence —\na channel high on both is a constant, not a signal",
        xy=(len(channels) - 0.5, 0.93), fontsize=7.5, color="#555", ha="right", va="top",
    )

    variants = sorted({r["variant"] for r in rows})
    order = ["full", *[v for v in variants if v != "full"]]
    hashes, reflows = [], []
    for v in order:
        sub = [r for r in rows if r["variant"] == v]
        hashes.append(_f(sub[0], "reflow_hash_match_rate", 0.0) if sub else float("nan"))
        reflows.append(float(np.mean([_f(r, "reflow_containment") for r in sub])) if sub else float("nan"))
    y = np.arange(len(order))
    ax2.barh(y - 0.19, hashes, 0.36, color="#00798c", label="rendered copy hashes identically")
    ax2.barh(y + 0.19, reflows, 0.36, color="#c9c9c9", label="mean containment of the rendering")
    ax2.set_yticks(y)
    ax2.set_yticklabels([v.replace("_", " ") for v in order], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("rate / containment")
    ax2.set_title("Normalisation ablation, on rendered HTML", loc="left")
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig5_ablation.png")


# --------------------------------------------------------------------------- fig 6
def fig_cost() -> None:
    """Per-document acquisition cost and cache behaviour -- axis (5)."""
    rows = read("cost.csv")
    if not rows:
        return
    dendro = next((r for r in rows if r["method"] == "C"), None)
    if dendro is None:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [1.0, 1.2]})

    hits, misses = _f(dendro, "hits", 0.0), _f(dendro, "misses", 0.0)
    ax.bar(["cache hits", "network"], [hits, misses], color=["#00798c", "#d1495b"])
    total = hits + misses
    ax.set_ylabel("archive requests")
    ax.set_title(f"Cache hit rate {hits / total:.0%}" if total else "Cache", loc="left")
    for i, v in enumerate([hits, misses]):
        ax.annotate(f"{int(v)}", xy=(i, v), ha="center", va="bottom", fontsize=8)

    labels, vals = [], []
    for r in rows:
        labels.append(LABEL[r["method"]].split(" ")[0])
        vals.append(_f(r, "seconds_per_doc", 0.0) * 1000.0)
    ax2.bar(labels, vals, color=[COLOUR[r["method"]] for r in rows])
    ax2.set_ylabel("ms per document")
    ax2.set_yscale("log")
    ax2.set_title("Scoring cost per document (log scale)", loc="left")
    for i, v in enumerate(vals):
        ax2.annotate(f"{v:.1f}", xy=(i, v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig6_cost.png")


def main() -> int:
    print("rendering figures ...")
    for fn in (fig_headline, fig_robustness, fig_calibration, fig_selective, fig_ablation, fig_cost):
        try:
            fn()
        except Exception as exc:
            print(f"  {fn.__name__} failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
