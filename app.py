"""Dendro — Hugging Face Space.

Paste a URL or a document; get back the archival evidence that it existed before
some date, a timeline of who witnessed it and when, and a calibrated
human-origin probability with an interval.

The interface is arranged to put the *evidence* first and the probability last,
which is the opposite of how a detector UI usually works and is deliberate.  The
witness table is checkable — every row carries a link you can open — while the
probability additionally depends on an assumed prevalence curve.  A user who
reads only the number has been handed the least reliable part of the output.

The banner is not boilerplate.  This tool answers "did independent archives see
this content, and when?"  It does not answer "did a person write this?", and the
UI says so on every result, because the failure mode that matters here is a
person being accused on the strength of a number.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import os
import pathlib
import sys
import traceback

import gradio as gr

REPO = pathlib.Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dendro import __version__
from dendro.cache import Cache, HttpClient, RateLimiter
from dendro.pipeline import Dendro
from dendro.types import to_utc

DISCLAIMER = """
> ### What this is, and what it is not
>
> Dendro reports **archival evidence that content existed before a date**. It works by
> asking independent archives — the Internet Archive, Common Crawl, arXiv, Crossref,
> public posting archives — what they saw and when. It never inspects the writing style.
>
> **It is not an AI-writing detector and must not be used as one.** No output here
> supports a conclusion that a particular person did or did not write something. A
> document with no archival record is *unknown*, not *generated* — most text that has
> ever existed was never archived by anyone.
>
> Use it to characterise **corpora**: what share of a dataset provably predates
> large-scale text generation. Do not use it to make a judgement about an individual.
"""

METHOD = """
**How the bound is established.** Each archive that has seen the content contributes a
witness: *operator O observed this at time T*. Witnesses are grouped by **operator**, not
by count — twenty Internet Archive captures are one operator, and one capture from each of
three organisations is three. For the reported date to be wrong, every supporting operator
must have failed independently, so the confidence multiplies across organisations and
barely moves when you add more records from one of them.

**Why paraphrasing doesn't help.** The fingerprint is format-blind and multi-channel. Exact
word n-grams die under rewriting; the rare-content-word and numeral/entity channels do not,
because a rewrite that changes the numbers has stopped being a rewrite. A rewritten document
that aligns to an archived ancestor inherits that ancestor's date.

**Why backdating doesn't help.** A forged `<meta name="date">` is invisible to a text
detector — the prose is unchanged. Dendro checks the claim against archives that were
*demonstrably crawling that neighbourhood* at the claimed time. "No record" only becomes
evidence when coverage is measured, which is why a page nobody ever crawled is never accused.
"""


def _build() -> Dendro:
    client = HttpClient(
        cache=Cache(root=os.environ.get("DENDRO_CACHE") or (REPO / ".dendro-cache")),
        rate_limiter=RateLimiter(),
        offline=os.environ.get("DENDRO_OFFLINE", "").lower() in {"1", "true", "yes"},
        timeout=25.0,
    )
    return Dendro(client=client)


DENDRO = _build()


# --------------------------------------------------------------------------- plot
def _timeline(verdict, claimed):
    """Witnesses on a time axis, with the bound and any contradicted claim.

    The visual carries one idea: how far apart the claim and the first
    independent observation are, and how many distinct operators stand behind
    the bound. Colour encodes operator, so a bound resting on a single
    organisation *looks* like one thing rather than several.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=140)
    witnesses = [w for w in verdict.bound.all_witnesses]
    if not witnesses and claimed is None:
        ax.text(0.5, 0.5, "no witnesses found", ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#888")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        return fig

    operators = sorted({w.operator for w in witnesses})
    palette = plt.get_cmap("tab10")
    colours = {op: palette(i % 10) for i, op in enumerate(operators)}

    for i, op in enumerate(operators):
        pts = [w.observed_at for w in witnesses if w.operator == op]
        independent = any(w.is_independent_evidence for w in witnesses if w.operator == op)
        ax.scatter(
            pts, [i] * len(pts),
            s=90 if independent else 60,
            color=colours[op],
            marker="o" if independent else "x",
            zorder=3,
            label=op,
        )
    ax.set_yticks(range(len(operators)))
    ax.set_yticklabels(
        [f"{op}{'' if any(w.is_independent_evidence for w in witnesses if w.operator == op) else '  (claim only)'}"
         for op in operators],
        fontsize=8,
    )

    na = verdict.not_after
    if na is not None:
        ax.axvline(na, color="#1a7f37", lw=2.0, ls="-", zorder=2)
        ax.annotate("existence bound", xy=(na, len(operators) - 0.4), fontsize=8,
                    color="#1a7f37", rotation=90, va="top", ha="right")
    if claimed is not None:
        ax.axvline(claimed, color="#cf222e", lw=1.6, ls="--", zorder=2)
        ax.annotate("claimed date", xy=(claimed, -0.6), fontsize=8, color="#cf222e",
                    rotation=90, va="bottom", ha="right")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_ylim(-0.8, len(operators) - 0.2)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_title("Who observed this content, and when", fontsize=10, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- handler
def analyse(url: str, text: str, claimed: str, probe: bool):
    url = (url or "").strip()
    text = (text or "").strip()
    claimed_dt = None
    if (claimed or "").strip():
        try:
            claimed_dt = to_utc(claimed.strip())
        except ValueError:
            return ("Could not parse the claimed date. Use `YYYY-MM-DD`.", None, [], "", "")

    if not url and not text:
        return ("Enter a URL or paste some text.", None, [], "", "")

    try:
        verdict = DENDRO.date(
            text=text or None,
            url=url or None,
            doc_id=url or "pasted-document",
            claimed_date=claimed_dt,
            probe_coverage=probe,
        )
    except Exception:
        return (f"```\n{traceback.format_exc()[-1500:]}\n```", None, [], "", "")

    na = verdict.not_after
    rows = [
        [
            w.source_id,
            w.operator,
            w.kind.value,
            w.observed_at.date().isoformat(),
            "yes" if w.is_independent_evidence else "no (claim)",
            f"{w.forgeability:.1e}",
            (w.url or "")[:90],
        ]
        for w in verdict.bound.all_witnesses
    ]

    if na is not None:
        headline = (
            f"## Existence proven on or before **{na.date().isoformat()}**\n\n"
            f"{verdict.bound.independent_operators} independent operator(s) · "
            f"evidence log-odds {verdict.bound.forgery_logodds:.1f}"
        )
    else:
        headline = (
            "## No archival evidence found\n\n"
            "No independent archive holds a record of this content. That is **not** evidence "
            "of synthetic origin — most text was never archived by anyone."
        )

    verdict_md = [headline, ""]
    if verdict.abstained:
        verdict_md.append(
            "**ABSTAIN** — the evidence does not narrow this enough to act on.\n"
        )
    verdict_md.append(
        f"**P(human-origin) = {verdict.human_origin_p:.3f}**  "
        f"(90% interval {verdict.ci_low:.2f} – {verdict.ci_high:.2f})\n"
    )
    for f in verdict.flags:
        icon = {"high": "🔴", "medium": "🟠", "low": "⚪"}.get(f.severity, "⚪")
        verdict_md.append(f"{icon} **{f.kind}** ({f.severity}, log-LR {f.log_lr:.1f}) — {f.detail}\n")
    if verdict.ancestor is not None and verdict.ancestor.is_ancestral:
        a = verdict.ancestor
        verdict_md.append(
            f"🌳 **Ancestor found**: `{a.relation.value}` of `{a.ref_doc_id}` "
            f"(confidence {a.confidence:.2f}) — this document inherits that date.\n"
        )

    fig = _timeline(verdict, claimed_dt)
    cache = json.dumps(DENDRO.stats.as_row())
    return ("\n".join(verdict_md), fig, rows, verdict.explanation, f"`{cache}`")


# --------------------------------------------------------------------------- ui
with gr.Blocks(title=f"Dendro {__version__}", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🌳 Dendro\n"
        "### Prove a document existed before a date — without reading it.\n"
        "Synthetic-text detectors ask *does this look generated?*, and that question gets "
        "harder every time a model ships. Dendro asks *who, other than the author, saw this "
        "and when?* — a question that does not get harder at all."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column(scale=3):
            url_in = gr.Textbox(
                label="URL",
                placeholder="https://www.python.org/dev/peps/pep-0020/",
                lines=1,
            )
            text_in = gr.Textbox(
                label="…or paste the document text",
                placeholder="Paste the text. Alignment against archived documents works on text alone.",
                lines=8,
            )
        with gr.Column(scale=1):
            claimed_in = gr.Textbox(label="Claimed date (optional)", placeholder="2019-03-11", lines=1)
            probe_in = gr.Checkbox(
                value=True,
                label="Probe archive coverage",
                info="Extra queries that measure how deeply archives crawled this host. "
                     "Required for backdate detection; costs ~1 request per source.",
            )
            run = gr.Button("Collect evidence", variant="primary")

    gr.Examples(
        examples=[
            ["https://www.python.org/dev/peps/pep-0020/", "", "", True],
            ["https://arxiv.org/abs/1706.03762", "", "", True],
            ["https://www.gnu.org/philosophy/free-sw.html", "", "1996-01-01", True],
        ],
        inputs=[url_in, text_in, claimed_in, probe_in],
        label="Try one",
    )

    out_md = gr.Markdown()
    out_plot = gr.Plot(label="Evidence timeline")
    out_table = gr.Dataframe(
        headers=["source", "operator", "kind", "observed", "independent?", "forgeability", "link"],
        label="Witnesses",
        wrap=True,
    )
    with gr.Accordion("Full explanation", open=False):
        out_expl = gr.Textbox(label="", lines=12)
    out_cache = gr.Markdown()

    with gr.Accordion("Method", open=False):
        gr.Markdown(METHOD)

    run.click(
        analyse,
        inputs=[url_in, text_in, claimed_in, probe_in],
        outputs=[out_md, out_plot, out_table, out_expl, out_cache],
    )

if __name__ == "__main__":
    demo.launch()
