---
title: Dendro
emoji: 🌳
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Prove a document existed before a date, from independent archives.
tags:
  - provenance
  - data-curation
  - near-duplicate-detection
  - minhash
  - web-archives
  - low-background
  - dataset-contamination
---

# 🌳 Dendro

**Stop asking whether a document *looks* generated. Ask who, other than its author, saw it — and when.**

Paste a URL or a document. Dendro asks independent archives — the Internet Archive, Common
Crawl, arXiv, Crossref, Hacker News — what they observed and when, and combines their answers
into a calibrated existence bound with an interval.

> ### What this is not
>
> **Dendro is not an AI-writing detector, and must not be used as one.** It reports
> *archival evidence that content existed before a date*. No output here supports a
> conclusion that a particular person did or did not write something.
>
> A document with no archival record is **unknown**, never *generated* — most text that has
> ever existed was never archived by anyone. Dendro abstains there, and the abstention falls
> out of the arithmetic rather than being bolted on.
>
> Use it to characterise **corpora**. Do not use it to judge a person.

**Code, benchmark and tests:** [github.com/NagaYu/dendro](https://github.com/NagaYu/dendro)
· **Dataset:** [NagaYu/dendro-lowbackground](https://huggingface.co/datasets/NagaYu/dendro-lowbackground)

---

## Why bother, when detectors exist?

Because detectors read prose, and prose is the part an adversary controls. Measured on 5,034
real arXiv records against a perplexity-family detector **(A)** and a learned classifier **(B)**:

| method | AUC clean | newest generator | unseen generator family | paraphrased | forgeries caught | unwitnessed recent |
|---|---|---|---|---|---|---|
| (A) statistical | 0.738 | 0.621 | 0.234 | 0.141 | 0% | 0.538 |
| (B) learned | 0.987 | 0.976 | 0.961 | 0.953 | 0% | 0.990 |
| **(C) Dendro** | **1.000** | **1.000** | **1.000** | **1.000** | **100%** | **0.500** |

Read the last column with the first. Where no archive holds a record, Dendro falls to chance
and abstains on 100% of cases — and (B) is the better tool there. Publishing only the
flattering column would be the easiest way to make this project dishonest.

![headline](figures/fig1_headline.png)

(A) and (B) read the prose, so their signal is a property of the generator and shrinks as
generators improve. (C) never reads the prose, so there is no channel through which the
generator *could* matter.

![robustness](figures/fig2_robustness.png)

Left: a perplexity detector doesn't just fail on paraphrased human writing — it goes *below
chance*, systematically calling rewritten human text synthetic. Right: forged `<meta>` dates
are invisible to anything that reads prose, because the prose is unchanged.

---

## How the evidence is combined

**Operators are the unit of independence, not witnesses.** Twenty Internet Archive captures
are one archive. One capture from each of three organisations is three. Within an operator,
failure is correlated — whoever can write to the archive can write to all of it — so a group's
failure probability is floored at its forgeability no matter how many records it holds. Across
operators it multiplies.

**The bound is a weighted order statistic, not a minimum.** A single injected early record
cannot drag the date backwards; moving it requires collusion across operators.

**Paraphrase loses because the fingerprint has more than one channel.** Exact word n-grams die
under rewriting; the rare-content-word and numeral/entity channels do not, because a rewrite
that changes "1,247 deaths in Bergamo" has stopped being a rewrite. A document that aligns to
an archived ancestor inherits that ancestor's date.

**Backdating loses because silence is quantified before it is used.** `log LR = −k·log(1−c)`
goes to exactly zero as coverage `c` goes to zero — so a page nobody ever crawled is never
suspected merely for being obscure.

---

## Notes on this Space

Bundled with a small cache of **real** recorded responses from the Internet Archive CDX,
Common Crawl, arXiv and Hacker News, so the examples answer instantly without touching those
APIs. Anything you type that isn't cached is fetched live, rate-limited to 0.5 requests/second
per host — the archives are a commons.

See [DATA.md](DATA.md) for provenance and attribution of everything shipped here. arXiv
metadata is used under CC0 via their OAI-PMH interface; thank you to arXiv for use of its open
access interoperability.

Apache-2.0.
