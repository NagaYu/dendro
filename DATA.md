# Data in this repository: what it is, where it came from, and why it is here

Dendro is a project about provenance, so its own data had better have some.

## `data/corpus/arxiv_pre2021.jsonl`, `data/corpus/arxiv_recent.jsonl`

**5,034 arXiv records** (identifier, title, abstract, categories, DOI, and the v1
submission date), harvested with `scripts/fetch_corpus.py` through arXiv's OAI-PMH
interface — the endpoint arXiv provides for exactly this purpose.

arXiv offers its **metadata**, including abstracts, under
[CC0 1.0](https://info.arxiv.org/help/api/tou.html) via the public API and OAI-PMH.
That is what is committed here. Full texts are *not* included; they carry
per-paper licences that vary and are not ours to redistribute.

Thank you to arXiv for use of its open access interoperability.

## `data/fixtures/cache/` — 556 entries, 3.3 MB

Recorded responses from four public APIs, committed so the tests and the benchmark
replay offline with **zero network calls**:

| operator | endpoint | what is stored |
|---|---|---|
| Internet Archive | CDX server | capture timestamps, URL keys, content digests |
| Common Crawl | index server | capture timestamps, URL keys, digests |
| arXiv | Atom API | submission dates for individual identifiers |
| Y Combinator / Algolia | HN search API | submission timestamps for URLs |

These are **factual records about when things were observed** — timestamps, URLs,
and hashes — not archived page content. They are small, they are the evidence the
paper's claims rest on, and committing them is what makes "reproduces offline"
true rather than aspirational.

The bulk OAI-PMH payloads (~43 MB) are pruned after extraction
(`python -m scripts.fetch_corpus --prune-bulk`) because `data/corpus/*.jsonl`
already contains everything downstream reads.

## What is deliberately *not* committed

`data/local/web_pre2021.jsonl` — verbatim copies of eight pre-2021 web pages
(python.org PEPs, gnu.org/philosophy, RFC 2119, W3C, curl.se, kernel.org,
docs.python.org), used to exercise the Wayback snapshot path against real archived
HTML.

Those pages belong to the FSF, the Linux Foundation, the PSF, the curl project and
others, under a mix of licences. Dendro only ever needs their *fingerprints*, and
the benchmark runs without them — the archive layer is 450 arXiv documents instead
of 458, and no reported number changes materially. Re-fetch them locally with:

```bash
python -m scripts.fetch_corpus --skip-arxiv
```

## Rate limiting and terms of use

Every request in this project goes through `dendro.cache.HttpClient`, which
enforces a persistent per-host token bucket (`dendro.cache.DEFAULT_RATES`,
0.5 req/s for the archives and 0.2 req/s for arXiv, above arXiv's requested 3-second
gap), identifies itself with a contactable User-Agent, honours `Retry-After`, and
caches everything so a re-run costs nothing. The bulk harvest uses OAI-PMH rather
than hammering the search API, because that is the interface arXiv asks harvesters
to use.

If you are running this at scale against these archives, please keep those defaults.
They are a commons.

## Regenerating everything

```bash
python -m scripts.fetch_corpus --prune-bulk   # ~5 min, mostly rate-limit sleep
python -m benchmarks.run                      # ~15 min cold, ~4 min warm
python -m benchmarks.figures && python -m benchmarks.tables
```
