"""Fetch the *real* documents the evaluation corpus is built from.

Everything this script downloads is cached under ``data/fixtures/cache`` and the
extracted records land in ``data/corpus/*.jsonl``.  Both are committed, so a
fresh clone reproduces every number in ``results/`` with ``DENDRO_OFFLINE=1`` and
no network at all.  That is the point: a provenance paper whose evidence
evaporates when an API changes is not evidence.

Why arXiv is the backbone of the corpus:

* the text is real, human-written, and long enough to fingerprint;
* the submission date is a *genuine independent registration witness*, not a
  self-reported field -- which is exactly the kind of evidence Dendro consumes;
* it spans both eras cleanly, so class (i) pre-2021 and class (ii) recent are
  drawn from the same genre and the comparison is not confounded by domain.

A caveat that the README repeats and that the whole project exists because of:
**class (ii) is "recent, human-attributed", not "verified human".**  By 2025 a
large share of abstracts have been through a model at some point.  There is no
clean recent human corpus to be had, anywhere -- which is the argument for
dating by archive rather than by inspection.

Usage::

    python -m scripts.fetch_corpus                 # everything, cached
    python -m scripts.fetch_corpus --limit 40      # smaller
    python -m scripts.fetch_corpus --skip-web      # arXiv only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from typing import Any, Iterable, Optional
from xml.etree import ElementTree as ET

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dendro.cache import Cache, HttpClient, RateLimiter  # noqa: E402
from dendro.types import to_utc  # noqa: E402

ARXIV_OAI = "http://export.arxiv.org/oai2"
_OAI = "{http://www.openarchives.org/OAI/2.0/}"
_ARX = "{http://arxiv.org/OAI/arXiv/}"

CORPUS_DIR = REPO / "data" / "corpus"
CACHE_DIR = REPO / "data" / "fixtures" / "cache"
#: Verbatim third-party page content lands here and is *not* committed.  The
#: arXiv records in ``data/corpus`` are metadata offered under CC0 and are safe to
#: redistribute; full copies of gnu.org, kernel.org, curl.se and python.org pages
#: are not ours to republish, and Dendro only ever needs their fingerprints.  The
#: benchmark runs without them.
LOCAL_DIR = REPO / "data" / "local"

#: (set, from, until).  OAI-PMH is arXiv's designated bulk-harvest interface --
#: one request returns 100-200 full records, where the Atom search API needs one
#: request per handful and rate-limits hard.  Using the interface the operator
#: actually wants harvesters to use is part of "respect the terms of service",
#: not merely a performance choice.
#:
#: Note that the window filters on *datestamp* (last modification), so a 2019
#: window also surfaces older papers that were revised then.  Era is therefore
#: assigned from each record's own ``<created>`` field, which is the true v1
#: submission date and the thing arXiv actually witnesses.
#: Scale matters more than it looks.  An early run trained the synthetic
#: generator on 41 abstracts, and the learned baseline then scored a perfect 1.0
#: on *every* generation -- not because it recognised generated text, but because
#: it recognised the 41-document vocabulary.  A generator whose lexicon is a
#: giveaway makes the generalisation axis meaningless, so the corpus is sized to
#: give the generator a few thousand documents to draw from.
ARXIV_WINDOWS: list[tuple[str, str, str]] = [
    # ---- pre-2021 ----
    ("cs", "2019-03-01", "2019-03-02"),
    ("cs", "2019-07-15", "2019-07-16"),
    ("cs", "2020-01-14", "2020-01-15"),
    ("cs", "2020-10-06", "2020-10-07"),
    ("math", "2019-06-11", "2019-06-12"),
    ("math", "2019-09-17", "2019-09-18"),
    ("math", "2020-04-21", "2020-04-22"),
    ("stat", "2019-11-05", "2019-11-06"),
    ("stat", "2020-06-09", "2020-06-10"),
    ("q-bio", "2020-02-05", "2020-02-08"),
    ("q-bio", "2019-08-06", "2019-08-12"),
    ("econ", "2020-05-01", "2020-05-12"),
    ("econ", "2019-04-02", "2019-04-16"),
    ("physics:astro-ph", "2020-09-01", "2020-09-02"),
    ("physics:astro-ph", "2019-05-14", "2019-05-15"),
    ("physics:cond-mat", "2019-10-08", "2019-10-09"),
    ("physics:cond-mat", "2020-03-10", "2020-03-11"),
    ("eess", "2020-07-07", "2020-07-09"),
    ("eess", "2019-12-03", "2019-12-05"),
    # ---- recent ----
    ("cs", "2025-03-04", "2025-03-05"),
    ("cs", "2025-09-09", "2025-09-10"),
    ("cs", "2026-01-13", "2026-01-14"),
    ("cs", "2026-04-07", "2026-04-08"),
    ("math", "2025-06-10", "2025-06-11"),
    ("math", "2026-02-17", "2026-02-18"),
    ("stat", "2025-11-04", "2025-11-05"),
    ("stat", "2026-03-17", "2026-03-18"),
    ("q-bio", "2026-02-10", "2026-02-13"),
    ("q-bio", "2025-05-06", "2025-05-12"),
    ("econ", "2026-03-02", "2026-03-10"),
    ("econ", "2025-07-01", "2025-07-14"),
    ("physics:astro-ph", "2026-05-06", "2026-05-07"),
    ("physics:astro-ph", "2025-08-12", "2025-08-13"),
    ("physics:cond-mat", "2025-10-14", "2025-10-15"),
    ("physics:cond-mat", "2026-06-02", "2026-06-03"),
    ("eess", "2025-04-08", "2025-04-10"),
    ("eess", "2026-05-19", "2026-05-21"),
]

#: Pre-2021 web pages with deep Wayback coverage.  Used for the web/HTML flavour
#: of class (i) and, more importantly, to exercise the *snapshot* witness path
#: end to end against the real Internet Archive.
WEB_SEEDS: list[str] = [
    "https://www.python.org/dev/peps/pep-0020/",
    "https://www.python.org/dev/peps/pep-0008/",
    "https://www.gnu.org/philosophy/free-sw.html",
    "https://www.rfc-editor.org/rfc/rfc2119.txt",
    "https://www.w3.org/Provider/Style/Introduction.html",
    "https://curl.se/docs/manpage.html",
    "https://www.kernel.org/doc/html/latest/process/coding-style.html",
    "https://docs.python.org/3/tutorial/introduction.html",
]


def _client(offline: bool = False) -> HttpClient:
    return HttpClient(
        cache=Cache(root=CACHE_DIR),
        rate_limiter=RateLimiter(),
        offline=offline,
        timeout=60.0,
        max_retries=4,
    )


# --------------------------------------------------------------------------- arXiv
def fetch_arxiv_window(
    client: HttpClient, oai_set: str, start: str, end: str, limit: int
) -> list[dict[str, Any]]:
    """Harvest one OAI-PMH window into records carrying their real v1 dates.

    Only ``<created>`` is kept as the date.  ``<updated>`` would silently loosen
    every bound on a revised paper, and a bound that is quietly wrong in the
    permissive direction is the single most damaging error this corpus could
    contain -- it would make Dendro look good for the wrong reason.
    """
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": oai_set, "from": start, "until": end}
    got = client.try_fetch(ARXIV_OAI, params, kind="text")
    if not got or not isinstance(got.get("body"), str):
        return []
    try:
        root = ET.fromstring(got["body"])
    except ET.ParseError:
        return []

    out: list[dict[str, Any]] = []
    for meta in root.iter(f"{_ARX}arXiv"):
        arxiv_id = _clean(meta.findtext(f"{_ARX}id") or "")
        created = _clean(meta.findtext(f"{_ARX}created") or "")
        title = _clean(meta.findtext(f"{_ARX}title") or "")
        abstract = _clean(meta.findtext(f"{_ARX}abstract") or "")
        categories = _clean(meta.findtext(f"{_ARX}categories") or "")
        doi = _clean(meta.findtext(f"{_ARX}doi") or "")
        if not (arxiv_id and created and title and abstract):
            continue
        if len(abstract.split()) < 90:
            # Too short to fingerprint meaningfully.  A 40-word abstract makes
            # every method look bad for reasons unrelated to the question asked.
            continue
        try:
            published = to_utc(created)
        except ValueError:
            continue
        out.append(
            {
                "doc_id": f"arxiv:{arxiv_id}",
                "arxiv_id": arxiv_id,
                "title": title,
                "text": f"{title}\n\n{abstract}",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "doi": doi or None,
                "published": published.isoformat(),
                "category": categories.split(" ")[0] if categories else oai_set,
                "genre": "abstract",
                # Era comes from the record, never from the harvest window.
                "era": "pre2021" if published.year <= 2020 else ("recent" if published.year >= 2024 else "mid"),
            }
        )
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------- web
def fetch_web_seed(client: HttpClient, url: str) -> Optional[dict[str, Any]]:
    """Pull the oldest usable Wayback capture of a seed URL.

    The *archived* rendering is fetched rather than the live page on purpose: it
    is the artefact whose existence the Internet Archive actually witnesses, and
    it carries the exact chrome (toolbars, banners, re-wrapping) that the
    normaliser has to survive.  Using it here means the reflow-invariance claim
    is exercised against real archive HTML rather than a synthetic mock-up.
    """
    from dendro.sources.wayback import _cdx

    rows = _cdx(
        client,
        {
            "url": url,
            "output": "json",
            "fl": "timestamp,original,digest,statuscode",
            "collapse": "digest",
            "from": "2015",
            "to": "2020",
            "limit": "12",
        },
    )
    for row in rows:
        ts, original, digest, status = (list(row) + ["", "", "", ""])[:4]
        if status not in ("200", "-", ""):
            continue
        snap = f"https://web.archive.org/web/{ts}id_/{original}"
        got = client.try_fetch(snap, kind="text")
        if not got or got.get("status") != 200 or not isinstance(got.get("body"), str):
            continue
        body = got["body"]
        if len(body) < 800:
            continue
        return {
            "doc_id": f"web:{digest or ts}",
            "title": _title_of(body) or url,
            "text": body[:120_000],
            "url": url,
            "published": to_utc(ts).isoformat(),
            "wayback_url": snap,
            "genre": "web",
            "category": "web",
        }
    return None


def _title_of(html: str) -> Optional[str]:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return _clean(m.group(1))[:200] if m else None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


# --------------------------------------------------------------------------- io
def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------- main
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=30, help="max records per arXiv window")
    ap.add_argument("--skip-web", action="store_true")
    ap.add_argument("--skip-arxiv", action="store_true")
    ap.add_argument("--offline", action="store_true", help="cache only; fail loudly on a miss")
    ap.add_argument(
        "--prune-bulk",
        action="store_true",
        help="drop the raw OAI-PMH harvest from the cache once records are extracted",
    )
    args = ap.parse_args(argv)

    client = _client(offline=args.offline)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_arxiv:
        buckets: dict[str, list[dict[str, Any]]] = {"pre2021": [], "recent": [], "mid": []}
        for oai_set, start, end in ARXIV_WINDOWS:
            t0 = time.time()
            rows = fetch_arxiv_window(client, oai_set, start, end, args.limit)
            for r in rows:
                buckets[r["era"]].append(r)
            eras = {k: sum(1 for r in rows if r["era"] == k) for k in ("pre2021", "recent", "mid")}
            print(
                f"  arxiv {oai_set:18s} {start}  {len(rows):3d} records "
                f"(pre2021={eras['pre2021']} recent={eras['recent']} mid={eras['mid']})"
                f"  ({time.time()-t0:.1f}s)",
                flush=True,
            )
        for label in ("pre2021", "recent"):
            seen: set[str] = set()
            deduped = [r for r in buckets[label] if not (r["doc_id"] in seen or seen.add(r["doc_id"]))]
            n = write_jsonl(CORPUS_DIR / f"arxiv_{label}.jsonl", deduped)
            print(f"wrote data/corpus/arxiv_{label}.jsonl  ({n} records)")

    if not args.skip_web:
        web: list[dict[str, Any]] = []
        for url in WEB_SEEDS:
            t0 = time.time()
            rec = fetch_web_seed(client, url)
            if rec:
                rec["era"] = "pre2021"
                web.append(rec)
            print(f"  web  {'ok ' if rec else 'MISS'} {url}  ({time.time()-t0:.1f}s)", flush=True)
        n = write_jsonl(LOCAL_DIR / "web_pre2021.jsonl", web)
        print(f"wrote data/local/web_pre2021.jsonl  ({n} records, not committed)")

    if args.prune_bulk:
        print(f"pruned {prune_bulk_harvest():.1f} MB of raw OAI-PMH responses from the cache")

    print("cache:", json.dumps(client.stats.as_row()))
    return 0


def prune_bulk_harvest() -> float:
    """Drop the raw OAI-PMH payloads once records have been extracted.

    They are ~40 MB and entirely redundant: everything downstream reads
    ``data/corpus/*.jsonl``, which is committed.  What stays in the cache is the
    evidence that is actually *replayed* -- Wayback CDX queries, Common Crawl
    index lookups, Hacker News search, the arXiv Atom API -- about 1.5 MB, so a
    fresh clone reproduces every test and every benchmark number offline without
    carrying a bulk harvest it will never read again.

    Re-run ``python -m scripts.fetch_corpus`` to fetch them again if you want to
    rebuild the corpus from scratch.
    """
    freed = 0
    for path in CACHE_DIR.rglob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        url = (record.get("meta") or {}).get("url", "")
        if "oai2" in url or "web.archive.org/web/" in url:
            freed += path.stat().st_size
            path.unlink()
    return freed / 1e6


if __name__ == "__main__":
    raise SystemExit(main())
