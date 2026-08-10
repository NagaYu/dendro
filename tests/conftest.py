"""Shared fixtures.

Every test runs **offline**.  The repository ships a cache of real responses from
the Internet Archive, Common Crawl, arXiv and Hacker News under
``data/fixtures/cache``, so the suite exercises genuine archive data with no
network and no flakiness.  A request that was never recorded raises
:class:`~dendro.cache.OfflineCacheMiss`, the source degrades to "contributed
nothing", and the assertion that fails is about *evidence*, not about wifi.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dendro.cache import Cache, HttpClient, RateLimiter  # noqa: E402

FIXTURE_CACHE = REPO / "data" / "fixtures" / "cache"
CORPUS_DIR = REPO / "data" / "corpus"


@pytest.fixture(scope="session")
def fixture_cache_dir() -> pathlib.Path:
    if not FIXTURE_CACHE.exists():
        pytest.skip("fixture cache missing — run `python -m scripts.fetch_corpus`")
    return FIXTURE_CACHE


@pytest.fixture
def offline_client(tmp_path, fixture_cache_dir) -> HttpClient:
    """A client that can only read the committed fixture cache.

    The writable root is a throwaway temp dir, so a test can never silently
    populate the committed cache and make itself pass on the second run.
    """
    cache = Cache(root=tmp_path / "cache", overlays=[fixture_cache_dir])
    return HttpClient(cache=cache, rate_limiter=RateLimiter(), offline=True)


@pytest.fixture
def no_network_client(tmp_path) -> HttpClient:
    """Offline client with *no* fixtures — every source must return nothing."""
    return HttpClient(cache=Cache(root=tmp_path / "empty", overlays=[]), offline=True)


@pytest.fixture(scope="session")
def real_documents() -> list[dict]:
    """Real pre-2021 arXiv abstracts with genuine submission dates."""
    path = CORPUS_DIR / "arxiv_pre2021.jsonl"
    if not path.is_file():
        pytest.skip("corpus missing — run `python -m scripts.fetch_corpus`")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) < 60:
        pytest.skip("corpus too small for the alignment tests")
    return rows


@pytest.fixture(scope="session")
def recent_documents() -> list[dict]:
    path = CORPUS_DIR / "arxiv_recent.jsonl"
    if not path.is_file():
        pytest.skip("corpus missing — run `python -m scripts.fetch_corpus`")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="session")
def small_corpus(real_documents):
    """A deterministic, cheap evaluation corpus for the behavioural tests."""
    from benchmarks.corpus import build_corpus

    return build_corpus(seed=4242, n_synthetic_per_generation=14, n_paraphrase=18, n_backdate=10)


@pytest.fixture(scope="session")
def archive_layer(small_corpus):
    from benchmarks.corpus import archive_entries
    from dendro.alignment import Aligner, ArchiveLayer
    from dendro.fingerprint import ReflowFingerprint

    rf = ReflowFingerprint()
    layer = ArchiveLayer(rf)
    for entry in archive_entries(small_corpus):
        layer.add(**entry)
    aligner = Aligner(rf).fit_null(layer)
    return layer, aligner, rf
