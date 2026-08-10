"""Condition (C): score the evaluation corpus with Dendro.

Two evidence paths, mirroring what :class:`dendro.pipeline.Dendro` does live:

* **witnesses** -- for corpus documents these are the *real* archival records
  already fetched by ``scripts/fetch_corpus`` (arXiv OAI ``<created>``, Wayback
  CDX captures) and cached on disk.  They are attached to the record rather than
  re-queried per document, purely to avoid 500 redundant API calls.  This is a
  caching optimisation, not a shortcut: ``tests/test_live_witness.py`` takes a
  sample of the same documents, runs the full live collector against the real
  APIs, and asserts the bound comes out the same.
* **alignment** -- against the archive layer, which is how a paraphrase inherits
  its ancestor's date.

Forged documents get no such help.  Their URLs are run through the live
collector exactly as an unknown document would be, so "no archive holds this
page" and "the archive was crawling that host deeply at the time" are both
measurements against the real Internet Archive, cached for offline replay.
"""

from __future__ import annotations

import datetime as _dt
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dendro.alignment import Aligner, ArchiveLayer
from dendro.cache import Cache, HttpClient, RateLimiter
from dendro.fingerprint import ReflowFingerprint
from dendro.propagate import Calibrator, ContaminationPropagator, PropagationConfig
from dendro.types import ConsensusBound, Verdict, Witness, to_utc
from dendro.witness import Target, WitnessCollector

CACHE_DIR = REPO / "data" / "fixtures" / "cache"


@dataclass
class DendroScorer:
    """Runs condition (C) over corpus records and returns calibrated verdicts."""

    archive_entries: Sequence[Mapping] = field(default_factory=list)
    probe_forgeries: bool = True
    #: ``None`` defers to ``DENDRO_OFFLINE``.  A hard ``False`` here overrode the
    #: environment variable, so a run launched with ``DENDRO_OFFLINE=1`` still made
    #: live archive requests -- the reproducibility claim silently untrue.
    offline: Optional[bool] = None
    propagation: Optional[PropagationConfig] = None

    def __post_init__(self) -> None:
        self.fingerprinter = ReflowFingerprint()
        self.archive = ArchiveLayer(self.fingerprinter)
        for entry in self.archive_entries:
            self.archive.add(**dict(entry))
        self.aligner = Aligner(self.fingerprinter)
        self.aligner.fit_null(self.archive)
        self.client = HttpClient(
            cache=Cache(root=CACHE_DIR),
            rate_limiter=RateLimiter(),
            offline=self.offline,
            # Bounded, because the benchmark must finish.  A probe that cannot
            # answer degrades to "no coverage", which suppresses the backdate flag
            # -- the conservative direction, and it is reported in the cost table
            # rather than hidden.
            timeout=25.0,
            max_retries=2,
        )
        self.collector = WitnessCollector(client=self.client)
        self.propagator = ContaminationPropagator(self.propagation or PropagationConfig())
        self._coverage_cache: dict[tuple[str, int], dict[str, float]] = {}

    # -- one document ------------------------------------------------------
    def score(self, doc: Mapping) -> Verdict:
        doc_id = doc["doc_id"]
        text = doc.get("text") or ""
        claimed = doc.get("claimed_date")
        claimed_dt = to_utc(claimed) if claimed else None

        witnesses: list[Witness] = list(doc.get("witnesses") or [])
        target = Target(
            doc_id=doc_id, url=doc.get("url"), text=text, claimed_date=claimed_dt
        )

        coverage: dict[str, float] = {}
        if not witnesses and doc.get("url") and self.probe_forgeries:
            # Unknown document at a URL: do exactly what a live query does.
            witnesses = self.collector.collect(target)
            if claimed_dt is not None:
                coverage = self._coverage(target, claimed_dt)
        elif claimed_dt is not None and doc.get("url") and self.probe_forgeries and not witnesses:
            coverage = self._coverage(target, claimed_dt)

        # Self-asserted claims always enter the witness list so the
        # inconsistency checks have something to contradict.
        from dendro.sources.selfasserted import SelfAssertedSource

        witnesses = witnesses + SelfAssertedSource().collect(target, self.client)

        bound = self.collector.consensus(witnesses)

        ancestor = None
        if text and len(self.archive):
            fp = self.fingerprinter.fingerprint(doc_id, text)
            ancestor = self.aligner.oldest_ancestor(fp, self.archive)

        flags = self.collector.detect_inconsistencies(target, witnesses, bound, coverage)
        return self.propagator.verdict(doc_id, bound, ancestor, flags)

    def _coverage(self, target: Target, when: _dt.datetime) -> dict[str, float]:
        """Cache coverage probes by (host, year).

        Coverage is a property of a *host in an era*, not of a page, so probing
        it per document would be both slow and wrong-headed.  Caching it here is
        what makes the forgery sweep affordable, and the hit rate lands in
        ``results/cost.csv``.
        """
        key = (target.host or "", when.year)
        if key not in self._coverage_cache:
            self._coverage_cache[key] = self.collector.coverage_profile(target, when)
        return self._coverage_cache[key]

    # -- batches -----------------------------------------------------------
    def score_many(self, docs: Iterable[Mapping]) -> list[Verdict]:
        return [self.score(d) for d in docs]

    def fit_calibration(self, docs: Sequence[Mapping]) -> "DendroScorer":
        """Fit the isotonic map on a *training* split, never on the test set."""
        verdicts = self.score_many(docs)
        self.propagator.calibrator = Calibrator().fit(
            [v.human_origin_p for v in verdicts], [int(d["label_human"]) for d in docs]
        )
        return self

    @property
    def stats(self):
        return self.client.stats
