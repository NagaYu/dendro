"""WitnessCollector: independent archival evidence, combined into an existence bound.

A *witness* is a record that somebody who is not the document's author observed
the content at a particular time.  The collector gathers them from plugged-in
sources and combines them into a :class:`~dendro.types.ConsensusBound`:

    "this content existed at or before T, and for that to be false, an adversary
     would have had to compromise k independent operators"

Two modelling decisions do all the work.

**Operators, not witnesses, are the unit of independence.**  Ten Wayback
captures are one operator.  One Wayback capture plus one Common Crawl record
plus one arXiv registration is three.  Under adversarial compromise the
witnesses inside an operator are perfectly correlated -- whoever can write to
the archive can write to all of it -- while accidental clock error is
independent.  The estimator below treats those two failure modes separately,
which is why adding captures from the same crawler barely moves the confidence
while adding a second operator moves it a lot.  That asymmetry *is* the
**adversarial-robustness** claim, in arithmetic.

**The bound is an order statistic, not a minimum.**  Taking ``min(observed_at)``
would let a single injected early witness drag the bound arbitrarily far into
the past -- exactly the attack the system exists to resist.  Instead the bound is
the earliest time whose *supporting evidence mass* clears a failure-probability
budget.

**Coverage makes silence interpretable.**  Every source reports how likely it is
to have seen a document that existed at a given time.  Without that number, "no
2019 snapshot" means nothing; with it, "the archive captured this domain 340
times in 2019 and never saw this page" is quantified evidence against a claimed
2019 date, which is how backdate forgery gets caught.

Nothing in this module reads the document's prose.  A new generation of language
models changes none of it (**generator-independence**).
"""

from __future__ import annotations

import abc
import datetime as _dt
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from .cache import CacheStats, HttpClient
from .types import ConsensusBound, InconsistencyFlag, Witness, WitnessKind, to_utc, utcnow

__all__ = [
    "Target",
    "WitnessSource",
    "WitnessCollector",
    "ConsensusConfig",
    "combine_failure_probability",
    "SOURCE_REGISTRY",
    "register_source",
]


# --------------------------------------------------------------------------- target
@dataclass
class Target:
    """What we are trying to date.

    A target may be a URL, a local file, raw text, or a bare identifier.  Sources
    declare which of those they can act on, so the collector degrades cleanly:
    text-only input simply gets fewer sources, a wider interval, and an honest
    ``abstained`` verdict rather than a fabricated bound.
    """

    doc_id: str
    url: Optional[str] = None
    text: Optional[str] = None
    path: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    claimed_date: Optional[_dt.datetime] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.claimed_date is not None:
            self.claimed_date = to_utc(self.claimed_date)
        if self.url and not self.arxiv_id:
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z-]+/[0-9]{7})", self.url, re.I)
            if m:
                self.arxiv_id = m.group(1)
        if self.url and not self.doi:
            m = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", self.url)
            if m:
                self.doi = m.group(1)

    @property
    def host(self) -> Optional[str]:
        if not self.url:
            return None
        import urllib.parse

        return urllib.parse.urlsplit(self.url).netloc.lower() or None


# --------------------------------------------------------------------------- source base
class WitnessSource(abc.ABC):
    """Plug-in interface for one archival evidence provider.

    Implementations must satisfy three contracts:

    1. **Never invent a bound.**  If the API is unreachable and nothing is
       cached, return ``[]``.  Fewer witnesses widen the interval; a fabricated
       witness corrupts it.
    2. **Report coverage honestly.**  :meth:`coverage` is what turns silence into
       evidence, and an inflated value produces false backdating accusations.
    3. **Go through the client.**  All I/O via :class:`~dendro.cache.HttpClient`
       so that caching, rate limiting, and offline replay hold universally.
    """

    source_id: str = "abstract"
    operator: str = "unknown"
    kind: WitnessKind = WitnessKind.SNAPSHOT
    reliability: float = 0.99
    forgeability: float = 1e-2

    def supports(self, target: Target) -> bool:
        return True

    @abc.abstractmethod
    def collect(self, target: Target, client: HttpClient) -> list[Witness]:
        """Return every witness this source can produce for the target."""

    def coverage(self, target: Target, when: _dt.datetime, client: HttpClient) -> float:
        """P(this source would have observed the target had it existed at ``when``).

        Default 0.0 -- i.e. "I cannot rule anything out".  Sources that can
        actually measure their own reach (by counting neighbouring captures)
        override this, and only those contribute to backdate detection.
        """
        return 0.0


SOURCE_REGISTRY: dict[str, type[WitnessSource]] = {}


def register_source(cls: type[WitnessSource]) -> type[WitnessSource]:
    """Decorator: make a source constructible by name from the CLI."""
    SOURCE_REGISTRY[cls.source_id] = cls
    return cls


# --------------------------------------------------------------------------- consensus math
@dataclass(frozen=True)
class ConsensusConfig:
    """Failure-probability budgets for the bound and its interval.

    ``alpha`` is the headline budget: the reported ``not_after`` is the earliest
    time whose supporting evidence would have to fail with probability below
    ``alpha`` for the bound to be wrong.  ``alpha_tight``/``alpha_loose`` produce
    the conservative and optimistic ends of the reported interval, so the bound
    always travels with an honest width (**calibration**).

    The default of 1e-2 is chosen with the group model in mind.  Because
    ``q_g = f_g + (1-f_g)*accidental`` is strictly greater than ``f_g``, a budget
    at or below an operator's own forgeability makes that operator *unable to
    ever* clear it alone -- so ``alpha = 1e-3`` silently means "two operators
    required".  That is a legitimate policy (``--alpha 1e-3`` selects it) but it
    is a surprising default, and hiding a quorum requirement inside a
    probability threshold is the kind of thing that makes a number untrustworthy.
    At 1e-2 a single Internet Archive capture *does* establish a bound, the
    ``single_operator`` flag says so out loud, and ``not_after_high`` shows what
    the date becomes once corroboration is demanded.
    """

    alpha: float = 1e-2
    alpha_tight: float = 1e-4
    alpha_loose: float = 5e-2
    within_operator_rho: float = 0.15   # design effect for repeat captures
    require_independent: bool = True     # SELF_ASSERTED never tightens a bound


def combine_failure_probability(
    witnesses: Sequence[Witness], within_operator_rho: float = 0.15
) -> tuple[float, int, float]:
    """P(every one of these witnesses is wrong), plus operator count and ESS.

    Failure decomposes into two modes with different correlation structure:

    *Adversarial* -- an operator is compromised.  All of that operator's
    witnesses fall together, so the group contributes ``min(forgeability)``
    regardless of how many captures it holds.

    *Accidental* -- a clock is wrong or a record is sloppy.  Independent within a
    group, so ``prod(1 - reliability)`` over the group's witnesses.

    A group fails if it is compromised, or if it is honest but every one of its
    witnesses is accidentally wrong:

        ``q_g = f_g + (1 - f_g) * prod_w (1 - r_w)``

    Groups are independent, so ``P = prod_g q_g``.  The consequence is the point:
    twenty Wayback captures buy you ``q ~ f_wayback`` and no more, while one
    Wayback plus one Common Crawl buys you ``f_wayback * f_cc`` -- six orders of
    magnitude better.  Evidence diversity, not evidence volume.
    """
    if not witnesses:
        return 1.0, 0, 0.0
    groups: dict[str, list[Witness]] = defaultdict(list)
    for w in witnesses:
        groups[w.operator].append(w)

    log_p = 0.0
    ess = 0.0
    for members in groups.values():
        f_g = min(w.forgeability for w in members)
        accidental = 1.0
        for w in members:
            accidental *= max(1e-9, 1.0 - w.reliability)
        q_g = min(1.0, f_g + (1.0 - f_g) * accidental)
        log_p += math.log(max(q_g, 1e-300))
        ess += 1.0 + (len(members) - 1) * within_operator_rho
    return math.exp(log_p), len(groups), ess


class WitnessCollector:
    """Gathers witnesses from every registered source and forms a consensus bound.

    The collector owns an :class:`~dendro.cache.HttpClient`, so a whole corpus
    annotation run shares one cache and one rate-limit budget.  ``stats`` is the
    per-document cost that benchmark axis (5) reports.
    """

    def __init__(
        self,
        sources: Optional[Sequence[WitnessSource]] = None,
        client: Optional[HttpClient] = None,
        config: Optional[ConsensusConfig] = None,
    ) -> None:
        self.client = client if client is not None else HttpClient()
        self.config = config or ConsensusConfig()
        if sources is None:
            from .sources import default_sources  # local import avoids a cycle

            sources = default_sources()
        self.sources = list(sources)

    @property
    def stats(self) -> CacheStats:
        return self.client.stats

    # -- collection --------------------------------------------------------
    def collect(self, target: Target) -> list[Witness]:
        """Query every applicable source.  A failing source costs evidence, not the run.

        Demonstrates **generator-independence**: the inputs are a URL and
        identifiers, never the prose, so no property of the text can change what
        comes back.
        """
        out: list[Witness] = []
        for src in self.sources:
            if not src.supports(target):
                continue
            try:
                out.extend(src.collect(target, self.client))
            except Exception:
                # A source that raises is a source that contributed nothing.  The
                # interval widens accordingly; that is the correct failure mode.
                continue
        return _dedupe(sorted(out, key=lambda w: w.observed_at))

    def coverage_profile(self, target: Target, when: _dt.datetime) -> dict[str, float]:
        """Per-source P(would have seen it at ``when``).  Feeds backdate detection.

        Demonstrates **adversarial-robustness**: this is the measurement that turns
        archival silence into usable evidence, and its absence is what keeps an
        obscure genuine document from being accused.
        """
        prof: dict[str, float] = {}
        for src in self.sources:
            if not src.supports(target):
                continue
            try:
                prof[src.source_id] = float(src.coverage(target, when, self.client))
            except Exception:
                prof[src.source_id] = 0.0
        return prof

    # -- consensus ---------------------------------------------------------
    def consensus(self, witnesses: Sequence[Witness]) -> ConsensusBound:
        """Earliest time whose supporting evidence clears the failure budget.

        Scanning candidate times in ascending order and stopping at the first one
        that clears ``alpha`` is what makes the estimator robust: an injected
        early witness sits alone in its operator group, so ``q_g = f_g`` for that
        one group and the product never clears the budget on its own.  It takes
        *collusion across operators* to move the bound, which is the security
        property being claimed.
        """
        cfg = self.config
        usable = [w for w in witnesses if (w.is_independent_evidence or not cfg.require_independent)]
        if not usable:
            return ConsensusBound(
                not_after=None,
                all_witnesses=tuple(witnesses),
                method="no-independent-evidence",
            )

        ordered = sorted(usable, key=lambda w: w.observed_at)
        times = sorted({w.observed_at for w in ordered})

        def support(t: _dt.datetime) -> list[Witness]:
            return [w for w in ordered if w.observed_at <= t]

        def first_clearing(alpha: float) -> Optional[_dt.datetime]:
            for t in times:
                p, _, _ = combine_failure_probability(support(t), cfg.within_operator_rho)
                if p <= alpha:
                    return t
            return None

        not_after = first_clearing(cfg.alpha)
        method = "weighted-order-statistic"
        if not_after is None:
            # Even all the evidence together is weak.  Report the weakest true
            # statement we can make rather than nothing, and let the log-odds
            # tell downstream how little it is worth.
            not_after = times[-1]
            method = "best-effort-below-alpha"

        supporting = tuple(support(not_after))
        p_fail, n_ops, ess = combine_failure_probability(supporting, cfg.within_operator_rho)
        logodds = -math.log(max(p_fail, 1e-300))

        return ConsensusBound(
            not_after=not_after,
            not_after_low=first_clearing(cfg.alpha_loose) or not_after,
            not_after_high=first_clearing(cfg.alpha_tight) or times[-1],
            independent_operators=n_ops,
            effective_witnesses=ess,
            forgery_logodds=logodds,
            supporting=supporting,
            all_witnesses=tuple(sorted(witnesses, key=lambda w: w.observed_at)),
            method=method,
        )

    def date(self, target: Target) -> ConsensusBound:
        """Collect then combine -- the one-call path used by ``dendro date``.

        Demonstrates **adversarial-robustness**: the returned bound carries
        ``forgery_logodds``, so how hard it would be to fake travels with the date
        rather than being left to the reader to assume.
        """
        return self.consensus(self.collect(target))

    # -- inconsistency -----------------------------------------------------
    def detect_inconsistencies(
        self,
        target: Target,
        witnesses: Sequence[Witness],
        bound: ConsensusBound,
        coverage: Optional[Mapping[str, float]] = None,
    ) -> list[InconsistencyFlag]:
        """Find conflicts between claims and independent observation.

        The headline case is **backdate forgery**: a document asserts 2019, but
        archives that demonstrably covered its neighbourhood in 2019 first saw it
        in 2025.  The likelihood ratio is

            ``log LR = -k * log(1 - c)``

        where ``c`` is the per-source coverage probability and ``k`` the number of
        covering sources.  Under the genuine hypothesis the probability that
        *every* covering source missed a document that existed is ``(1-c)^k``;
        under the forgery hypothesis missing it is certain.  The formula has the
        right shape: with no coverage the LR is exactly 1 and no accusation is
        made, which is the difference between evidence of absence and absence of
        evidence.

        Demonstrates **adversarial-robustness**: this fires on the backdated-metadata
        attack, which statistical detectors are structurally blind to because
        they never look at metadata at all.
        """
        flags: list[InconsistencyFlag] = []
        claimed = target.claimed_date
        if claimed is None:
            claimed = _earliest_self_asserted(witnesses)

        independent = [w for w in witnesses if w.is_independent_evidence]
        earliest_independent = min((w.observed_at for w in independent), default=None)

        if claimed is not None and earliest_independent is not None:
            gap_days = (earliest_independent - claimed).days
            if gap_days > 365:
                cov = coverage or {}
                covering = [(s, c) for s, c in cov.items() if c > 0.05]
                log_lr = -sum(math.log(max(1e-6, 1.0 - c)) for _, c in covering)
                # An unexplained multi-year gap is suspicious even without a
                # coverage measurement, but only weakly so.
                log_lr = max(log_lr, 0.0) + min(1.0, gap_days / 3650.0)
                if log_lr > 0.5:
                    if covering:
                        why = (
                            f"and {len(covering)} source(s) were demonstrably covering this "
                            f"neighbourhood at that time"
                        )
                    else:
                        # Without a coverage measurement this is only the gap
                        # heuristic, and the message must not imply otherwise --
                        # an unmeasured gap is a reason to look, not a finding.
                        why = (
                            "but archive coverage at that time was not measured, so this is a weak "
                            "signal only; re-run with coverage probing for a usable likelihood ratio"
                        )
                    flags.append(
                        InconsistencyFlag(
                            kind="backdate",
                            log_lr=log_lr,
                            detail=(
                                f"document claims {claimed.date()} but the earliest independent "
                                f"observation is {earliest_independent.date()} ({gap_days} days later), "
                                f"{why}"
                            ),
                            claimed=claimed,
                            observed=earliest_independent,
                            coverage=max((c for _, c in covering), default=0.0),
                            sources=tuple(s for s, _ in covering),
                        )
                    )

        # The pure-forgery case: a document claims to be old and *nothing*
        # independent corroborates it -- in a neighbourhood the archives were
        # demonstrably crawling.  This is the shape the backdating attack takes
        # when the forger controls the whole page, and it is invisible to any
        # detector that only reads prose.  Note the guard: without a positive
        # coverage measurement no flag is raised at all, so an obscure genuine
        # document is never accused merely for being obscure.
        if claimed is not None and earliest_independent is None:
            cov = coverage or {}
            covering = [(s, c) for s, c in cov.items() if c > 0.05]
            age_days = (utcnow() - claimed).days
            if covering and age_days > 730:
                log_lr = -sum(math.log(max(1e-6, 1.0 - c)) for _, c in covering)
                flags.append(
                    InconsistencyFlag(
                        kind="backdate",
                        log_lr=log_lr,
                        detail=(
                            f"document claims {claimed.date()} but no independent archive holds any "
                            f"record of it, while {len(covering)} source(s) were crawling this "
                            f"neighbourhood at that time (max coverage {max(c for _, c in covering):.2f})"
                        ),
                        claimed=claimed,
                        observed=None,
                        coverage=max((c for _, c in covering), default=0.0),
                        sources=tuple(s for s, _ in covering),
                    )
                )

        if claimed is not None and earliest_independent is not None:
            if (claimed - earliest_independent).days > 365:
                flags.append(
                    InconsistencyFlag(
                        kind="postdate",
                        log_lr=1.0,
                        detail=(
                            f"an archive observed this content on {earliest_independent.date()}, "
                            f"before its own claimed date {claimed.date()}"
                        ),
                        claimed=claimed,
                        observed=earliest_independent,
                    )
                )

        # A commit that predates the repository it lives in is a rewritten
        # history.  The commit date is author-controlled; the repository creation
        # timestamp lives on the forge's servers and is not.  This is the one
        # check a git-based backdate cannot survive.
        for w in witnesses:
            if w.kind is not WitnessKind.COMMIT:
                continue
            created_raw = (w.raw or {}).get("repo_created_at")
            if not created_raw:
                continue
            try:
                repo_created = to_utc(created_raw)
            except (ValueError, TypeError):
                continue
            if (repo_created - w.observed_at).days > 30:
                flags.append(
                    InconsistencyFlag(
                        kind="commit_predates_repo",
                        log_lr=5.0,
                        detail=(
                            f"commit dated {w.observed_at.date()} sits in a repository the forge "
                            f"says was created {repo_created.date()}; the commit date was rewritten"
                        ),
                        claimed=w.observed_at,
                        observed=repo_created,
                        coverage=1.0,
                        sources=(w.source_id,),
                    )
                )

        # Two operators that disagree by years about first sighting, where both
        # had coverage, is a sign that one of them is looking at different content.
        by_op: dict[str, _dt.datetime] = {}
        for w in independent:
            by_op[w.operator] = min(by_op.get(w.operator, w.observed_at), w.observed_at)
        if len(by_op) >= 2:
            first, last = min(by_op.values()), max(by_op.values())
            if (last - first).days > 3 * 365:
                flags.append(
                    InconsistencyFlag(
                        kind="operator_disagreement",
                        log_lr=0.6,
                        detail=(
                            "independent operators disagree by "
                            f"{(last - first).days // 365} years on first observation"
                        ),
                        observed=first,
                        sources=tuple(sorted(by_op)),
                    )
                )

        if bound.has_evidence and bound.independent_operators == 1:
            flags.append(
                InconsistencyFlag(
                    kind="single_operator",
                    log_lr=0.0,
                    detail=(
                        f"bound rests on a single operator ({bound.supporting[0].operator}); "
                        "compromising it would be sufficient to move the date"
                    ),
                    observed=bound.not_after,
                    sources=(bound.supporting[0].operator,),
                )
            )
        return flags


def _dedupe(witnesses: Sequence[Witness]) -> list[Witness]:
    """Collapse witnesses that are the same observation seen twice.

    An archive that indexed both the ``http://`` and ``https://`` form of a URL in
    one crawl is one observation, not two.  Left uncollapsed it would inflate the
    within-operator accidental-error term -- an artefact of URL normalisation
    masquerading as corroboration -- and make the evidence table misleading to
    anyone reading it.
    """
    seen: set[tuple] = set()
    out: list[Witness] = []
    for w in witnesses:
        key = (w.operator, w.kind, w.observed_at.date(), (w.raw or {}).get("field"))
        if key in seen:
            continue
        seen.add(key)
        out.append(w)
    return out


def _earliest_self_asserted(witnesses: Iterable[Witness]) -> Optional[_dt.datetime]:
    times = [w.observed_at for w in witnesses if w.kind is WitnessKind.SELF_ASSERTED]
    return min(times) if times else None
