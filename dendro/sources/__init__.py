"""Pluggable witness sources, one module per archive operator.

The registry exists so that "independence" is a property you can *count* at the
command line::

    dendro date https://example.org/post --sources wayback,commoncrawl,crossref

Each source declares the ``operator`` that runs it, and
:func:`dendro.witness.combine_failure_probability` groups by that field.  Adding
another Wayback-derived source would therefore buy nothing; adding a source run
by a different institution buys a multiplicative reduction in the probability
that the bound is forged.  That is the design pressure the plugin interface is
meant to create.

Forgeability is assigned per source, and the numbers are opinions with reasons:

===================  ============  =========================================
source               forgeability  why
===================  ============  =========================================
wayback              1e-3          "Save Page Now" can create a capture *today*;
                                   it cannot create one dated 2019.  Backdating
                                   requires compromising the archive itself.
commoncrawl          1e-3          Published WARC files with fixed release dates.
arxiv                5e-4          Immutable public record, versioned, mirrored.
crossref (created)   2e-3          Deposit timestamp, not author-controlled.
crossref (issued)    5e-2          Publisher-asserted; a cooperative publisher
                                   could restate it.
hackernews           3e-3          Third-party posting archive with its own clock.
github               3e-2          API-reported commit dates come from the commit
                                   object, which the author signs -- see below.
git (local)          5e-1          ``GIT_AUTHOR_DATE`` is a environment variable.
                                   A git date is barely evidence at all, and
                                   modelling it as such is the honest choice.
self-asserted        9.8e-1        The document's own claim.  Never tightens a
                                   bound; exists only to be contradicted.
===================  ============  =========================================
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..witness import SOURCE_REGISTRY, WitnessSource
from .commoncrawl import CommonCrawlSource
from .hackernews import HackerNewsSource
from .scholarly import ArxivSource, CrossrefSource
from .selfasserted import SelfAssertedSource
from .vcs import GitHubSource, LocalGitSource
from .wayback import WaybackSource

__all__ = [
    "WaybackSource",
    "CommonCrawlSource",
    "ArxivSource",
    "CrossrefSource",
    "HackerNewsSource",
    "LocalGitSource",
    "GitHubSource",
    "SelfAssertedSource",
    "default_sources",
    "build_sources",
    "SOURCE_REGISTRY",
]


def default_sources() -> list[WitnessSource]:
    """The stock line-up: six independent operators plus the self-asserted probe."""
    return [
        WaybackSource(),
        CommonCrawlSource(),
        ArxivSource(),
        CrossrefSource(),
        HackerNewsSource(),
        LocalGitSource(),
        GitHubSource(),
        SelfAssertedSource(),
    ]


def build_sources(names: Optional[Sequence[str]] = None) -> list[WitnessSource]:
    """Instantiate sources by ``source_id``; ``None`` gives :func:`default_sources`."""
    if not names:
        return default_sources()
    out: list[WitnessSource] = []
    for name in names:
        key = name.strip()
        if not key:
            continue
        if key not in SOURCE_REGISTRY:
            raise KeyError(f"unknown source {key!r}; known: {', '.join(sorted(SOURCE_REGISTRY))}")
        out.append(SOURCE_REGISTRY[key]())
    return out
