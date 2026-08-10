"""Version-control witnesses -- and an honest account of how weak they are.

Git commit dates are the most commonly cited "proof" that a file is old, and
they are almost worthless as evidence::

    GIT_AUTHOR_DATE="2019-03-01T09:00:00" GIT_COMMITTER_DATE="2019-03-01T09:00:00" \\
        git commit -m "original draft"

Two environment variables and the file "existed" in 2019.  Modelling that at
``forgeability = 0.5`` rather than the 1e-3 given to an archive crawl is not
pessimism; it is the difference between a provenance system and a decoration.
A repository full of backdated commits will produce a bound whose
``forgery_logodds`` is near zero, and :mod:`dendro.propagate` will correctly
refuse to draw a confident conclusion from it.

What *is* good evidence is the **inconsistency** this source can expose.  A
commit claiming 2019 inside a repository GitHub says was created in 2025 is a
contradiction the forger cannot fix, because the repository creation timestamp
lives on GitHub's servers and not in the object graph.  ``GitHubSource`` reports
it in ``raw['repo_created_at']`` and :meth:`dendro.witness.WitnessCollector.detect_inconsistencies`
turns it into a flag -- a concrete instance of the **adversarial-robustness**
claim.
"""

from __future__ import annotations

import datetime as _dt
import pathlib
import re
import subprocess
from typing import Any, Optional

from ..cache import HttpClient
from ..types import Witness, WitnessKind, to_utc
from ..witness import Target, WitnessSource, register_source

GITHUB_API = "https://api.github.com"
_GITHUB_URL_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/|$)", re.I)


@register_source
class LocalGitSource(WitnessSource):
    """Earliest commit touching a file in a local repository.

    Emitted at high ``forgeability`` on purpose.  This source exists so that a
    user pointing Dendro at their own working tree gets a *labelled weak*
    witness rather than silence -- and so that the resulting bound visibly fails
    to reach the confidence budget on its own.
    """

    source_id = "git"
    operator = "self-vcs"
    kind = WitnessKind.COMMIT
    reliability = 0.9
    forgeability = 0.5

    def supports(self, target: Target) -> bool:
        return bool(target.path)

    def collect(self, target: Target, client: HttpClient) -> list[Witness]:
        path = pathlib.Path(target.path or "")
        if not path.exists():
            return []
        repo = _find_repo(path)
        if repo is None:
            return []
        try:
            proc = subprocess.run(
                ["git", "log", "--follow", "--reverse", "--format=%H%x1f%aI%x1f%cI", "--", str(path)],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        first = proc.stdout.strip().splitlines()[0]
        parts = first.split("\x1f")
        if len(parts) < 3:
            return []
        sha, author_date, commit_date = parts[0], parts[1], parts[2]
        try:
            observed = min(to_utc(author_date), to_utc(commit_date))
        except ValueError:
            return []
        return [
            Witness(
                source_id=self.source_id,
                operator=self.operator,
                kind=self.kind,
                observed_at=observed,
                target=str(path),
                reliability=self.reliability,
                forgeability=self.forgeability,
                url=None,
                raw={"sha": sha, "author_date": author_date, "commit_date": commit_date, "repo": str(repo)},
            )
        ]


@register_source
class GitHubSource(WitnessSource):
    """Commit dates from the GitHub API, plus the repository creation timestamp.

    ``repo_created_at`` is the interesting field.  It is GitHub's own record, not
    the author's, so a commit that predates it is proof of a rewritten history.
    """

    source_id = "github"
    operator = "github"
    kind = WitnessKind.COMMIT
    reliability = 0.97
    forgeability = 3e-2

    def supports(self, target: Target) -> bool:
        return bool(target.url and _GITHUB_URL_RE.search(target.url or ""))

    def collect(self, target: Target, client: HttpClient) -> list[Witness]:
        m = _GITHUB_URL_RE.search(target.url or "")
        if not m:
            return []
        owner, repo = m.group(1), m.group(2)

        repo_info = client.try_fetch(f"{GITHUB_API}/repos/{owner}/{repo}", kind="json")
        repo_created = None
        if repo_info and repo_info.get("status") == 200 and isinstance(repo_info.get("body"), dict):
            created = repo_info["body"].get("created_at")
            if created:
                try:
                    repo_created = to_utc(created)
                except ValueError:
                    repo_created = None

        path = _path_in_repo(target.url or "")
        params: dict[str, Any] = {"per_page": 1}
        if path:
            params["path"] = path
        commits = client.try_fetch(f"{GITHUB_API}/repos/{owner}/{repo}/commits", params, kind="json")

        out: list[Witness] = []
        if repo_created is not None:
            # Not an existence bound on the content -- it is a floor on the
            # repository.  Recorded as a REGISTRATION witness about the repo so
            # the inconsistency check can reach it, with coverage 0 so it never
            # tightens the document's bound by itself.
            out.append(
                Witness(
                    source_id=self.source_id,
                    operator=self.operator,
                    kind=WitnessKind.REGISTRATION,
                    observed_at=repo_created,
                    target=f"{owner}/{repo}",
                    reliability=0.999,
                    forgeability=1e-3,
                    url=f"https://github.com/{owner}/{repo}",
                    raw={"field": "repo_created_at", "repo_created_at": repo_created.isoformat()},
                )
            )
        body = commits.get("body") if commits and commits.get("status") == 200 else None
        if isinstance(body, list) and body:
            commit = body[-1].get("commit", {})
            when = (commit.get("committer") or {}).get("date") or (commit.get("author") or {}).get("date")
            if when:
                try:
                    observed = to_utc(when)
                except ValueError:
                    observed = None
                if observed is not None:
                    out.append(
                        Witness(
                            source_id=self.source_id,
                            operator=self.operator,
                            kind=WitnessKind.COMMIT,
                            observed_at=observed,
                            target=path or f"{owner}/{repo}",
                            reliability=self.reliability,
                            forgeability=self.forgeability,
                            url=body[-1].get("html_url"),
                            raw={
                                "sha": body[-1].get("sha"),
                                "repo_created_at": repo_created.isoformat() if repo_created else None,
                            },
                        )
                    )
        return out


def _find_repo(path: pathlib.Path) -> Optional[pathlib.Path]:
    for parent in [path if path.is_dir() else path.parent, *path.resolve().parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _path_in_repo(url: str) -> Optional[str]:
    m = re.search(r"github\.com/[\w.-]+/[\w.-]+/(?:blob|tree)/[^/]+/(.+)$", url)
    return m.group(1) if m else None
