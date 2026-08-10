"""Assemble and publish the Gradio Space.

    python -m scripts.publish_space                      # dry run: stage and verify
    python -m scripts.publish_space --push               # publish to NagaYu/dendro

The Space is a *subset* of this repository, assembled here rather than pushed
wholesale. It carries the library, the app, the figures and the fixture cache —
about 4 MB — and leaves out the corpus, the benchmark harness and the tests,
which a hosted demo never reads.

The fixture cache is the interesting inclusion. With it, the bundled examples
answer from recorded Internet Archive / Common Crawl / arXiv / Hacker News
responses in milliseconds and without touching those APIs at all; only URLs a
visitor types themselves reach the network, rate-limited to 0.5 req/s per host.
A public demo that hammers the archives on every page view would be a bad
citizen, and this is how that is avoided.

Note: Hugging Face requires a PRO subscription to host Gradio Spaces on free CPU
hardware (static Spaces remain free for everyone). ``--push`` fails with a clear
402 if the account cannot host one; nothing is created in that case.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Everything the hosted app needs, and nothing else.
INCLUDE_FILES = ("app.py", "requirements.txt", "LICENSE", "DATA.md")
INCLUDE_DIRS = ("dendro", "figures")
CACHE_DIR = pathlib.Path("data") / "fixtures" / "cache"

SPACE_README = REPO / "space" / "README.md"


def stage(dest: pathlib.Path) -> pathlib.Path:
    """Build the Space tree at ``dest`` and return it."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for name in INCLUDE_FILES:
        shutil.copy2(REPO / name, dest / name)
    for name in INCLUDE_DIRS:
        shutil.copytree(REPO / name, dest / name, ignore=shutil.ignore_patterns("__pycache__"))
    (dest / CACHE_DIR).parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / CACHE_DIR, dest / CACHE_DIR)

    if not SPACE_README.is_file():
        raise SystemExit(f"missing Space card: {SPACE_README}")
    shutil.copy2(SPACE_README, dest / "README.md")
    return dest


def verify(dest: pathlib.Path) -> None:
    """Run the app's own analyse() from the staged tree, offline.

    Staging can silently drop a module or the cache and the failure would only
    show up as a broken Space after a push. Importing and exercising the app from
    the staged directory catches that here instead.
    """
    code = (
        "import os,sys; sys.path.insert(0,'.'); os.environ['DENDRO_OFFLINE']='1';"
        "import app;"
        "md,_,rows,_,_ = app.analyse('https://arxiv.org/abs/1706.03762','','',False);"
        "assert '2017-06-12' in md, md[:200];"
        "assert rows, 'no witnesses';"
        "print('  staged app OK ->', md.splitlines()[0], f'({len(rows)} witnesses)')"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=dest, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"staged Space failed to run:\n{proc.stdout}\n{proc.stderr[-2000:]}")
    print(proc.stdout.strip())


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="NagaYu/dendro")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dest", help="where to stage (default: a temp directory)")
    args = ap.parse_args(argv)

    dest = pathlib.Path(args.dest) if args.dest else pathlib.Path(tempfile.mkdtemp()) / "dendro-space"
    print(f"staging Space -> {dest}")
    stage(dest)
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"  {sum(1 for _ in dest.rglob('*') if _.is_file())} files, {size / 1e6:.1f} MB")
    verify(dest)

    if not args.push:
        print(f"\ndry run — staged and verified. Pass --push to publish to {args.repo_id}.")
        return 0

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk="gradio",
                        private=args.private, exist_ok=True)
    except HfHubHTTPError as exc:
        if "402" in str(exc):
            raise SystemExit(
                "Hugging Face returned 402: hosting a Gradio Space on free CPU needs a PRO "
                "subscription (static Spaces are free). Nothing was created.\n"
                "Subscribe at https://huggingface.co/pro, then re-run this command."
            ) from exc
        raise

    api.upload_folder(
        folder_path=str(dest),
        repo_id=args.repo_id,
        repo_type="space",
        commit_message="Dendro: prove a document existed before a date, without reading it",
    )
    print(f"pushed https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
