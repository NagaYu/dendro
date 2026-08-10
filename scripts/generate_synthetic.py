"""Produce class (iii) — synthetic documents — with a swappable backend.

The benchmark ships with an **offline** generator so that every number in
``results/`` reproduces on a laptop with no API key and no model download. That
choice has a cost, and the README states it plainly: an n-gram model at varying
coherence reproduces the *statistical axis* that perplexity detectors consume,
but it cannot reproduce an **architecture shift**. The published failure mode of
learned detectors — trained on one model family, deployed against the next — is
therefore under-represented by the offline ladder, and the measured robustness of
baseline (B) in ``results/generalization.csv`` should be read with that in mind.

This script exists so that limitation is fixable by anyone who wants to spend the
tokens::

    # offline, the default the benchmark uses
    python -m scripts.generate_synthetic --n 300 --out data/corpus/synthetic_ngram.jsonl

    # real model output, if you have credentials
    export ANTHROPIC_API_KEY=...
    python -m scripts.generate_synthetic --backend anthropic \\
        --model claude-sonnet-5 --n 300 --out data/corpus/synthetic_llm.jsonl

    # then re-run with the substituted class (iii)
    python -m benchmarks.run --synthetic data/corpus/synthetic_llm.jsonl

Note what does *not* change when you swap the backend: condition (C). Dendro
never reads the prose, so a better generator alters its inputs not at all. That
is the claim, and this script is the way to try to falsify it.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from typing import Iterable, Optional

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.corpus import load_real_documents  # noqa: E402
from benchmarks.generators import GENERATION_LADDER, NgramLM, SyntheticGenerator  # noqa: E402

PROMPTS = (
    "Write the abstract of a research paper about {topic}. Return only the abstract.",
    "Draft a 180-word technical abstract on {topic} in the style of an academic preprint.",
    "Summarise, as a standalone abstract, a hypothetical study of {topic}.",
    "Produce an abstract for a preprint reporting new results on {topic}.",
)

TOPICS = (
    "variance reduction in stochastic optimisation",
    "tidal heating in exoplanetary systems",
    "identification in dynamic panel data models",
    "cortical representations of temporal sequences",
    "phase transitions in disordered spin systems",
    "distribution shift in medical image segmentation",
    "sparse recovery under adversarial noise",
    "long-range dependence in network traffic",
    "kinetics of protein folding intermediates",
    "instrumental variables with weak instruments",
)


# --------------------------------------------------------------------------- offline
def generate_ngram(n: int, seed: int, tokens_per_doc: int) -> list[dict]:
    """The default: the same generator the committed benchmark uses."""
    old, _, _ = load_real_documents()
    if not old:
        raise SystemExit("no corpus — run `python -m scripts.fetch_corpus` first")
    rng = random.Random(seed)
    pool = list(old)
    rng.shuffle(pool)
    lm = NgramLM(max_order=4).fit(d["text"] for d in pool[:900])
    gen = SyntheticGenerator(lm)

    out: list[dict] = []
    per_rung = max(1, n // len(GENERATION_LADDER))
    for cfg in GENERATION_LADDER:
        out.extend(gen.generate(cfg, per_rung, tokens_per_doc=tokens_per_doc, seed=seed))
    return out[:n]


# --------------------------------------------------------------------------- LLM
def generate_anthropic(n: int, model: str, seed: int, max_tokens: int) -> list[dict]:
    """Real model output, one document per call.

    Deliberately unbatched and rate-unlimited-by-you: this spends real tokens, so
    the caller should see exactly how many calls they are authorising.
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise SystemExit("pip install anthropic, and set ANTHROPIC_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    client = Anthropic()
    rng = random.Random(seed)
    out: list[dict] = []
    for i in range(n):
        topic = TOPICS[rng.randrange(len(TOPICS))]
        template = PROMPTS[rng.randrange(len(PROMPTS))]
        prompt = template.format(topic=topic)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=1.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        if not text.strip():
            continue
        out.append(
            {
                "doc_id": f"syn:{model}:{i:04d}",
                "text": text.strip(),
                "generator": model,
                # A real model is treated as a *later* generation than anything on
                # the offline ladder, which is the honest ordering: it is closer to
                # human text than an n-gram model at any coherence setting.
                "generation": 9,
                "family": "llm",
                "prompt": prompt,
                "label_human": 0,
                "klass": "synthetic",
                "era": "recent",
            }
        )
        print(f"  {i + 1}/{n}", end="\r", flush=True)
    print()
    return out


# --------------------------------------------------------------------------- main
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("ngram", "anthropic"), default="ngram")
    ap.add_argument("--model", default="claude-sonnet-5", help="model id for --backend anthropic")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--tokens-per-doc", type=int, default=230)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--out", default="data/corpus/synthetic.jsonl")
    args = ap.parse_args(argv)

    if args.backend == "ngram":
        docs = generate_ngram(args.n, args.seed, args.tokens_per_doc)
    else:
        print(f"about to make {args.n} API calls to {args.model}", flush=True)
        docs = generate_anthropic(args.n, args.model, args.seed, args.max_tokens)

    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} documents to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
