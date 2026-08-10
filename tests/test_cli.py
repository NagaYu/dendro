"""The command line must work end to end, offline, and exit meaningfully.

Exit codes are part of the contract (0 bound, 3 abstain, 4 flagged) so that
``dendro date`` composes in a shell script without anybody parsing prose.  The
disclaimer going to stderr on every path is also asserted -- it is the one piece
of output that must not be lost when a user pipes stdout into a file.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from dendro.cli import EXIT_ABSTAIN, EXIT_FLAGGED, EXIT_OK, main


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path, fixture_cache_dir):
    monkeypatch.setenv("DENDRO_OFFLINE", "1")
    monkeypatch.setenv("DENDRO_CACHE", str(tmp_path / "cache"))


def test_sources_lists_operators_and_forgeability(capsys):
    assert main(["sources"]) == EXIT_OK
    out = capsys.readouterr().out
    for expected in ("wayback", "internet-archive", "commoncrawl", "arxiv", "self_asserted"):
        assert expected in out
    assert "claim only" in out, "self-asserted source not marked as non-evidence"


def test_cache_reports_both_layers(capsys):
    assert main(["cache"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "writable" in out and "fixtures" in out


def test_date_a_known_old_url(capsys, monkeypatch, fixture_cache_dir, tmp_path):
    """A real cached Wayback record, dated offline, exits 0."""
    monkeypatch.setenv("DENDRO_CACHE", str(fixture_cache_dir))
    code = main(["date", "https://www.python.org/dev/peps/pep-0020/", "--offline", "--no-coverage"])
    captured = capsys.readouterr()
    assert code == EXIT_OK, captured.out
    assert "existence bound" in captured.out
    assert "internet-archive" in captured.out
    assert "not an authorship test" in captured.err


def test_date_json_output_is_machine_readable(capsys, monkeypatch, fixture_cache_dir):
    monkeypatch.setenv("DENDRO_CACHE", str(fixture_cache_dir))
    main(["date", "https://arxiv.org/abs/1706.03762", "--offline", "--no-coverage", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["not_after_year"] == 2017
    assert 0.0 <= payload["human_origin_p"] <= 1.0
    assert "explanation" in payload


def test_date_unknown_document_abstains(capsys, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("A short note about nothing in particular, written just now.", encoding="utf-8")
    code = main(["date", str(doc), "--offline", "--no-coverage"])
    out = capsys.readouterr().out
    assert code == EXIT_ABSTAIN
    assert "ABSTAIN" in out
    assert "not evidence of synthetic origin" in out or "absence of archival evidence" in out


def test_fingerprint_shows_reflow_invariance(capsys, tmp_path):
    prose = tmp_path / "a.txt"
    html = tmp_path / "a.html"
    body = ("The registry recorded 1,247 excess deaths in March 2019.\n"
            "Researchers compared the figures with prior baselines.\n"
            "The delay accounted for roughly 18 percent of the gap.")
    prose.write_text(body, encoding="utf-8")
    html.write_text(
        f"<html><head><title>x</title></head><body><nav>Home | About</nav>"
        f"<p>{body}</p><footer>&copy; 2019</footer></body></html>",
        encoding="utf-8",
    )
    main(["fingerprint", str(prose)])
    a = capsys.readouterr().out
    main(["fingerprint", str(html)])
    b = capsys.readouterr().out

    def sha(text):
        return next(l.split(":")[1].strip() for l in text.splitlines() if l.startswith("normalized sha256"))

    assert sha(a) == sha(b), "CLI fingerprints of the same prose diverged"


def test_report_on_a_local_jsonl(capsys, tmp_path, real_documents):
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"doc_id": d["doc_id"], "text": d["text"], "url": d["url"], "published": d["published"]}
        for d in real_documents[:8]
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    out_path = tmp_path / "verdicts.jsonl"
    code = main(["report", str(path), "--offline", "--limit", "8", "--out", str(out_path)])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "low-background" in out
    assert "purity / retention trade-off" in out
    written = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(written) == 8
    assert "human_origin_p" in written[0]
    assert "is_synthetic" not in written[0], "the CLI emitted a binary label"


def _first_json_object(text: str) -> dict:
    obj, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    return obj


@pytest.mark.parametrize("limit", [0.2, 0.5])
def test_subset_respects_the_constraint(capsys, tmp_path, real_documents, limit):
    """The constraint must hold at every setting -- including by selecting nothing.

    Offline these documents have no cached witnesses, so they sit near the prior
    and a 20% purity budget correctly admits none of them.  Selecting zero rather
    than relaxing the constraint is the right behaviour, and pinning it stops a
    future "helpful" fallback from quietly loosening a published guarantee.
    """
    path = tmp_path / "corpus.jsonl"
    rows = [{"doc_id": d["doc_id"], "text": d["text"], "url": d["url"]} for d in real_documents[:10]]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out_path = tmp_path / "clean.jsonl"
    code = main(["subset", str(path), "--offline", "--max-synthetic", str(limit), "--out", str(out_path)])
    payload = _first_json_object(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["expected_synthetic_fraction"] <= limit + 1e-9
    assert payload["n_total"] == 10


def test_archive_layer_lets_a_paraphrase_inherit_a_date(capsys, tmp_path, real_documents):
    """The CLI path for the paraphrase defence, end to end."""
    from benchmarks.generators import Paraphraser

    archive = tmp_path / "archive.jsonl"
    archive.write_text(
        "\n".join(
            json.dumps({"doc_id": d["doc_id"], "text": d["text"], "not_after": d["published"],
                        "url": d["url"]})
            for d in real_documents[:60]
        ),
        encoding="utf-8",
    )
    para = tmp_path / "rewrite.txt"
    para.write_text(Paraphraser(strength=0.55, seed=1).paraphrase(real_documents[0]["text"]),
                    encoding="utf-8")

    main(["date", str(para), "--offline", "--no-coverage", "--archive", str(archive), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ancestor"] == real_documents[0]["doc_id"], payload
    assert payload["not_after_year"] <= 2020


def test_unknown_source_name_is_reported_not_crashed(capsys):
    """Bad input exits non-zero with a message naming the valid options."""
    from dendro.cli import EXIT_ERROR

    code = main(["date", "https://example.org/x", "--sources", "not-a-real-source"])
    assert code == EXIT_ERROR
    err = capsys.readouterr().err
    assert "unknown source" in err and "wayback" in err
