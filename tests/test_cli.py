import json

import pytest

from pdfqa.cli import main

CASES = "examples/eval.json"


def test_index_then_ask(tmp_path, sample_pdf, capsys):
    index = str(tmp_path / "index.json")
    assert main(["--index", index, "index", str(sample_pdf)]) == 0
    assert "employee-handbook.pdf: 3 chunks" in capsys.readouterr().out
    assert main(["--index", index, "ask", "What is the hotel limit in Paris?"]) == 0
    out = capsys.readouterr().out
    assert "160 euros" in out and "[1] employee-handbook.pdf, p. 3" in out


def test_ask_with_empty_index(tmp_path, capsys):
    assert main(["--index", str(tmp_path / "none.json"), "ask", "anything"]) == 1
    assert "Index is empty" in capsys.readouterr().err


def test_eval_prints_metrics(capsys):
    assert main(["eval", CASES]) == 0
    assert json.loads(capsys.readouterr().out)["questions"] == 31


def test_bad_config_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "antropic")
    with pytest.raises(SystemExit):
        main(["ask", "x"])
    assert "LLM_PROVIDER" in capsys.readouterr().err


def test_pdfs_indexed_on_the_command_line_can_be_downloaded(tmp_path, sample_pdf, monkeypatch):
    from fastapi.testclient import TestClient

    from pdfqa.api import create_app
    from pdfqa.config import Settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert main(["index", str(sample_pdf)]) == 0
    res = TestClient(create_app(Settings(data_dir=tmp_path))).get(f"/documents/{sample_pdf.name}")
    assert res.status_code == 200 and res.content == sample_pdf.read_bytes()
