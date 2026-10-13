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


@pytest.mark.parametrize("cases", ["examples/eval.json", "examples/grounding_cases.json"])
def test_grounding_benchmark_has_no_false_alarms(cases, capsys):
    assert main(["bench-grounding", cases]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["false_alarm_rate"] == 0.0
    assert summary["detection_rate"] >= 0.85
