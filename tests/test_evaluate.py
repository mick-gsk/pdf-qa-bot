from pathlib import Path

from pdfqa.evaluate import Case, evaluate, run
from pdfqa.index import Index
from pdfqa.ingest import Chunk

CASES = Path(__file__).parent.parent / "examples" / "eval.json"


def test_counts_hits_answers_and_refusals():
    index = Index([
        Chunk("a.pdf", 1, "Employees receive 28 vacation days per year."),
        Chunk("a.pdf", 2, "Hotel costs are reimbursed up to 120 euros per night."),
    ])
    report = evaluate(index, [
        Case("How many vacation days?", "28 vacation days", "a.pdf", 1),
        Case("What is the hotel limit?", "120 euros", "a.pdf", 1),  # wrong page on purpose
        Case("Who is the CEO?", None),
    ])
    summary = report.summary()
    assert summary["questions"] == 3
    assert summary["hit@4"] == 0.5 and summary["mrr"] == 0.5
    assert summary["answer_accuracy"] == 1.0
    assert summary["correct_refusals"] == 1.0
    assert report.failures == []


def test_example_cases_run():
    summary = run(CASES).summary()
    assert summary["questions"] == 31
    assert summary["hit@4"] >= 0.9
