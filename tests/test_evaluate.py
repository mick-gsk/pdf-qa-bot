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
        Case("How much are hotel costs reimbursed?", "120 euros", "a.pdf", 1),  # wrong page on purpose
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


def test_each_question_is_retrieved_once():
    index = Index([Chunk("a.pdf", 1, "Employees receive 28 vacation days per year.")])
    calls = []
    search = index.search
    index.search = lambda *a, **kw: calls.append(a) or search(*a, **kw)  # type: ignore[method-assign]
    evaluate(index, [Case("How many vacation days?", "28 vacation days", "a.pdf", 1)])
    assert len(calls) == 1


def test_missing_question_type_has_no_rate():
    index = Index([Chunk("a.pdf", 1, "Employees receive 28 vacation days per year.")])
    summary = evaluate(index, [Case("How many vacation days?", "28 vacation days", "a.pdf", 1)]).summary()
    assert summary["correct_refusals"] is None
    assert summary["answer_accuracy"] == 1.0


def test_case_document_may_be_given_with_its_folder():
    index = Index([Chunk("hr.pdf", 3, "Employees receive 28 vacation days per year.")])
    case = Case("How many vacation days?", "28 vacation days", "policies/hr.pdf", 3)
    assert evaluate(index, [case]).summary()["hit@4"] == 1.0
