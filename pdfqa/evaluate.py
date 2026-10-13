"""Measure retrieval and answer quality against a set of questions with known answers.

    pdfqa eval examples/eval.json

The case file lists the PDFs (relative to the file) and the questions. Each question
names the page the answer is on and a phrase the answer must contain. Questions with
"expect": null have no answer in the documents; the bot should say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .index import EmbedFn, Index
from .ingest import chunk_pdf
from .llm import LLM
from .qa import answer


@dataclass
class Case:
    question: str
    expect: str | None
    doc: str | None = None
    page: int | None = None


@dataclass
class Report:
    mode: str
    k: int
    answerable: int = 0
    unanswerable: int = 0
    hits: int = 0  # right page among the top k excerpts
    reciprocal_ranks: float = 0.0
    correct: int = 0  # answer contains the expected phrase
    refused: int = 0  # said "not found" for an unanswerable question
    failures: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, float | int | str | None]:
        """Rates per question type; None where the case file has no question of that type."""
        a, u = self.answerable, self.unanswerable
        return {
            "mode": self.mode,
            "questions": a + u,
            f"hit@{self.k}": round(self.hits / a, 3) if a else None,
            "mrr": round(self.reciprocal_ranks / a, 3) if a else None,
            "answer_accuracy": round(self.correct / a, 3) if a else None,
            "correct_refusals": round(self.refused / u, 3) if u else None,
        }


def load_cases(path: str | Path) -> tuple[list[Path], list[Case]]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = [path.parent / name for name in data["documents"]]
    return docs, [Case(**c) for c in data["cases"]]


def evaluate(index: Index, cases: list[Case], llm: LLM | None = None, k: int = 4) -> Report:
    report = Report(mode=llm.name if llm else "extractive", k=k)
    for case in cases:
        hits = index.search(case.question, k=k)
        result = answer(case.question, index, llm, k=k, hits=hits)
        if case.expect is None:
            report.unanswerable += 1
            if result.found:
                report.failures.append(f"answered an unanswerable question: {case.question!r} -> {result.text!r}")
            else:
                report.refused += 1
            continue

        report.answerable += 1
        pages = [(c.doc, c.page) for c, _ in hits]
        if (case.doc, case.page) in pages:
            report.hits += 1
            report.reciprocal_ranks += 1 / (pages.index((case.doc, case.page)) + 1)
        if case.expect.lower() in result.text.lower():
            report.correct += 1
        else:
            report.failures.append(f"{case.question!r}: expected {case.expect!r}, got {result.text!r}")
    return report


def run(case_file: str | Path, llm: LLM | None = None, embed: EmbedFn | None = None, k: int = 4,
        chunk_words: int = 120) -> Report:
    docs, cases = load_cases(case_file)
    index = Index(embed=embed)
    for doc in docs:
        index.add(chunk_pdf(doc, max_words=chunk_words))
    return evaluate(index, cases, llm, k=k)
