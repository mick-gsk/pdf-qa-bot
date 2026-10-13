"""Measure the value check on answers with known mistakes.

    pdfqa bench-grounding examples/eval.json

For every question in the case file, the extractive answer (sentences copied from the
excerpts, so every value in it is correct) is checked once as is, then once per kind of
mistake planted into it:

    number    a value changed slightly (120 -> 121, 1.8 -> 2.0)
    unit      the unit swapped for one of another kind (120 euros -> 120 days)
    citation  the citation pointed at an excerpt that does not contain the value
    date      a date moved by one day

The report gives the share of planted mistakes that were caught and the share of correct
answers that were wrongly flagged. The mistakes are synthetic and simple; they measure
whether the check works as specified, not how often a particular model makes mistakes.

    pdfqa bench-grounding examples/grounding_cases.json

runs hand-written answers instead, phrased the way a model would phrase them ("1,8 litres",
"€160", "half an hour"), each marked as correct or not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .claims import Claim, extract
from .evaluate import load_cases
from .grounding import verify
from .index import Index
from .ingest import chunk_pdf
from .qa import answer, retrieve

_OTHER_UNIT = {"EUR": "days", "USD": "days", "%": "euros", "day": "euros", "week": "euros", "month": "euros",
               "year": "euros", "s": "euros", "l": "kg", "g": "liters", "m": "kg", "bar": "volts", "W": "kg",
               "V": "watts"}


@dataclass
class BenchReport:
    correct: int = 0
    false_alarms: int = 0
    planted: dict[str, int] = field(default_factory=dict)
    caught: dict[str, int] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "correct_answers": self.correct,
            "false_alarm_rate": round(self.false_alarms / (self.correct or 1), 3),
            "caught": {kind: f"{self.caught.get(kind, 0)}/{n}" for kind, n in self.planted.items()},
            "detection_rate": round(sum(self.caught.values()) / (sum(self.planted.values()) or 1), 3),
        }


def plant(text: str, kind: str, n_sources: int) -> str | None:
    """Return `text` with one mistake of the given kind, or None if it has nothing to change."""
    claims = [c for c in extract(text) if c.kind == ("date" if kind == "date" else "amount")]
    if kind == "citation":
        cited = re.search(r"\[(\d+)\]", text)
        if not cited or n_sources < 2 or not claims:
            return None
        other = 1 if cited.group(1) != "1" else 2
        return re.sub(r"\[\d+\]", f"[{other}]", text)
    claims = [c for c in claims if c.text[0].isdigit() or kind != "unit"]
    if not claims:
        return None
    claim = claims[0]
    if kind == "number":
        number = re.search(r"\d+(?:[.,]\d+)?", claim.text)
        if not number:
            return None
        old = number.group(0)
        new = str(int(old) + 1) if old.isdigit() else f"{float(old.replace(',', '.')) + 0.2:.1f}"
        return _replace(text, claim, claim.text.replace(old, new, 1))
    if kind == "unit":
        unit_text = claim.text[len(re.match(r"[\d.,\s]*", claim.text).group(0)):]  # type: ignore[union-attr]
        if not claim.unit or claim.unit not in _OTHER_UNIT or not unit_text or claim.text[0] in "€$£":
            return None
        return _replace(text, claim, claim.text.replace(unit_text, _OTHER_UNIT[claim.unit]))
    if kind == "date":
        day = re.search(r"\b(\d{1,2})\b", claim.text)
        if not day:
            return None
        moved = str(int(day.group(1)) % 28 + 1)
        return _replace(text, claim, claim.text[:day.start()] + moved + claim.text[day.end():])
    raise ValueError(kind)


def _replace(text: str, claim: Claim, new: str) -> str:
    return text[:claim.start] + new + text[claim.end:]


def run(case_file: str | Path, k: int = 4) -> BenchReport:
    if "answer" in json.loads(Path(case_file).read_text(encoding="utf-8"))["cases"][0]:
        return run_written(case_file, k)
    docs, cases = load_cases(case_file)
    index = Index()
    for doc in docs:
        index.add(chunk_pdf(doc))
    report = BenchReport()
    for case in cases:
        if case.expect is None:
            continue
        sources = [s.text for s in retrieve(case.question, index, k)]
        text = answer(case.question, index, None, k, grounding="off").text
        if not extract(text):
            continue
        report.correct += 1
        if not verify(text, sources, case.question).grounded:
            report.false_alarms += 1
            report.misses.append(f"false alarm: {text!r}")
        for kind in ("number", "unit", "citation", "date"):
            changed = plant(text, kind, len(sources))
            if changed is None or changed == text:
                continue
            report.planted[kind] = report.planted.get(kind, 0) + 1
            if verify(changed, sources, case.question).grounded:
                report.misses.append(f"missed {kind}: {changed!r}")
            else:
                report.caught[kind] = report.caught.get(kind, 0) + 1
    return report


def run_written(case_file: str | Path, k: int = 4) -> BenchReport:
    path = Path(case_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    index = Index()
    for name in data["documents"]:
        index.add(chunk_pdf(path.parent / name))
    report = BenchReport()
    for case in data["cases"]:
        # Excerpt [1] is the page the answer comes from, as if the model had cited it correctly;
        # the other excerpts are what retrieval returns for the question.
        page = [c.text for c in index.chunks if (c.doc, c.page) == (case["doc"], case["page"])]
        sources = ["\n".join(page)] + [s.text for s in retrieve(case["question"], index, k) if s.text not in page]
        grounded = verify(case["answer"], sources, case["question"]).grounded
        if case["ok"]:
            report.correct += 1
            if not grounded:
                report.false_alarms += 1
                report.misses.append(f"false alarm: {case['answer']!r}")
        else:
            report.planted["written"] = report.planted.get("written", 0) + 1
            if grounded:
                report.misses.append(f"missed: {case['answer']!r}")
            else:
                report.caught["written"] = report.caught.get("written", 0) + 1
    return report
