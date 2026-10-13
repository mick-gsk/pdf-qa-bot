"""Check every number, date and quote in an answer against the excerpts it cites.

An LLM told to "cite every statement" still writes "up to 150 euros [1]" when excerpt [1]
says 120. The citation looks right, the number is wrong, and nobody notices. verify()
catches that:

    supported       the value is in a cited excerpt (or anywhere, for an uncited sentence)
    derived         not written in the excerpts, but the sum, difference, product or ratio of
                    two values that are ("40 euros more" from 160 and 120)
    wrong_citation  the value exists, but in a different excerpt than the one cited
    unit_mismatch   the number exists, but with another unit ("30 days" vs. "30 euros")
    unsupported     nothing in the excerpts backs it

Values are compared after normalisation (see claims.py), so "three weeks", "1,8 l" or
"1. Januar 2026" match their other spellings. Values that appear in the question are
accepted too: repeating the user's own number is not a hallucination.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from .claims import _MONTH_RE, Claim, extract, normalise_text

PROBLEMS = ("unsupported", "unit_mismatch", "wrong_citation")

_BOUNDARY_RE = re.compile(r"[.!?](?:\s*\[\d+\])*(?=\s|$)")
# a period after these does not end a sentence: "1. Januar", "e.g. 5", "ca. 20"
_NO_BREAK_RE = re.compile(r"(?:\b\d{1,2}|\b(?:e\.g|i\.e|z\.b|ca|approx|nr|no|p|pp|vs|etc|bzw|inkl|ggf))$",
                          re.IGNORECASE)
_MONTH_START_RE = re.compile(rf"\s+(?:{_MONTH_RE})\b", re.IGNORECASE)
_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Check:
    text: str
    kind: str
    start: int
    end: int
    status: str
    source: int | None = None  # 1-based excerpt number that backs the value
    evidence: str = ""  # the excerpt sentence it was found in, or how it was derived

    @property
    def ok(self) -> bool:
        return self.status not in PROBLEMS


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def problems(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def grounded(self) -> bool:
        return not self.problems

    def summary(self) -> dict[str, int]:
        counts = {status: 0 for status in ("supported", "derived", *PROBLEMS)}
        for check in self.checks:
            counts[check.status] += 1
        return counts


@dataclass
class _Source:
    text: str
    claims: list[Claim]
    plain: str  # normalised text, for quotes

    def evidence(self, claim: Claim) -> str:
        """The sentence around a claim, for showing where a value came from."""
        start = max(self.text.rfind(".", 0, claim.start), self.text.rfind("\n", 0, claim.start)) + 1
        end = min((i for i in (self.text.find(".", claim.end), self.text.find("\n", claim.end)) if i >= 0),
                  default=len(self.text))
        return self.text[start:end + 1].strip()


def verify(answer: str, sources: Sequence[str], question: str = "") -> Report:
    """Check the claims in `answer` against `sources`, which the answer cites as [1], [2], ..."""
    indexed = [_Source(s, extract(s), normalise_text(s)) for s in sources]
    asked = extract(question)
    report = Report()
    for sentence, cited in _sentences(answer):
        # several readings of one number ("1,234") share a span; one supported reading is enough.
        # Both ends of a range are checked separately.
        by_span: dict[tuple[int, int, int], list[Claim]] = {}
        for claim in extract(sentence.group(0)):
            by_span.setdefault((claim.start, claim.end, claim.part), []).append(claim)
        for readings in by_span.values():
            check = min((_check(c, indexed, cited, asked, sentence.group(0)) for c in readings), key=_rank)
            check.start += sentence.start()
            check.end += sentence.start()
            report.checks.append(check)
    return report


def repair_instructions(report: Report) -> str:
    """Feedback for one correction round with the model."""
    lines = []
    for c in report.problems:
        if c.status == "wrong_citation":
            lines.append(f'- "{c.text}" is in excerpt [{c.source}], not in the excerpt you cited')
        elif c.status == "unit_mismatch":
            lines.append(f'- "{c.text}": the excerpts give this number with a different unit ({c.evidence})')
        else:
            lines.append(f'- "{c.text}" does not appear in the cited excerpts')
    return ("Some values in your answer are not backed by the excerpts:\n" + "\n".join(lines) +
            "\n\nRewrite the answer. Take every number, date and quote exactly from the excerpts and "
            "cite the excerpt it comes from. Leave out anything the excerpts do not state. "
            "Do not mention this check.")


def strip_unsupported(answer: str, report: Report) -> str:
    """Remove the sentences that still contain a problem. Used when a repair round did not help."""
    bad = [(c.start, c.end) for c in report.problems]
    kept = [m.group(0).strip() for m, _ in _sentences(answer)
            if not any(m.start() <= s < m.end() for s, _ in bad)]
    return " ".join(k for k in kept if k)


# --- internals ----------------------------------------------------------------------------


@dataclass
class _Sentence:
    text: str
    begin: int

    def group(self, _: int = 0) -> str:
        return self.text

    def start(self) -> int:
        return self.begin

    def end(self) -> int:
        return self.begin + len(self.text)


def _sentences(answer: str) -> Iterator[tuple[_Sentence, set[int]]]:
    """Split into sentences, each with the excerpt numbers it cites. A citation right after the
    period ("... night. [1]") belongs to the sentence before it."""
    start = 0
    for m in _BOUNDARY_RE.finditer(answer):
        before = answer[start:m.start()]
        if before.strip().isdigit():
            continue  # list numbering: "1. Hotel costs ..."
        if _NO_BREAK_RE.search(before) and (not before[-1:].isdigit() or _MONTH_START_RE.match(answer, m.end())):
            continue
        yield from _sentence(answer, start, m.end())
        start = m.end()
    yield from _sentence(answer, start, len(answer))


def _sentence(answer: str, start: int, end: int) -> Iterator[tuple[_Sentence, set[int]]]:
    text = answer[start:end]
    if text.strip():
        yield _Sentence(text, start), {int(n) for n in _CITE_RE.findall(text)}


def _rank(check: Check) -> int:
    order = ("supported", "derived", "wrong_citation", "unit_mismatch", "unsupported")
    return order.index(check.status)


def _check(claim: Claim, sources: list[_Source], cited: set[int], question: list[Claim],
           sentence: str) -> Check:
    def result(status: str, n: int | None = None, evidence: str = "") -> Check:
        return Check(claim.text, claim.kind, claim.start, claim.end, status, n, evidence)

    if claim.key in {c.key for c in question}:
        return result("supported", None, "given in the question")
    numbered = list(enumerate(sources, start=1))
    in_scope = [(n, s) for n, s in numbered if not cited or n in cited]
    others = [(n, s) for n, s in numbered if cited and n not in cited]

    for scope, status in ((in_scope, "supported"), (others, "wrong_citation")):
        for n, source in scope:
            match = _find(claim, source, _words(sentence))
            if match is not None:
                return result(status, n, match)

    if claim.kind == "amount" and claim.value is not None:
        for n, source in in_scope:
            other_unit = next((c for c in source.claims if c.kind == "amount" and c.unit and claim.unit
                               and c.unit != claim.unit and _same(c.number, claim.number)), None)
            if other_unit:
                return result("unit_mismatch", n, other_unit.text)
        derived = _derive(claim, [c for _, s in in_scope for c in s.claims] + question)
        if derived and _CALCULATION_RE.search(sentence):
            return result("derived", None, derived)
    return result("unsupported")


def _find(claim: Claim, source: _Source, context: set[str]) -> str | None:
    if claim.kind == "quote":
        return claim.key if claim.key in source.plain else None
    for c in source.claims:
        if c.kind != claim.kind:
            continue
        if claim.kind == "date" and _same_date(c.key, claim.key):
            return source.evidence(c)
        if claim.kind == "time" and c.key == claim.key:
            return source.evidence(c)
        if claim.kind == "amount" and _same(c.value, claim.value) and c.unit == claim.unit:
            return source.evidence(c)
        if claim.kind == "amount" and claim.unit is None and _same(c.value, claim.value):
            # A bare number ("settings 4 to 6") matches a value with any unit, but only if the
            # source sentence shares a word with the answer sentence; otherwise "5" would be
            # backed by "5 minutes" anywhere in the excerpt.
            evidence = source.evidence(c)
            if context & _words(evidence):
                return evidence
    return None


def _words(text: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[^\W\d_]{4,}", text.lower())}


def _same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _same_date(a: str, b: str) -> bool:
    (ya, rest_a), (yb, rest_b) = a.split("-", 1), b.split("-", 1)
    return rest_a == rest_b and (ya == yb or "*" in (ya, yb))


_COUNT_UNITS = (None, "day", "week", "month", "year")
# A value only counts as derived if the sentence says it is a calculation. Otherwise "5 years"
# would pass as 2 years + 3 years when the excerpt only mentions those two warranties.
_CALCULATION_RE = re.compile(
    r"\b(?:more|less|fewer|difference|total|together|combined|sum|altogether|times|each|per|gives|"
    r"makes|comes to|adds up|mehr|weniger|insgesamt|zusammen|gesamt|differenz|jeweils|pro|ergibt|macht)\b",
    re.IGNORECASE)


def _derive(claim: Claim, facts: list[Claim]) -> str | None:
    """Explain the claim as one arithmetic step over two values from the excerpts, if possible.

    Only steps that make sense for the units are tried: sums and differences of two values
    with the claim's unit, a percentage change, or a value multiplied or divided by a count
    or a duration ("28 euros per day" for "3 days"). Bare numbers are never derived; too many
    would match by chance.
    """
    target, unit = claim.value, claim.unit
    if target is None or unit is None or not claim.number:
        return None
    # compare at the precision the answer wrote, in the unit it wrote: "251 ml" is not 0.25 l
    scale = target / claim.number
    decimals = len(re.split(r"[.,]", claim.text.split()[0])[-1]) if re.match(r"\d+[.,]\d+\b", claim.text) else 0
    values = {(c.value, c.unit, c.text) for c in facts if c.kind == "amount" and c.value is not None}
    for (a, ua, ta), (b, ub, tb) in itertools.permutations(values, 2):
        candidates = []
        if ua == ub == unit:
            candidates += [(a + b, f"{ta} + {tb}"), (a - b, f"{ta} − {tb}")]
        if unit == "%" and ua == ub and ua is not None and b:
            candidates.append(((a - b) / b * 100, f"({ta} − {tb}) / {tb}"))
        if ua == unit and ub in _COUNT_UNITS and b >= 2 and b == int(b):
            candidates += [(a * b, f"{ta} × {tb}"), (a / b, f"{ta} / {tb}")]
        for value, how in candidates:
            if value > 0 and round(value / scale, decimals) == round(claim.number, decimals):
                return how
    return None
