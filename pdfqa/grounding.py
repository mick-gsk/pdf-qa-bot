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
from collections.abc import Sequence
from dataclasses import dataclass, field

from .claims import Claim, extract, normalise_text

PROBLEMS = ("unsupported", "unit_mismatch", "wrong_citation")

_SENTENCE_RE = re.compile(r".+?(?:[.!?](?:\s*\[\d+\])*(?=\s|$)|$)", re.DOTALL)
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
    known = {c.key for c in extract(question)}
    report = Report()
    for sentence, cited in _sentences(answer):
        # several readings of one number ("1,234") share a span; one supported reading is enough
        by_span: dict[tuple[int, int], list[Claim]] = {}
        for claim in extract(sentence.group(0)):
            by_span.setdefault((claim.start, claim.end), []).append(claim)
        for readings in by_span.values():
            check = min((_check(c, indexed, cited, known) for c in readings), key=_rank)
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


def _sentences(answer: str):
    for m in _SENTENCE_RE.finditer(answer):
        if m.group(0).strip():
            yield m, {int(n) for n in _CITE_RE.findall(m.group(0))}


def _rank(check: Check) -> int:
    order = ("supported", "derived", "wrong_citation", "unit_mismatch", "unsupported")
    return order.index(check.status)


def _check(claim: Claim, sources: list[_Source], cited: set[int], known: set[str]) -> Check:
    def result(status: str, n: int | None = None, evidence: str = "") -> Check:
        return Check(claim.text, claim.kind, claim.start, claim.end, status, n, evidence)

    if claim.key in known:
        return result("supported", None, "given in the question")
    numbered = list(enumerate(sources, start=1))
    in_scope = [(n, s) for n, s in numbered if not cited or n in cited]
    others = [(n, s) for n, s in numbered if cited and n not in cited]

    for scope, status in ((in_scope, "supported"), (others, "wrong_citation")):
        for n, source in scope:
            match = _find(claim, source)
            if match is not None:
                return result(status, n, match)

    if claim.kind == "amount" and claim.value is not None:
        for n, source in in_scope:
            other_unit = next((c for c in source.claims if c.kind == "amount" and c.unit and claim.unit
                               and c.unit != claim.unit and _same(c.number, claim.number)), None)
            if other_unit:
                return result("unit_mismatch", n, other_unit.text)
        derived = _derive(claim, [c for _, s in in_scope for c in s.claims])
        if derived:
            return result("derived", None, derived)
    return result("unsupported")


def _find(claim: Claim, source: _Source) -> str | None:
    if claim.kind == "quote":
        return claim.key if claim.key in source.plain else None
    for c in source.claims:
        if c.kind != claim.kind:
            continue
        if claim.kind == "date" and _same_date(c.key, claim.key):
            return source.evidence(c)
        if claim.kind == "time" and c.key == claim.key:
            return source.evidence(c)
        if (claim.kind == "amount" and _same(c.value, claim.value)
                and (claim.unit is None or c.unit in (None, claim.unit))):
            return source.evidence(c)
    return None


def _same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _same_date(a: str, b: str) -> bool:
    (ya, rest_a), (yb, rest_b) = a.split("-", 1), b.split("-", 1)
    return rest_a == rest_b and (ya == yb or "*" in (ya, yb))


def _derive(claim: Claim, facts: list[Claim]) -> str | None:
    """Explain the claim as one arithmetic step over two values from the excerpts, if possible."""
    target = claim.value
    if target is None or (claim.unit is None and abs(target) < 10):
        return None  # small bare numbers are too easy to hit by accident
    decimals = len(claim.text.split(".")[-1]) if re.search(r"\d\.\d+\b", claim.text) else 0
    values = {(c.value, c.unit, c.text) for c in facts if c.kind == "amount" and c.value is not None}
    for (a, ua, ta), (b, ub, tb) in itertools.permutations(values, 2):
        same_unit = ua == ub
        candidates = []
        if same_unit and claim.unit in (None, ua):
            candidates += [(a + b, f"{ta} + {tb}"), (a - b, f"{ta} − {tb}")]
        if claim.unit == "%" and same_unit and b:
            candidates.append(((a - b) / b * 100, f"({ta} − {tb}) / {tb}"))
        if claim.unit in (None, ua, ub):
            candidates.append((a * b, f"{ta} × {tb}"))
            if b:
                candidates.append((a / b, f"{ta} / {tb}"))
        for value, how in candidates:
            if round(value, decimals) == round(target, decimals) and value > 0:
                return how
    return None
