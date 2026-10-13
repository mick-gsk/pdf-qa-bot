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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .claims import Claim, extract, normalise_text

PROBLEMS = ("unsupported", "unit_mismatch", "wrong_citation")

# "120 Mio. euros" is one sentence: a magnitude abbreviation does not end it
_SENTENCE_RE = re.compile(
    r".+?(?:(?<!\bMio)(?<!\bMrd)(?<!\bTsd)(?<!\bmio)(?<!\bmrd)(?<!\btsd)[.!?](?:\s*\[\d+\])*(?=\s|$)|$)",
    re.DOTALL)
_CITE_RE = re.compile(r"\[(\d+)\]")
_CITE_GROUP_RE = re.compile(r"(?:\[\d+\])+")
# durations a per-day/week/... amount can be multiplied with; seconds-based units are not, since
# "8 hours" is stored as 28800 s
_PER = {"day", "week", "month", "year"}


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
    for sentence, _ in _sentences(answer):
        # several readings of one number ("1,234") share a span; one supported reading is enough.
        # Both ends of a range are checked separately.
        by_span: dict[tuple[int, int, int], list[Claim]] = {}
        for claim in extract(sentence.group(0)):
            by_span.setdefault((claim.start, claim.end, claim.part), []).append(claim)
        groups = list(_CITE_GROUP_RE.finditer(sentence.group(0)))
        for (start, end, _), readings in by_span.items():
            cited = _citation_for(start, end, groups, sentence.group(0))
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


def _citation_for(start: int, end: int, groups: list[re.Match[str]], text: str) -> set[int]:
    """The citation a value belongs to: the next one after it in the sentence, else the last
    one before it. "120 euros [1] and 28 days [2]" checks each value against its own excerpt.
    A citation earlier in the value's own clause beats the next one behind a semicolon:
    "according to [1], hotels cost 120 euros; according to [2], ..." is [1]."""
    after = next((g for g in groups if g.start() >= end), None)
    before = next((g for g in reversed(groups) if g.end() <= start), None)
    if after and before and ";" in text[end:after.start()] and ";" not in text[before.end():start]:
        after = None
    group = after or before
    return {int(n) for n in _CITE_RE.findall(group.group(0))} if group else set()


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
        derived = _derive(claim, _occurrences(s for _, s in in_scope))
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
                and (claim.unit is None or c.unit == claim.unit)):
            return source.evidence(c)
    return None


def _same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _same_date(source: str, claim: str) -> bool:
    """An answer may leave out the year, but not add one the source does not give."""
    (ys, rest_s), (yc, rest_c) = source.split("-", 1), claim.split("-", 1)
    return rest_s == rest_c and (ys == yc or yc == "*")


def _occurrences(sources: Iterable[_Source]) -> list[Claim]:
    """The values of the sources, each sentence once: overlapping chunks repeat a sentence."""
    seen, found = set(), []
    for source in sources:
        for c in source.claims:
            key = (source.evidence(c), c.text, c.value, c.part)
            if key not in seen:
                seen.add(key)
                found.append(c)
    return found


def _derive(claim: Claim, facts: list[Claim]) -> str | None:
    """Explain the claim as one arithmetic step over two values from the excerpts, if possible."""
    target = claim.value
    if target is None or (claim.unit is None and abs(target) < 10):
        return None  # small bare numbers are too easy to hit by accident
    # compare at the precision the answer wrote, with a decimal point or comma
    written = re.search(r"\d[.,](\d+)\b", claim.text)
    decimals = len(written.group(1)) if written else 0
    # every occurrence counts: two line items of 6 euros add up to 12 euros
    amounts = [(c.value, c.unit, c.text, c.start, c.part) for c in facts
               if c.kind == "amount" and c.value is not None]
    for (a, ua, ta, sa, pa), (b, ub, tb, sb, pb) in itertools.permutations(amounts, 2):
        if (sa, pa, ta) == (sb, pb, tb) and a != b:
            continue  # two readings of the same number ("1,234")
        same_unit = ua == ub
        candidates = []
        if same_unit and claim.unit in (None, ua):
            candidates += [(a + b, f"{ta} + {tb}"), (a - b, f"{ta} − {tb}")]
        if claim.unit == "%" and same_unit and b:
            candidates.append(((a - b) / b * 100, f"({ta} − {tb}) / {tb}"))
        product, quotient = _product_unit(ua, ub), _quotient_unit(ua, ub)
        if product != _INVALID and claim.unit in (None, product):
            candidates.append((a * b, f"{ta} × {tb}"))
        if b and quotient != _INVALID and claim.unit in (None, quotient):
            candidates.append((a / b, f"{ta} / {tb}"))
        for value, how in candidates:
            if round(value, decimals) == round(target, decimals):
                return how
    return None


_INVALID = "invalid"


def _product_unit(ua: str | None, ub: str | None) -> str | None:
    """Unit of a × b: a count or a per-day/week duration times an amount. 160 euros × 28 euros
    is not an amount of euros."""
    if ua is None or ub is None:
        return ua or ub
    if ub in _PER and ua not in _PER:
        return ua
    if ua in _PER and ub not in _PER:
        return ub
    return _INVALID


def _quotient_unit(ua: str | None, ub: str | None) -> str | None:
    """Unit of a / b: same units give a ratio, dividing by a count or duration keeps the unit."""
    if ua == ub:
        return None
    if ub is None or ub in _PER:
        return ua
    return _INVALID
