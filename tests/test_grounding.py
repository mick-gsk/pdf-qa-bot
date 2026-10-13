from pdfqa.grounding import repair_instructions, strip_unsupported, verify

SOURCES = [
    "Hotel costs are reimbursed up to 120 euros per night, or up to 160 euros per night in Munich, "
    "Paris and London. Meals are covered by a daily allowance of 28 euros.",
    "Full-time employees receive 28 vacation days per calendar year. The rules apply from 1 January 2026.",
]


def statuses(answer, question=""):
    return [(c.text, c.status) for c in verify(answer, SOURCES, question).checks]


def test_supported_values_with_other_spellings():
    report = verify("Up to €120 per night [1]. The rules apply from 01.01.2026 [2].", SOURCES)
    assert report.grounded
    assert [(c.source, c.status) for c in report.checks] == [(1, "supported"), (2, "supported")]
    assert report.checks[0].evidence.startswith("Hotel costs are reimbursed up to 120 euros")


def test_invented_number_is_unsupported():
    report = verify("Hotels are reimbursed up to 150 euros per night [1].", SOURCES)
    assert not report.grounded
    assert statuses("Hotels are reimbursed up to 150 euros per night [1].") == [("150 euros", "unsupported")]


def test_value_from_the_wrong_excerpt():
    assert statuses("You get 28 vacation days [1].") == [("28 vacation days", "wrong_citation")]


def test_unit_mismatch():
    assert statuses("The meal allowance is 28 hours [1].") == [("28 hours", "unit_mismatch")]


def test_wrong_citation_beats_unit_mismatch():
    # "28 days" is in excerpt [2]; excerpt [1] only has 28 euros
    assert statuses("The meal allowance is 28 days [1].") == [("28 days", "wrong_citation")]


def test_derived_values():
    report = verify("In Paris you get 40 euros more than elsewhere [1].", SOURCES)
    assert report.checks[0].status == "derived"
    assert report.checks[0].evidence == "160 euros − 120 euros"
    assert report.grounded


def test_small_bare_numbers_are_not_derived():
    assert statuses("There are 4 rules [1].") == [("4", "unsupported")]


def test_numbers_from_the_question_are_accepted():
    assert statuses("Yes, 3 nights in Paris cost up to 480 euros in total [1].", "Are 3 nights in Paris ok?") == [
        ("3", "supported"), ("480 euros", "derived")]  # 3 × 160 euros


def test_uncited_sentence_may_use_any_excerpt():
    assert statuses("The yearly allowance is 28 vacation days.") == [("28 vacation days", "supported")]


def test_quotes():
    assert statuses('The handbook says "reimbursed up to 120 euros per night" [1].') == [
        ('"reimbursed up to 120 euros per night"', "supported")]
    assert statuses('It says "up to 200 euros in Berlin" [1].')[0][1] == "unsupported"


def test_one_supported_reading_of_an_ambiguous_number_is_enough():
    assert statuses("Budget: 1,234 [1].") == [("1,234", "unsupported")]
    assert verify("Budget: 1,234 [1].", ["The budget is 1234 euros."]).grounded


def test_repair_feedback_and_stripping():
    answer = "Hotels: up to 120 euros [1]. In Paris up to 180 euros [1]. Meals: 28 euros [1]."
    report = verify(answer, SOURCES)
    assert [c.text for c in report.problems] == ["180 euros"]
    assert '"180 euros" does not appear' in repair_instructions(report)
    assert strip_unsupported(answer, report) == "Hotels: up to 120 euros [1]. Meals: 28 euros [1]."


def test_summary_counts():
    report = verify("Up to 120 euros [1], in Paris 40 euros more [1], and 99 euros for taxis [1].", SOURCES)
    assert report.summary() == {"supported": 1, "derived": 1, "unsupported": 1,
                                "unit_mismatch": 0, "wrong_citation": 0}


def test_derived_needs_a_calculation_word():
    sources = ["The warranty is 2 years. Registering extends it to 3 years."]
    assert verify("The warranty is 5 years [1].", sources).checks[0].status == "unsupported"
    assert verify("Together that makes 5 years [1].", sources).checks[0].status == "derived"
