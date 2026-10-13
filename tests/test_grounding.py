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


def test_numbered_list_items_are_not_values():
    answer = "1. Hotels: up to 120 euros per night [1].\n2. Meals: 28 euros per day [1]."
    assert statuses(answer) == [("120 euros", "supported"), ("28 euros", "supported")]


def test_each_value_is_checked_against_its_own_citation():
    # 28 vacation days is in [2], 120 euros in [1]; pooling both citations would hide the swap
    assert statuses("Hotels cost up to 120 euros [2] and you get 28 vacation days [1].") == [
        ("120 euros", "wrong_citation"), ("28 vacation days", "wrong_citation")]
    assert statuses("Hotels cost up to 120 euros [1] and you get 28 vacation days [2].") == [
        ("120 euros", "supported"), ("28 vacation days", "supported")]


def test_value_after_its_citation_uses_the_one_before():
    assert statuses("According to [2], you get 28 vacation days.") == [("28 vacation days", "supported")]


def test_amounts_of_the_same_unit_are_not_multiplied():
    # 160 euros x 28 euros is not a meaningful amount of euros
    assert statuses("Paris costs 4480 euros [1].") == [("4480 euros", "unsupported")]
    assert statuses("Paris costs 4480 [1].") == [("4480", "unsupported")]
    # a count times an amount is
    assert verify("Two nights cost 240 euros in total.", ["Hotels: up to 120 euros per night, 2 nights."]).grounded


def test_answer_may_not_add_a_year_the_source_does_not_give():
    sources = ["The office closes on 24 December."]
    assert [c.status for c in verify("It closes on 24 December 2025 [1].", sources).checks] == ["unsupported"]
    assert verify("It closes on 24 December [1].", sources).grounded
    # the other way round is fine: the answer leaves out a year the source gives
    assert verify("The rules apply from 1 January [2].", SOURCES).grounded


def test_citation_after_an_abbreviation_ends_the_sentence():
    answer = "Hotels up to 150 euros, taxis etc. [1] You get 28 vacation days [2]."
    assert strip_unsupported(answer, verify(answer, SOURCES)) == "You get 28 vacation days [2]."


def test_both_ends_of_a_range_are_checked():
    checks = verify("Use 20-220 ml [1].", ["Use 20 ml of water."]).checks
    assert [c.status for c in checks] == ["supported", "unsupported"]


def test_answer_may_not_add_a_unit_to_a_bare_source_number():
    assert [c.status for c in verify("It costs 120 euros [1].", ["The limit is 120."]).checks] == ["unsupported"]


def test_decimal_comma_keeps_its_precision_when_deriving():
    assert [c.status for c in verify("Together 1,4 euros [1].", ["1 euro and 0,1 euros."]).checks] == [
        "unsupported"]


def test_rate_word_is_no_sign_of_a_calculation():
    sources = ["Hotel costs are reimbursed up to 120 euros per night, or up to 160 euros per night in Paris."]
    assert [c.status for c in verify("Hotels cost 280 euros per night [1].", sources).checks] == ["unsupported"]
    assert [c.status for c in verify("Together that is 280 euros [1].", sources).checks] == ["derived"]


def test_sign_before_the_currency_is_kept():
    assert [c.status for c in verify("The balance is -€120 [1].", ["The balance is €120."]).checks] == ["unsupported"]


def test_composed_number_words_are_one_number():
    assert [c.status for c in verify("You get one hundred days [1].", ["You get one day."]).checks] == ["unsupported"]


def test_equal_line_items_can_be_added():
    sources = ["Shipping for the cable: 6 euros. Shipping for the lamp: 6 euros."]
    assert [c.status for c in verify("Shipping is 12 euros in total [1].", sources).checks] == ["derived"]


def test_overlapping_chunks_do_not_count_a_value_twice():
    sources = ["Shipping costs 6 euros.", "Shipping costs 6 euros. Returns are free."]
    assert [c.status for c in verify("Shipping is 12 euros in total [1][2].", sources).checks] == ["unsupported"]


def test_bare_number_needs_the_same_counted_thing_in_the_excerpt():
    sources = ["There are 3 meetings this week."]
    assert [c.status for c in verify("There are 3 blockers [1].", sources).checks] == ["unsupported"]
    assert verify("There are 3 meetings [1].", sources).grounded


def test_each_bare_number_is_bound_to_its_own_noun():
    checks = verify("There are 3 meetings and 3 blockers [1].", ["There are 3 meetings and 7 blockers."]).checks
    assert [c.status for c in checks] == ["supported", "unsupported"]


def test_negative_totals_can_be_derived():
    sources = ["Fee A: -€5. Fee B: -€7."]
    assert [c.status for c in verify("The fees are -€12 in total [1].", sources).checks] == ["derived"]
