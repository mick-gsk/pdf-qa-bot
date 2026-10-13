import pytest

from pdfqa.claims import extract, parse_number


def keys(text):
    return [(c.kind, c.key) for c in extract(text)]


@pytest.mark.parametrize("text, values", [
    ("120", [120]),
    ("1.8", [1.8]),
    ("1,8", [1.8]),
    ("1,234", [1234, 1.234]),  # English thousands or German decimal
    ("1.234,50", [1234.5]),
    ("1,234.50", [1234.5]),
    ("three", [3]),
    ("zwölf", [12]),
])
def test_parse_number(text, values):
    assert parse_number(text) == values


def test_units_are_normalised():
    assert keys("up to 120 euros, €160 or EUR 200") == [
        ("amount", "120 EUR"), ("amount", "160 EUR"), ("amount", "200 EUR")]
    assert keys("1,8 l and 1800 ml") == [("amount", "1.8 l"), ("amount", "1.8 l")]
    assert keys("8.2 kg") == [("amount", "8200 g")]
    assert keys("25 minutes") == keys("1500 seconds")


def test_number_words_count_only_with_a_unit():
    assert keys("three weeks") == [("amount", "3 week")]
    assert keys("one of the rules") == []


def test_number_word_glued_to_a_unit_letter_is_a_word():
    # "ones" is not "one s", "eins" is not "ein s"; "three-day" still is a duration
    assert keys("the ones listed, eins") == []
    assert keys("a three-day trip") == [("amount", "3 day")]


def test_one_descriptive_word_between_number_and_unit():
    assert keys("28 vacation days") == [("amount", "28 day")]
    assert keys("20 working days") == [("amount", "20 day")]


def test_ranges_produce_both_ends():
    assert keys("between 20 to 220 ml") == [("amount", "0.02 l"), ("amount", "0.22 l")]


@pytest.mark.parametrize("text", ["1 January 2026", "January 1, 2026", "2026-01-01", "01.01.2026",
                                  "1. Januar 2026", "1st of January 2026"])
def test_date_spellings(text):
    assert keys(text) == [("date", "2026-01-01")]


def test_date_without_year():
    assert keys("until 31 March") == [("date", "*-03-31")]


def test_times():
    assert keys("from 10:00 to 15:00") == [("time", "10:00"), ("time", "15:00")]
    assert keys("ab 9.30 Uhr") == [("time", "09:30")]
    assert keys("1.234,50 €") == [("amount", "1234.5 EUR")]  # not a time


def test_citations_and_list_numbers_are_ignored():
    assert keys("1. Fill the tank [2].\n2. Press start [3].") == []


def test_quotes_need_three_words():
    assert keys('He said "fine" and "cannot be reimbursed later".') == [
        ("quote", "cannot be reimbursed later")]


@pytest.mark.parametrize("text", ["120 million euros", "€120 million", "120 Mio. €", "120 Millionen Euro",
                                  "120.000.000 €"])
def test_magnitude_words_scale_the_amount(text):
    assert keys(text) == [("amount", "1.2e+08 EUR")]


def test_billion_and_milliarden():
    assert keys("3 billion dollars") == keys("3 Mrd. Dollar") == [("amount", "3e+09 USD")]


@pytest.mark.parametrize("text, key", [("-5%", "-5 %"), ("−5 %", "-5 %"), ("+5 %", "5 %"),
                                       ("balance: -120 €", "-120 EUR"), ("€-120", "-120 EUR")])
def test_signs_are_kept(text, key):
    assert keys(text) == [("amount", key)]


@pytest.mark.parametrize("text, key", [("9:30 AM", "09:30"), ("9:30 PM", "21:30"), ("2:30 pm", "14:30"),
                                       ("5 p.m.", "17:00"), ("12 a.m.", "00:00"),
                                       ("um 9:30 am Montag", "09:30")])  # German "am", not a.m.
def test_twelve_hour_times(text, key):
    assert keys(text) == [("time", key)]


def test_number_words_up_to_twenty_and_the_tens():
    assert keys("fourteen days, vierzehn Tage, seventy euros") == [
        ("amount", "14 day"), ("amount", "14 day"), ("amount", "70 EUR")]


def test_lowercase_currency_code():
    assert keys("usd 120") == [("amount", "120 USD")]


def test_leading_sign_belongs_to_the_first_end_of_a_range():
    assert [c.key for c in extract("-5 to 3%")] == ["-5 %", "3 %"]
