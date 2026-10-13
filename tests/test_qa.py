from pdfqa.index import Index
from pdfqa.ingest import chunk_pdf
from pdfqa.qa import NOT_FOUND, SYSTEM_PROMPT, answer


class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        return self.reply


def test_extractive_answer_cites_page(sample_pdf):
    result = answer("How many vacation days do full-time employees get?", Index(chunk_pdf(sample_pdf)))
    assert "28 vacation days" in result.text
    assert result.mode == "extractive"
    assert any(s.citation.endswith("p. 1") for s in result.sources)


def test_unknown_question_is_not_answered(sample_pdf):
    result = answer("What is the CEO's favourite colour?", Index(chunk_pdf(sample_pdf)))
    assert result.text == NOT_FOUND
    assert not result.found


def test_llm_gets_numbered_excerpts_and_rules(sample_pdf):
    llm = FakeLLM("Hotels are reimbursed up to 120 euros per night [1].")
    result = answer("hotel cost limit", Index(chunk_pdf(sample_pdf)), llm)
    system, prompt = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "[1] (employee-handbook.pdf, p. 3)" in prompt
    assert [s.n for s in result.sources] == [1]
    assert result.mode == "fake"


def test_llm_not_found_drops_sources(sample_pdf):
    llm = FakeLLM(NOT_FOUND)
    result = answer("vacation", Index(chunk_pdf(sample_pdf)), llm)
    assert result.text == NOT_FOUND and result.sources == []


def test_extractive_skips_weak_matches(sample_pdf):
    result = answer("Can I work remotely from abroad?", Index(chunk_pdf(sample_pdf)))
    assert "abroad" in result.text
    assert "vacation" not in result.text.lower()
    assert not result.text.startswith("Remote Work")


def test_extractive_lists_only_cited_sources(sample_pdf):
    result = answer("vacation days full-time", Index(chunk_pdf(sample_pdf)))
    assert [s.citation for s in result.sources] == ["employee-handbook.pdf, p. 1"]
