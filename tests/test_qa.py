from pdfqa.index import Index
from pdfqa.ingest import chunk_pdf
from pdfqa.qa import NOT_FOUND, SYSTEM_PROMPT, answer, answer_stream


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


def test_extractive_refuses_when_only_common_words_match(sample_pdf):
    # "company" appears in the handbook, "car" does not
    assert answer("Is there a company car policy?", Index(chunk_pdf(sample_pdf))).text == NOT_FOUND


class StreamingLLM(FakeLLM):
    def stream(self, system, prompt):
        yield from ["Hotels: up to ", "120 euros ", "per night [1]."]


def test_stream_yields_pieces_then_final_answer(sample_pdf):
    events = list(answer_stream("hotel cost limit", Index(chunk_pdf(sample_pdf)), StreamingLLM("")))
    assert [v for k, v in events if k == "token"] == ["Hotels: up to ", "120 euros ", "per night [1]."]
    kind, final = events[-1]
    assert kind == "done" and final.text == "Hotels: up to 120 euros per night [1]."
    assert [s.n for s in final.sources] == [1]


def test_stream_not_found_is_normalised(sample_pdf):
    llm = StreamingLLM("")
    llm.stream = lambda system, prompt: iter(["I couldn't find this in the ", "uploaded documents."])
    *_, (kind, final) = answer_stream("vacation", Index(chunk_pdf(sample_pdf)), llm)
    assert final.text == NOT_FOUND and final.sources == []


class ScriptedLLM(FakeLLM):
    """Returns the given replies in order, one per call."""

    def __init__(self, *replies):
        super().__init__(None)
        self.replies = list(replies)

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        return self.replies.pop(0)


def test_invented_number_is_repaired(sample_pdf):
    llm = ScriptedLLM("Hotels are reimbursed up to 150 euros per night [1].",
                      "Hotels are reimbursed up to 120 euros per night [1].")
    result = answer("hotel cost limit", Index(chunk_pdf(sample_pdf)), llm)
    assert result.text == "Hotels are reimbursed up to 120 euros per night [1]."
    assert result.repaired and [c.status for c in result.checks] == ["supported"]
    assert '"150 euros" does not appear' in llm.calls[1][1]


def test_sentence_still_wrong_after_repair_is_dropped(sample_pdf):
    llm = ScriptedLLM("Up to 120 euros per night [1]. Taxis up to 45 euros [1].",
                      "Up to 120 euros per night [1]. Taxis are covered up to 45 euros [1].")
    result = answer("hotel cost limit", Index(chunk_pdf(sample_pdf)), llm)
    assert result.text == "Up to 120 euros per night [1]."
    assert result.repaired and all(c.ok for c in result.checks)


def test_flag_mode_reports_without_changing(sample_pdf):
    llm = ScriptedLLM("Hotels are reimbursed up to 150 euros per night [1].")
    result = answer("hotel cost limit", Index(chunk_pdf(sample_pdf)), llm, grounding="flag")
    assert "150 euros" in result.text and not result.repaired
    assert [c.status for c in result.checks] == ["unsupported"] and len(llm.calls) == 1


def test_grounded_answer_needs_one_call(sample_pdf):
    llm = ScriptedLLM("Hotels are reimbursed up to 120 euros per night [1].")
    result = answer("hotel cost limit", Index(chunk_pdf(sample_pdf)), llm)
    assert len(llm.calls) == 1 and not result.repaired
