from pdfqa.ingest import chunk_page, chunk_pdf, read_pdf, split_sentences


def test_reads_every_page(sample_pdf):
    pages = read_pdf(sample_pdf)
    assert [p for p, _ in pages] == [1, 2, 3]
    assert "28 vacation days" in pages[0][1]


def test_chunks_keep_page_numbers(sample_pdf):
    chunks = chunk_pdf(sample_pdf)
    assert {c.page for c in chunks} == {1, 2, 3}
    assert all(c.doc == "employee-handbook.pdf" for c in chunks)
    assert chunks[0].citation.startswith("employee-handbook.pdf, p. 1")


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} has five words." for i in range(40))
    chunks = chunk_page("d.pdf", 1, text, max_words=30, overlap=1)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 36 for c in chunks)
    # last sentence of one chunk starts the next
    assert split_sentences(chunks[0].text)[-1] == split_sentences(chunks[1].text)[0]


def test_headings_are_separate_sentences():
    text = "Remote Work\nEmployees may work remotely three days per\nweek. Core hours apply."
    assert split_sentences(text) == ["Remote Work", "Employees may work remotely three days per week.", "Core hours apply."]
