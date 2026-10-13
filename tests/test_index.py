from pdfqa.index import Index, _rrf, tokenize
from pdfqa.ingest import Chunk

CHUNKS = [
    Chunk("a.pdf", 1, "Employees receive 28 vacation days per year."),
    Chunk("a.pdf", 2, "Hotel costs are reimbursed up to 120 euros per night."),
    Chunk("b.pdf", 1, "The laptop is provided by the company."),
]


def test_tokenize_drops_stopwords_and_case():
    assert tokenize("How many Vacation days do I get?") == tokenize("vacation day")
    assert len(tokenize("How many Vacation days do I get?")) == 2


def test_tokenize_stems_word_forms():
    assert tokenize("descaling") == tokenize("Descale")
    assert tokenize("recommended settings") == tokenize("we recommend setting")


def test_bm25_finds_relevant_chunk():
    hits = Index(CHUNKS).search("vacation days")
    assert hits[0][0].page == 1 and hits[0][0].doc == "a.pdf"


def test_no_match_returns_nothing():
    assert Index(CHUNKS).search("quantum chromodynamics") == []


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "index.json"
    Index(CHUNKS).save(path)
    loaded = Index.load(path)
    assert len(loaded) == 3
    assert loaded.documents() == ["a.pdf", "b.pdf"]
    assert loaded.search("hotel")[0][0].page == 2


def test_hybrid_uses_embeddings_for_paraphrases():
    # Fake embedding: "accommodation" means the same as "hotel".
    def embed(texts):
        return [[1.0, 0.0] if ("hotel" in t.lower() or "accommodation" in t.lower()) else [0.0, 1.0] for t in texts]

    index = Index(CHUNKS, embed=embed)
    assert index.bm25("accommodation") == []
    assert index.search("accommodation")[0][0].page == 2


def test_rrf_rewards_agreement():
    fused = _rrf([[(0, 9.0), (1, 5.0)], [(1, 0.9), (0, 0.1)], [(1, 0.5)]])
    assert fused[0][0] == 1
