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


def test_vectors_from_another_embedding_model_are_rebuilt(tmp_path):
    def old_model(texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def new_model(texts):
        return [[1.0, 0.0] if "hotel" in t.lower() else [0.0, 1.0] for t in texts]

    path = tmp_path / "index.json"
    Index(CHUNKS, embed=old_model).save(path)
    loaded = Index.load(path, embed=new_model)
    assert loaded.vector("hotel")[0][0] == 1
    assert len(loaded._vectors[0]) == 2


def test_vectors_of_a_model_with_the_same_dimension_are_rebuilt_and_saved(tmp_path):
    calls = []

    def model(name):
        def embed(texts):
            calls.append(name)
            return [[1.0, 0.0] for _ in texts]
        embed.model = name
        return embed

    path = tmp_path / "index.json"
    Index(CHUNKS, embed=model("a")).save(path)
    Index.load(path, embed=model("b"))
    assert calls == ["a", "b"]
    Index.load(path, embed=model("b"))  # saved after the rebuild: no second rebuild
    assert calls == ["a", "b"]


def test_rrf_rewards_agreement():
    fused = _rrf([[(0, 9.0), (1, 5.0)], [(1, 0.9), (0, 0.1)], [(1, 0.5)]])
    assert fused[0][0] == 1


def test_remove_document_keeps_vectors_of_the_rest():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    index = Index(CHUNKS, embed=embed)
    assert index.remove("a.pdf") == 2
    assert index.documents() == ["b.pdf"] and len(calls) == 1
    assert index._vectors == [[float(len(CHUNKS[2].text)), 1.0]]
    assert index.search("laptop")[0][0].doc == "b.pdf"
    assert index.remove("missing.pdf") == 0


def test_inflected_stopwords_are_dropped_like_their_base_form():
    assert tokenize("What are employees getting?") == tokenize("Employees get") == ["employe"]
