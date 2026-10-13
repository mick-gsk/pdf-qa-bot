from pdfqa.config import Settings
from pdfqa.store import Store


def test_store_swaps_in_a_new_index_instead_of_changing_the_live_one(tmp_path, sample_pdf):
    store = Store(Settings(data_dir=tmp_path))
    store.add_pdf("a.pdf", sample_pdf.read_bytes())
    live = store.index  # what a running search holds
    store.add_pdf("b.pdf", sample_pdf.read_bytes())
    assert live.documents() == ["a.pdf"]
    live = store.index
    store.remove("a.pdf")
    assert live.documents() == ["a.pdf", "b.pdf"] and store.documents() == ["b.pdf"]


def test_open_file_stays_readable_after_remove(tmp_path, sample_pdf):
    store = Store(Settings(data_dir=tmp_path))
    store.add_pdf("a.pdf", sample_pdf.read_bytes())
    with store.open("a.pdf") as file:
        store.remove("a.pdf")
        assert file.read() == sample_pdf.read_bytes()
