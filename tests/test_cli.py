from pdfqa.cli import main


def test_index_and_ask(tmp_path, sample_pdf, capsys):
    index = tmp_path / "index.json"
    assert main(["--index", str(index), "index", str(sample_pdf)]) == 0
    assert index.exists()
    assert main(["--index", str(index), "ask", "How many vacation days?", "-k", "2"]) == 0
    assert "[1]" in capsys.readouterr().out
