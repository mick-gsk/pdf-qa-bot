from fastapi.testclient import TestClient

from pdfqa.api import create_app


def client(tmp_path):
    return TestClient(create_app(tmp_path))


def test_health_reports_extractive_mode(tmp_path):
    body = client(tmp_path).get("/health").json()
    assert body["status"] == "ok" and body["llm"].startswith("none")


def test_upload_then_ask(tmp_path, sample_pdf):
    c = client(tmp_path)
    with sample_pdf.open("rb") as f:
        res = c.post("/documents", files={"files": ("employee-handbook.pdf", f, "application/pdf")})
    assert res.status_code == 201
    assert res.json()["documents"] == ["employee-handbook.pdf"]

    res = c.post("/ask", json={"question": "How many days can I work remotely per week?"})
    body = res.json()
    assert res.status_code == 200 and body["found"]
    assert "three days per week" in body["answer"]
    assert body["sources"][0]["citation"] == "employee-handbook.pdf, p. 2"


def test_index_survives_restart(tmp_path, sample_pdf):
    c = client(tmp_path)
    with sample_pdf.open("rb") as f:
        c.post("/documents", files={"files": ("employee-handbook.pdf", f, "application/pdf")})
    assert client(tmp_path).get("/documents").json()["documents"] == ["employee-handbook.pdf"]


def test_rejects_non_pdf_and_duplicates(tmp_path, sample_pdf):
    c = client(tmp_path)
    assert c.post("/documents", files={"files": ("notes.txt", b"hi", "text/plain")}).status_code == 415
    data = sample_pdf.read_bytes()
    assert c.post("/documents", files={"files": ("h.pdf", data, "application/pdf")}).status_code == 201
    assert c.post("/documents", files={"files": ("h.pdf", data, "application/pdf")}).status_code == 409


def test_broken_pdf_is_rejected_cleanly(tmp_path):
    res = client(tmp_path).post("/documents", files={"files": ("bad.pdf", b"not a pdf", "application/pdf")})
    assert res.status_code == 422
    assert not (tmp_path / "docs" / "bad.pdf").exists()


def test_ask_without_documents(tmp_path):
    assert client(tmp_path).post("/ask", json={"question": "hi"}).status_code == 400


def test_reset(tmp_path, sample_pdf):
    c = client(tmp_path)
    c.post("/documents", files={"files": ("h.pdf", sample_pdf.read_bytes(), "application/pdf")})
    assert c.delete("/documents").json() == {"documents": []}
    assert c.get("/health").json()["chunks"] == 0
