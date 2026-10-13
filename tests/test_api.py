from unittest import mock

from fastapi.testclient import TestClient

from pdfqa.api import create_app
from pdfqa.config import Settings


def client(tmp_path):
    return TestClient(create_app(Settings(data_dir=tmp_path)))


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
    c = client(tmp_path)
    assert c.post("/documents", files={"files": ("bad.pdf", b"not a pdf", "application/pdf")}).status_code == 415
    res = c.post("/documents", files={"files": ("bad.pdf", b"%PDF-1.7 garbage", "application/pdf")})
    assert res.status_code == 422
    assert not (tmp_path / "docs" / "bad.pdf").exists()


def test_ask_without_documents(tmp_path):
    assert client(tmp_path).post("/ask", json={"question": "hi"}).status_code == 400


def test_reset(tmp_path, sample_pdf):
    c = client(tmp_path)
    c.post("/documents", files={"files": ("h.pdf", sample_pdf.read_bytes(), "application/pdf")})
    assert c.delete("/documents").json() == {"documents": []}
    assert c.get("/health").json()["chunks"] == 0


def test_token_is_required_when_configured(tmp_path):
    c = TestClient(create_app(Settings(data_dir=tmp_path, api_token="s3cret")))
    assert c.get("/health").json()["auth"] is True
    assert c.get("/documents").status_code == 401
    assert c.get("/documents", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/documents", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_download_and_remove_single_document(tmp_path, sample_pdf):
    c = client(tmp_path)
    data = sample_pdf.read_bytes()
    c.post("/documents", files=[("files", ("a.pdf", data, "application/pdf")),
                                ("files", ("b.pdf", data, "application/pdf"))])
    res = c.get("/documents/a.pdf")
    assert res.status_code == 200 and res.content == data
    assert res.headers["content-type"] == "application/pdf"
    assert c.delete("/documents/a.pdf").json() == {"documents": ["b.pdf"]}
    assert c.get("/documents/a.pdf").status_code == 404
    assert c.get("/documents/..%2Fapi.py").status_code == 404
    assert c.post("/ask", json={"question": "vacation days"}).json()["sources"][0]["doc"] == "b.pdf"


def test_llm_failure_returns_502(tmp_path, sample_pdf):
    c = client(tmp_path)
    c.post("/documents", files={"files": ("h.pdf", sample_pdf.read_bytes(), "application/pdf")})
    with mock.patch("pdfqa.api.answer", side_effect=ConnectionError("refused")):
        res = c.post("/ask", json={"question": "vacation"})
    assert res.status_code == 502


def test_responses_carry_request_id(tmp_path):
    c = client(tmp_path)
    assert len(c.get("/health").headers["X-Request-ID"]) == 12
    assert c.get("/health", headers={"X-Request-ID": "abc"}).headers["X-Request-ID"] == "abc"


def test_non_ascii_token_header_is_rejected_not_an_error(tmp_path):
    c = TestClient(create_app(Settings(data_dir=tmp_path, api_token="s3cret")))
    assert c.get("/documents", headers={"Authorization": "Bearer pässwort".encode()}).status_code == 401


def test_download_of_non_latin_filename(tmp_path, sample_pdf):
    c = client(tmp_path)
    c.post("/documents", files={"files": ("合同.pdf", sample_pdf.read_bytes(), "application/pdf")})
    res = c.get("/documents/合同.pdf")
    assert res.status_code == 200
    assert res.headers["content-disposition"].startswith("inline;")


def test_unexpected_error_carries_request_id(tmp_path):
    c = TestClient(create_app(Settings(data_dir=tmp_path)), raise_server_exceptions=False)
    c.app.state.store.documents = mock.Mock(side_effect=RuntimeError("boom"))
    res = c.get("/documents", headers={"X-Request-ID": "abc"})
    assert res.status_code == 500
    assert res.json()["request_id"] == "abc" and res.headers["X-Request-ID"] == "abc"


def test_non_ascii_token_can_authenticate(tmp_path):
    c = TestClient(create_app(Settings(data_dir=tmp_path, api_token="pässwort")))
    assert c.get("/documents", headers={"Authorization": "Bearer pässwort".encode()}).status_code == 200
