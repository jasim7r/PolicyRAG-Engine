"""HTTP tests using FastAPI's TestClient.  Needs: pip install -r requirements-dev.txt"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")          # force retrieval-only mode: no network
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[name]
    from app.main import app
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm"] == "retrieval-only"


def test_upload_list_ask_delete(client):
    files = {"file": ("faq.txt", b"Refunds are processed within 7 business days of approval.", "text/plain")}
    assert client.post("/documents/upload", files=files).status_code == 201
    assert any(d["name"] == "faq.txt" for d in client.get("/documents").json())

    answer = client.post("/ask", json={"question": "How long do refunds take?"}).json()
    assert answer["sources"][0]["document"] == "faq.txt"

    assert client.delete("/documents/faq.txt").status_code == 200
    assert client.delete("/documents/faq.txt").status_code == 404


def test_validation_errors(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    bad = {"file": ("x.exe", b"data", "application/octet-stream")}
    assert client.post("/documents/upload", files=bad).status_code == 400


def test_home_page(client):
    assert "Ask your documents" in client.get("/").text
