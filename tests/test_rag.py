"""Core RAG tests - no API key, internet or web server needed.  Run: python -m pytest -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from app.rag import BM25Index, chunk_text, tokenize
from app.service import NO_ANSWER, RagService

SAMPLES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "docs")


class FakeLLM:
    name = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return "Fake answer [1]"


@pytest.fixture
def service(tmp_path):
    svc = RagService(index_path=tmp_path / "index.json", llm=FakeLLM())
    svc.load_directory(SAMPLES)
    return svc


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("How many days of leaves do I get?") == ["many", "day", "leave", "get"]


def test_chunking_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), size=40, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]       # overlap
    assert chunks[-1].split()[-1] == "w99"                         # nothing lost
    assert chunk_text("   ") == []


def test_chunk_overlap_validation():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, overlap=5)


def test_bm25_ranks_relevant_document_first(service):
    best, _score = service.index.search("annual leave carry over", k=1)[0]
    assert best.doc == "leave_policy.md"
    best, _score = service.index.search("hotel per night metro cities", k=1)[0]
    assert best.doc == "expense_policy.md"
    best, _score = service.index.search("VPN public wifi", k=1)[0]
    assert best.doc == "remote_work_policy.md"


def test_ask_uses_context_and_returns_sources(service):
    result = service.ask("How many days of annual leave do I get?")
    assert result["mode"] == "fake" and result["answer"] == "Fake answer [1]"
    assert result["sources"][0]["document"] == "leave_policy.md"
    system, prompt = service.llm.calls[0]
    assert "ONLY" in system and "24 days" in prompt and "How many days" in prompt


def test_unrelated_question_skips_llm(service):
    result = service.ask("quantum chromodynamics lagrangian")
    assert result == {"answer": NO_ANSWER, "sources": [], "mode": "no-context"}
    assert service.llm.calls == []


def test_retrieval_only_mode_without_llm(tmp_path):
    svc = RagService(index_path=tmp_path / "i.json", llm=None)
    svc.load_directory(SAMPLES)
    result = svc.ask("sick leave medical certificate")
    assert result["mode"] == "retrieval-only" and "leave_policy.md" in result["answer"]


def test_ingest_replace_delete_and_persistence(tmp_path):
    path = tmp_path / "index.json"
    svc = RagService(index_path=path)
    assert svc.ingest_text("notes.txt", "Kubernetes pods run containers on cluster nodes.") == 1
    svc.ingest_text("notes.txt", "Airflow schedules pipelines using DAG files.")      # same name replaces
    assert [d["name"] for d in svc.list_documents()] == ["notes.txt"]
    assert svc.index.search("airflow dag")[0][0].doc == "notes.txt"
    assert svc.index.search("kubernetes") == []

    reloaded = RagService(index_path=path)                                            # survives restart
    assert reloaded.list_documents() == [{"name": "notes.txt", "chunks": 1}]
    assert reloaded.delete_document("notes.txt") is True
    assert reloaded.delete_document("notes.txt") is False


def test_file_upload_validation(tmp_path):
    svc = RagService(index_path=tmp_path / "i.json")
    assert svc.ingest_file("../../etc/hello.txt", b"hello world data")[0] == "hello.txt"   # path stripped
    with pytest.raises(ValueError):
        svc.ingest_file("virus.exe", b"x")
    with pytest.raises(ValueError):
        svc.ingest_file("empty.txt", b"   ")


def test_pdf_upload(tmp_path):
    from pypdf import PdfWriter
    import io
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    svc = RagService(index_path=tmp_path / "i.json")
    with pytest.raises(ValueError):                          # blank PDF = no readable text
        svc.ingest_file("blank.pdf", buf.getvalue())
