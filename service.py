"""RagService: ingest documents, retrieve relevant chunks, ask Claude using only that context."""
import io
import os
from pathlib import Path

from . import config
from .rag import BM25Index, chunk_text

SYSTEM_PROMPT = (
    "You answer questions about the user's documents. Use ONLY the numbered context passages provided. "
    "Cite the passages you used, like [1] or [2]. If the answer is not in the passages, reply exactly: "
    "I don't know based on the provided documents. Be concise and do not invent facts."
)
NO_ANSWER = "I don't know based on the provided documents."
SUPPORTED = (".txt", ".md", ".pdf")


class RagService:
    def __init__(self, index_path: Path = config.INDEX_FILE, llm=None):
        self.index_path = Path(index_path)
        self.llm = llm
        self.index = BM25Index()
        self.index.load(self.index_path)

    # ------------------------------------------------------------ ingestion
    def ingest_text(self, name: str, text: str) -> int:
        pieces = chunk_text(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP)
        if not pieces:
            raise ValueError("The document has no readable text.")
        self.index.add_document(name, pieces)
        self.index.save(self.index_path)
        return len(pieces)

    def ingest_file(self, filename: str, data: bytes) -> tuple[str, int]:
        name = os.path.basename(filename or "")
        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED:
            raise ValueError(f"Unsupported file type '{ext}'. Use: {', '.join(SUPPORTED)}")
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise ValueError("File is too large (limit 5 MB).")
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            text = data.decode("utf-8", errors="ignore")
        return name, self.ingest_text(name, text)

    def load_directory(self, folder: Path) -> int:
        """Index every supported file in a folder. Returns the number of documents loaded."""
        loaded = 0
        for path in sorted(Path(folder).glob("*")):
            if path.suffix.lower() in SUPPORTED and path.is_file():
                self.ingest_file(path.name, path.read_bytes())
                loaded += 1
        return loaded

    def list_documents(self) -> list[dict]:
        return self.index.documents()

    def delete_document(self, name: str) -> bool:
        removed = self.index.remove_document(name)
        if removed:
            self.index.save(self.index_path)
        return removed

    # ------------------------------------------------------------ question answering
    def ask(self, question: str, top_k: int = config.DEFAULT_TOP_K) -> dict:
        hits = self.index.search(question, k=top_k)
        if not hits:
            return {"answer": NO_ANSWER, "sources": [], "mode": "no-context"}

        sources = [
            {"id": i, "document": chunk.doc, "chunk": chunk.chunk_id,
             "score": round(score, 3), "snippet": chunk.text[:220]}
            for i, (chunk, score) in enumerate(hits, start=1)
        ]
        if self.llm is None:                          # no API key: still useful, just no generation
            best = hits[0][0]
            return {"answer": f"(Retrieval-only mode, no Claude key set) Best match from "
                              f"{best.doc}: {best.text[:400]}",
                    "sources": sources, "mode": "retrieval-only"}

        context = "\n\n".join(f"[{i}] (source: {c.doc})\n{c.text}" for i, (c, _s) in enumerate(hits, start=1))
        prompt = f"Context passages:\n\n{context}\n\nQuestion: {question}"
        answer = self.llm.generate(SYSTEM_PROMPT, prompt)
        return {"answer": answer, "sources": sources, "mode": self.llm.name}
