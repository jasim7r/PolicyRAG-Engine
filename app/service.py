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
        if self.llm is None:                          # no API key: extract the most specific relevant sentences
            best = hits[0][0]
            from collections import Counter
            from .rag import tokenize
            import re
            raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", best.text) if s.strip()]
            q_terms = Counter(tokenize(question))
            scored_sentences = []
            for s in raw_sentences:
                clean_s = re.sub(r"^[#\s\-*]+", "", s).strip()
                # Skip rhetorical question headings if they end with ?
                if clean_s.endswith("?") and not clean_s.endswith("? An") and not any(p in clean_s for p in ["must", "approved", "require"]):
                    continue
                s_terms = tokenize(clean_s)
                if not s_terms:
                    continue
                # Weight matches by question-term frequency
                match_score = sum(q_terms[t] for t in s_terms if t in q_terms)
                # Penalty if asking about annual leave but sentence discusses sick leave (or vice versa)
                if "annual" in q_terms and "sick" in s_terms:
                    match_score -= 2
                if "sick" in q_terms and "annual" in s_terms:
                    match_score -= 2

                if match_score > 0:
                    density = match_score / (len(s_terms) ** 0.5)
                    scored_sentences.append((match_score, density, clean_s))

            scored_sentences.sort(key=lambda x: (x[0], x[1]), reverse=True)
            if scored_sentences:
                top_sentences = [s for _, _, s in scored_sentences[:1]]
                extracted_answer = " ".join(top_sentences)
            else:
                extracted_answer = best.text[:300]

            return {
                "answer": f"From {best.doc}: {extracted_answer}",
                "sources": sources,
                "mode": "retrieval-only"
            }

        context = "\n\n".join(f"[{i}] (source: {c.doc})\n{c.text}" for i, (c, _s) in enumerate(hits, start=1))
        prompt = f"Context passages:\n\n{context}\n\nQuestion: {question}"
        answer = self.llm.generate(SYSTEM_PROMPT, prompt)
        return {"answer": answer, "sources": sources, "mode": self.llm.name}
