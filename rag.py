"""Core RAG building blocks: chunking and a BM25 keyword retriever (pure Python, no heavy deps)."""
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "i", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "will", "with", "do", "does", "can", "my", "me", "we", "you", "your", "our", "if",
}
K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase words, minus stop-words, with very light plural stemming."""
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS or len(word) < 2:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        tokens.append(word)
    return tokens


def chunk_text(text: str, size: int = 150, overlap: int = 30) -> list[str]:
    """Split text into overlapping word windows so answers are not cut at chunk borders."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


@dataclass
class Chunk:
    doc: str
    chunk_id: int
    text: str


class BM25Index:
    """Keeps chunks in memory and ranks them against a question with BM25."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._avg_len = 0.0

    def _rebuild(self):
        self._tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._df = Counter(term for tf in self._tf for term in tf)
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0

    def add_document(self, doc: str, pieces: list[str]):
        self.remove_document(doc, rebuild=False)
        self.chunks.extend(Chunk(doc, i, piece) for i, piece in enumerate(pieces))
        self._rebuild()

    def remove_document(self, doc: str, rebuild: bool = True) -> bool:
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.doc != doc]
        if rebuild:
            self._rebuild()
        return len(self.chunks) != before

    def documents(self) -> list[dict]:
        counts = Counter(c.doc for c in self.chunks)
        return [{"name": name, "chunks": n} for name, n in sorted(counts.items())]

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        terms = set(tokenize(query))
        n = len(self.chunks)
        if not terms or n == 0:
            return []
        scored = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                df = self._df[term]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                norm = 1 - B + B * self._len[i] / (self._avg_len or 1)
                score += idf * freq * (K1 + 1) / (freq + K1 * norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [(self.chunks[i], score) for score, i in scored[:k]]

    # ---- persistence
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [{"doc": c.doc, "chunk_id": c.chunk_id, "text": c.text} for c in self.chunks]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path):
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(d["doc"], d["chunk_id"], d["text"]) for d in data]
        self._rebuild()
