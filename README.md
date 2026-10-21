# RAG API with Claude

A **Retrieval-Augmented Generation (RAG)** service exposed as a **REST API (FastAPI)**. Upload your own documents, ask questions in plain English, and get answers from **Claude** that are grounded in those documents, with the source passages shown for every answer.

## Why RAG?
A language model only knows what it was trained on. RAG first **retrieves** the passages that match the question, then asks the model to answer **using only those passages**. That keeps answers tied to your data, reduces made-up answers and lets you verify the sources.

## How it works
```
 upload (.txt / .md / .pdf)             question
          |                                |
   extract text                            v
          |                       BM25 retriever ranks chunks
   split into overlapping                  |
   chunks (150 words, 30 overlap)          v
          |                       top-k chunks + question -> prompt
   BM25 index (saved to JSON)              |
                                           v
                              Claude answers using ONLY the context,
                              citing passages like [1], [2]
                                           |
                                           v
                              JSON: answer + sources + mode
```
- If **no chunk matches**, the API replies "I don't know based on the provided documents" **without calling the model** (saves cost and avoids hallucination).
- If **no API key** is set, it runs in **retrieval-only mode** and returns the best matching passage, so you can try it for free.

## API endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Simple web page: upload documents and ask questions |
| GET | `/health` | Status, model mode, number of documents |
| GET | `/documents` | List indexed documents |
| POST | `/documents/upload` | Upload a `.txt`, `.md` or `.pdf` file (max 5 MB) |
| POST | `/documents/text` | Add a document as JSON `{name, text}` |
| DELETE | `/documents/{name}` | Remove a document |
| POST | `/ask` | `{question, top_k}` returns `{answer, sources, mode}` |
| GET | `/docs` | Interactive Swagger documentation |

Example:
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of annual leave do I get?"}'
```
```json
{
  "answer": "Full-time employees get 24 days of paid annual leave per year [1].",
  "sources": [{"id": 1, "document": "leave_policy.md", "chunk": 0, "score": 6.66, "snippet": "..."}],
  "mode": "claude"
}
```

## Setup
Requires Python 3.9+.
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# edit .env and paste your Anthropic API key (https://console.anthropic.com)

uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000 for the web page or http://127.0.0.1:8000/docs for the API docs.

Three sample documents (a fictional company's leave, remote-work and expense policies) are indexed on the first run, so you can ask questions straight away, for example:
- "How many days of annual leave do I get?"
- "Which days do I work from the office?"
- "Who approves an expense of 20,000 rupees?"

## Configuration
Environment variables (or `.env`): `ANTHROPIC_API_KEY`, `CLAUDE_MODEL` (default `claude-sonnet-5-5`), `MAX_TOKENS`, `RAG_DATA_DIR`. Chunk size, overlap and top-k are in `app/config.py`.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
`tests/test_rag.py` tests chunking, retrieval ranking, persistence, uploads and the prompt, with a fake model (no key or internet needed). `tests/test_api.py` tests the HTTP endpoints.

## Project structure
```
app/main.py      FastAPI routes and the small web page
app/service.py   RagService: ingest, retrieve, build the prompt, answer
app/rag.py       chunking and the BM25 retriever
app/llm.py       Claude wrapper (easy to swap or fake)
app/config.py    settings
data/docs/       sample documents
tests/           unit and API tests
```

## Design decisions
- **BM25 keyword retrieval** instead of embeddings: no extra model downloads, runs anywhere, easy to explain and test.
- **Citations and sources** in every response so answers can be checked.
- **Strict system prompt** ("use ONLY the context") plus a no-match shortcut to limit hallucination.
- The model client is behind a tiny interface, so tests run without network access.

## Limitations and next steps
- Keyword search can miss synonyms ("vacation" vs "leave"). **Next step:** replace BM25 with embeddings and a vector database (for example FAISS, Chroma or pgvector), and combine both (hybrid search).
- The index is a JSON file and fully reloaded in memory, so it suits small to medium document sets.
- No authentication or rate limiting yet.
- Add answer evaluation (a set of test questions with expected sources).

## Tech stack
Python · FastAPI · Claude (Anthropic API) · BM25 · pypdf · pytest
