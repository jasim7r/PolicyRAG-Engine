"""FastAPI app: upload documents, ask questions, get answers with sources.

Run:  uvicorn app.main:app --reload     then open http://127.0.0.1:8000  (docs at /docs)
"""
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config
from .llm import LLMError, build_llm
from .service import RagService

app = FastAPI(title="RAG API with Claude", version="1.0.0",
              description="Retrieval-Augmented Generation over your own documents, answered by Claude.")

service = RagService(llm=build_llm())
if not service.list_documents() and config.SAMPLE_DOCS_DIR.exists():
    service.load_directory(config.SAMPLE_DOCS_DIR)         # first run: index the sample documents


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(config.DEFAULT_TOP_K, ge=1, le=10)


class TextDocument(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "llm": "claude" if service.llm else "retrieval-only",
            "model": config.CLAUDE_MODEL if service.llm else None,
            "documents": len(service.list_documents())}


@app.get("/documents")
def list_documents():
    return service.list_documents()


@app.post("/documents/text", status_code=201)
def add_text_document(doc: TextDocument):
    try:
        chunks = service.ingest_text(doc.name, doc.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"name": doc.name, "chunks": chunks}


@app.post("/documents/upload", status_code=201)
def upload_document(file: UploadFile = File(...)):
    try:
        name, chunks = service.ingest_file(file.filename, file.file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"name": name, "chunks": chunks}


@app.delete("/documents/{name}")
def delete_document(name: str):
    if not service.delete_document(name):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": name}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return service.ask(request.question, request.top_k)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"Language model error: {exc}")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG with Claude</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:30px auto;padding:0 16px;background:#0f1720;color:#e6edf3}
h1{font-size:22px} .card{background:#16212c;border:1px solid #243242;border-radius:10px;padding:14px;margin:14px 0}
textarea,input[type=text]{width:100%;box-sizing:border-box;background:#0f1720;color:#e6edf3;border:1px solid #2d4056;border-radius:8px;padding:10px;font-size:15px}
button{background:#1f6feb;color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:15px;cursor:pointer;margin-top:8px}
.src{font-size:13px;color:#9fb3c8;border-left:3px solid #2d4056;padding-left:10px;margin:8px 0} .mode{font-size:12px;color:#7ee787}
pre{white-space:pre-wrap;margin:0;font-family:inherit}
</style></head><body>
<h1>Ask your documents (RAG + Claude)</h1>
<div class="card"><b>1. Add a document</b> (.txt, .md or .pdf)<br>
<input type="file" id="file" accept=".txt,.md,.pdf"> <button onclick="upload()">Upload</button>
<div id="docs" class="src"></div></div>
<div class="card"><b>2. Ask a question</b>
<textarea id="q" rows="3" placeholder="e.g. How many days of annual leave do I get?"></textarea>
<button onclick="ask()">Ask</button></div>
<div class="card" id="out" style="display:none"><span class="mode" id="mode"></span><pre id="answer"></pre><div id="sources"></div></div>
<script>
async function docs(){const r=await fetch('/documents');const d=await r.json();
document.getElementById('docs').textContent='Indexed: '+(d.map(x=>x.name+' ('+x.chunks+' chunks)').join(', ')||'none');}
async function upload(){const f=document.getElementById('file').files[0];if(!f)return;
const fd=new FormData();fd.append('file',f);const r=await fetch('/documents/upload',{method:'POST',body:fd});
if(!r.ok){alert((await r.json()).detail);}docs();}
async function ask(){const q=document.getElementById('q').value.trim();if(q.length<3)return;
const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
const d=await r.json();const out=document.getElementById('out');out.style.display='block';
if(!r.ok){document.getElementById('answer').textContent=d.detail;document.getElementById('sources').textContent='';return;}
document.getElementById('mode').textContent='mode: '+d.mode;document.getElementById('answer').textContent=d.answer;
const s=document.getElementById('sources');s.textContent='';
d.sources.forEach(x=>{const e=document.createElement('div');e.className='src';
e.textContent='['+x.id+'] '+x.document+' (score '+x.score+'): '+x.snippet+'...';s.appendChild(e);});}
docs();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return PAGE
