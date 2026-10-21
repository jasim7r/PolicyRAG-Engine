"""Settings, read from environment variables (or a .env file)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:          # python-dotenv is optional
    pass

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))

BASE_DIR = Path(__file__).resolve().parent.parent          # project folder, so it works from any directory
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data"))
INDEX_FILE = DATA_DIR / "index.json"
SAMPLE_DOCS_DIR = BASE_DIR / "data" / "docs"

CHUNK_WORDS = 150            # words per chunk
CHUNK_OVERLAP = 30           # words shared between neighbouring chunks
DEFAULT_TOP_K = 4
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
