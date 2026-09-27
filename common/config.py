"""Central config. Every phase imports from here so models are swapped in one place (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Embeddings: "openai" (API) or "hf" (local sentence-transformers model, no key, runs on CPU/MPS)
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "hf").lower()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Optional: only needed for gated/private models or higher Hub rate limits. Empty -> anonymous.
HF_TOKEN = os.getenv("HF_TOKEN") or None
EMBED_MODEL = HF_EMBED_MODEL if EMBED_PROVIDER == "hf" else OPENAI_EMBED_MODEL

# Cross-encoder reranker (local sentence-transformers model), Phase 2+
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
# Phase 4: model that GRADES answers / generates the golden set. Should differ from (and ideally be
# stronger than) CHAT_MODEL, the model being graded: a model judging its own outputs is biased.
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1")

# SEC EDGAR requires "Name email" in the User-Agent of every request
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")

CACHE_DIR = ROOT / ".cache"
DATA_DIR = ROOT / "data"


def require_key() -> None:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is empty — paste your key into rag-mastery/.env")
