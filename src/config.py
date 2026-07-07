"""Central configuration. Everything tweakable for eval iterations lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load .env if present so keys/settings work without exporting them by hand.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = str(DATA_DIR / "chroma")
LOG_PATH = DATA_DIR / "requests.log.jsonl"

# --- Chunking (v0: fixed size; v2 will switch to structure-aware) ---
CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", 350))   # ~500 tokens
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", 50))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 5))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # vector | bm25 | hybrid
RRF_K = 60  # reciprocal rank fusion constant

# --- Embeddings (local, free) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --- Reranking (v2: cross-encoder over a wider candidate pool) ---
RERANK = os.getenv("RERANK", "false").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", 20))  # pool size before rerank

# --- Generation ---
# Provider: "ollama" (Ollama Cloud) or "anthropic". The generation + judge paths
# both route through src/providers.py, so switching provider needs no code change.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Ollama Cloud (https://ollama.com): create an API key in account settings.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com").rstrip("/")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
# The judge may live on a different endpoint than the generator — e.g. a small
# LOCAL generator (localhost:11434) scored by a strong CLOUD judge — so runs stay
# comparable to cloud-generated rows. Defaults to the generator's endpoint.
OLLAMA_JUDGE_HOST = os.getenv("OLLAMA_JUDGE_HOST", OLLAMA_HOST).rstrip("/")
OLLAMA_JUDGE_API_KEY = os.getenv("OLLAMA_JUDGE_API_KEY", OLLAMA_API_KEY)
# Fallback only — the model is normally chosen from the server's list via
# `python -m src.select_model`, which writes SELECTED_MODEL_PATH.
OLLAMA_DEFAULT_MODEL = "gpt-oss:120b"
SELECTED_MODEL_PATH = DATA_DIR / "selected_model.txt"

# Anthropic (used only when LLM_PROVIDER=anthropic).
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")

MAX_TOKENS = 1024
REFUSAL_MARKER = "NOT_IN_DOCUMENT"  # model must emit this token when answer is absent
PROMPT_VARIANT = os.getenv("PROMPT_VARIANT", "grounded")  # naive | grounded

# $/MTok — for cost telemetry in logs; override when the price sheet changes.
# Ollama Cloud has no per-token price by default, so cost defaults to 0 there.
_price_default_in = "0.0" if LLM_PROVIDER == "ollama" else "3.0"
_price_default_out = "0.0" if LLM_PROVIDER == "ollama" else "15.0"
PRICE_IN_PER_MTOK = float(os.getenv("PRICE_IN_PER_MTOK", _price_default_in))
PRICE_OUT_PER_MTOK = float(os.getenv("PRICE_OUT_PER_MTOK", _price_default_out))


def _read_selected_model() -> str:
    if SELECTED_MODEL_PATH.exists():
        return SELECTED_MODEL_PATH.read_text(encoding="utf-8").strip()
    return ""


def active_model() -> str:
    """The generation model actually used, resolved at call time.

    Ollama: OLLAMA_MODEL env override -> model picked via select_model -> default.
    Anthropic: LLM_MODEL.
    """
    if LLM_PROVIDER == "ollama":
        return os.getenv("OLLAMA_MODEL") or _read_selected_model() or OLLAMA_DEFAULT_MODEL
    return LLM_MODEL


def active_judge_model() -> str:
    """Judge model. Defaults to the generation model on Ollama, so the eval
    harness needs only one provider configured."""
    explicit = os.getenv("JUDGE_MODEL")
    if explicit:
        return explicit
    if LLM_PROVIDER == "ollama":
        return active_model()
    return "claude-sonnet-5"

# --- Tracing (optional, Langfuse self-hosted or cloud) ---
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"

# --- Eval ---
GOLDEN_DATASET = ROOT / "eval" / "golden_dataset.jsonl"
