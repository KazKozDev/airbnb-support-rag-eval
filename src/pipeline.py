"""End-to-end pipeline: question -> retrieval -> generation -> logged answer."""
import datetime
import json

from src import config
from src.generation import generate_answer
from src.retrieval import Retriever
from src.tracing import get_tracer

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def log_request(record: dict):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ask(question: str, k: int = config.TOP_K, mode: str = config.RETRIEVAL_MODE,
        rerank: bool = config.RERANK,
        prompt_variant: str = config.PROMPT_VARIANT) -> dict:
    chunks = get_retriever().retrieve(question, k=k, mode=mode, rerank=rerank)
    gen = generate_answer(question, chunks, prompt_variant=prompt_variant)
    result = {
        "question": question,
        "retrieval_mode": mode,
        "k": k,
        "rerank": rerank,
        "retrieved_chunks": [{"id": c["id"], "page": c["page"],
                              "rerank_score": c.get("rerank_score")} for c in chunks],
        **gen,
    }
    log_request({"ts": datetime.datetime.utcnow().isoformat(), **result})
    get_tracer().log_qa(result)
    return result
