"""v2: cross-encoder reranking.

Why a cross-encoder: bi-encoder retrieval (embeddings) scores query and chunk
independently, so it misses fine-grained interactions ("Low rating" vs "Lowest
rating"). A cross-encoder reads (query, chunk) as one input and re-scores the
candidate pool. The pattern is: retrieve a WIDE pool cheaply (hybrid, top-20),
then re-rank it precisely and keep top-k. Cost: +~100-300ms CPU latency,
zero API cost (local model).

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~90MB, fast on CPU).
Stronger option for the quality/latency trade-off table: BAAI/bge-reranker-base.
"""
from sentence_transformers import CrossEncoder

from src import config


class Reranker:
    def __init__(self, model_name: str = config.RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """Score (query, chunk_text) pairs, return top_k chunks by score."""
        if len(chunks) <= top_k:
            # still rerank: order matters for context assembly and MRR
            pass
        scores = self.model.predict([(query, c["text"]) for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        out = []
        for chunk, score in ranked[:top_k]:
            out.append({**chunk, "rerank_score": round(float(score), 4)})
        return out
