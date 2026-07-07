"""Retrieval layer: vector (Chroma), keyword (BM25), hybrid (reciprocal rank fusion)."""
import re

from rank_bm25 import BM25Okapi

from src import config
from src.index import get_collection, load_chunks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self):
        self.chunks = load_chunks()
        self.by_id = {c["id"]: c for c in self.chunks}
        self.collection = get_collection()
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])
        self._reranker = None  # lazy: don't load the cross-encoder unless used

    @property
    def reranker(self):
        if self._reranker is None:
            from src.rerank import Reranker
            self._reranker = Reranker()
        return self._reranker

    # --- single-strategy searches: return ranked list of chunk ids ---
    def vector_search(self, query: str, k: int) -> list[str]:
        res = self.collection.query(query_texts=[query], n_results=k)
        return res["ids"][0]

    def bm25_search(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i]["id"] for i in ranked[:k]]

    def hybrid_search(self, query: str, k: int) -> list[str]:
        """Reciprocal Rank Fusion over vector and BM25 rankings."""
        fused: dict[str, float] = {}
        for ranking in (self.vector_search(query, k * 2), self.bm25_search(query, k * 2)):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        ranked = sorted(fused, key=fused.get, reverse=True)
        return ranked[:k]

    # --- public API ---
    def retrieve(self, query: str, k: int = config.TOP_K,
                 mode: str = config.RETRIEVAL_MODE,
                 rerank: bool = config.RERANK) -> list[dict]:
        """Two-stage retrieval: cheap wide recall, optional precise rerank.

        rerank=False: top-k straight from the chosen strategy (v0/v1).
        rerank=True:  top-RERANK_CANDIDATES pool -> cross-encoder -> top-k (v2).
        """
        pool_k = config.RERANK_CANDIDATES if rerank else k
        ids = {
            "vector": self.vector_search,
            "bm25": self.bm25_search,
            "hybrid": self.hybrid_search,
        }[mode](query, pool_k)
        candidates = [self.by_id[cid] for cid in ids]
        if rerank:
            return self.reranker.rerank(query, candidates, top_k=k)
        return candidates
