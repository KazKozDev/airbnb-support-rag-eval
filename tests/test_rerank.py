"""Tests for v2: reranker ordering (mocked cross-encoder) and remap Jaccard."""
from unittest.mock import MagicMock, patch

from eval.remap_chunk_ids import jaccard


def test_reranker_orders_by_score():
    chunks = [
        {"id": "c0000", "text": "irrelevant text"},
        {"id": "c0001", "text": "the actual answer"},
        {"id": "c0002", "text": "somewhat related"},
    ]
    # stub sentence_transformers so the test needs neither the package nor a model
    fake_st = MagicMock()
    fake_st.CrossEncoder.return_value.predict = MagicMock(return_value=[0.1, 0.9, 0.5])
    with patch.dict("sys.modules", {"sentence_transformers": fake_st}):
        import importlib
        import src.rerank
        importlib.reload(src.rerank)
        top = src.rerank.Reranker("fake-model").rerank("question?", chunks, top_k=2)

    assert [c["id"] for c in top] == ["c0001", "c0002"]
    assert top[0]["rerank_score"] == 0.9
    # original chunk dicts are not mutated
    assert "rerank_score" not in chunks[1]


def test_jaccard():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"c"}) == 0.0
    assert jaccard(set(), set()) == 0.0
    assert abs(jaccard({"a", "b", "c"}, {"b", "c", "d"}) - 0.5) < 1e-9
