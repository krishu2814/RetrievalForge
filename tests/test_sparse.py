"""
Unit and integration tests for BM25Retriever.
Verifies sparse lexical matching on error codes, exact terms, and metadata filtering.
"""

from app.retrieval.sparse import BM25Retriever
from app.models.schemas import MetadataFilter


def test_bm25_exact_error_code_retrieval():
    retriever = BM25Retriever()
    # Search for an exact technical error code
    results = retriever.retrieve("What is ERR_AUTH_TIMEOUT_504?", top_k=3)

    assert len(results) > 0
    top_result = results[0]
    assert top_result.retriever == "bm25"
    assert top_result.bm25_score is not None
    assert top_result.bm25_score > 0
    # The top result should contain the exact error code
    assert "ERR_AUTH_TIMEOUT_504" in top_result.content


def test_bm25_exact_keyword_retrieval():
    retriever = BM25Retriever()
    # Search for specific hardware appliance model
    results = retriever.retrieve("NovaEdge-500 return policy", top_k=3)

    assert len(results) > 0
    assert results[0].rank == 1
    assert "NovaEdge-500" in results[0].content or "NovaEdge" in results[0].content


def test_bm25_scores_descending():
    retriever = BM25Retriever()
    results = retriever.retrieve("refund request policy deadline", top_k=5)

    assert len(results) > 1
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_empty_and_unknown_queries():
    retriever = BM25Retriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []

    # Non-existent gibberish word should return empty (no overlapping vocabulary)
    assert retriever.retrieve("xyznonexistentterm123456789") == []


def test_bm25_metadata_filtering():
    retriever = BM25Retriever()

    # Query matching refunds, but filtered by department=support and version=2.0
    filter_v2 = MetadataFilter(department="support", version="2.0")
    results = retriever.retrieve("refund policy for customer accounts", top_k=3, filters=filter_v2)

    assert len(results) > 0
    for res in results:
        assert res.metadata.get("department") == "support"
        assert str(res.metadata.get("version")) == "2.0"
