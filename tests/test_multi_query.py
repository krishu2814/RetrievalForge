"""
Unit and integration tests for MultiQueryRetriever.
Verifies multi-query generation, candidate deduplication, and query variant provenance tracking.
"""

from unittest.mock import MagicMock
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.query_expansion import QueryExpander
from app.models.schemas import MetadataFilter


def test_multi_query_retrieval_end_to_end():
    retriever = MultiQueryRetriever()
    results = retriever.retrieve("What is the refund period?", top_k=3, num_queries=3)

    assert len(results) == 3
    assert len(retriever.last_generated_queries) >= 2

    # Check ranks, retriever tag, and provenance
    seen_ids = set()
    for rank, res in enumerate(results, start=1):
        assert res.rank == rank
        assert res.retriever == "multi_query"
        assert res.query_variant is not None
        assert len(res.query_variant) > 0

        # Verify deduplication
        chunk_key = res.metadata.get("chunk_id") or res.document_id
        assert chunk_key not in seen_ids
        seen_ids.add(chunk_key)


def test_multi_query_empty_query():
    retriever = MultiQueryRetriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.last_generated_queries == []


def test_multi_query_with_mock_expander():
    # Mock expander returning 2 distinct queries
    mock_expander = MagicMock(spec=QueryExpander)
    mock_expander.expand.return_value = [
        "What is the vacation policy?",
        "employee paid time off vacation accrual"
    ]

    retriever = MultiQueryRetriever(expander=mock_expander)
    results = retriever.retrieve("What is the vacation policy?", top_k=3)

    assert len(results) > 0
    assert retriever.last_generated_queries == [
        "What is the vacation policy?",
        "employee paid time off vacation accrual"
    ]
    for res in results:
        assert res.retriever == "multi_query"


def test_multi_query_metadata_filtering():
    retriever = MultiQueryRetriever()
    filter_v2 = MetadataFilter(version="2.0")
    results = retriever.retrieve("refund request policy", top_k=3, filters=filter_v2)

    assert len(results) > 0
    for res in results:
        assert str(res.metadata.get("version")) == "2.0"
        assert res.metadata.get("document_id") != "refund-policy-v1"
