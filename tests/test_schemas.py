"""
Unit tests for app.models.schemas.
Verifies RetrievalResult data structure and MetadataFilter matching logic.
"""

from app.models.schemas import RetrievalResult, MetadataFilter, RetrievalTrace


def test_retrieval_result_defaults_and_diagnostics():
    result = RetrievalResult(
        document_id="refund-v2",
        content="Refunds within 30 days are allowed.",
        metadata={"department": "support", "version": "2.0"},
        score=0.89,
        rank=1,
        retriever="hybrid"
    )

    assert result.document_id == "refund-v2"
    assert result.rank == 1
    assert result.retriever == "hybrid"
    assert result.dense_score is None

    # Test updating diagnostic scores
    result.dense_score = 0.85
    result.bm25_score = 12.4
    result.fusion_score = 0.032
    result.rerank_score = 0.94
    result.query_variant = "customer refund window"

    assert result.dense_score == 0.85
    assert result.bm25_score == 12.4
    assert result.fusion_score == 0.032
    assert result.rerank_score == 0.94
    assert result.query_variant == "customer refund window"


def test_metadata_filter_matches():
    doc_metadata = {
        "department": "support",
        "document_type": "policy",
        "version": "2.0",
        "access_level": "public",
        "topic": "refunds"
    }

    # Empty filter should match everything
    empty_filter = MetadataFilter()
    assert empty_filter.matches(doc_metadata) is True

    # Matching department and version
    matching_filter = MetadataFilter(department="support", version="2.0")
    assert matching_filter.matches(doc_metadata) is True

    # Mismatched department
    mismatched_filter = MetadataFilter(department="engineering")
    assert mismatched_filter.matches(doc_metadata) is False

    # Mismatched version
    old_version_filter = MetadataFilter(version="1.0")
    assert old_version_filter.matches(doc_metadata) is False

    # Custom field filter
    custom_filter = MetadataFilter(custom={"access_level": "public"})
    assert custom_filter.matches(doc_metadata) is True

    custom_fail = MetadataFilter(custom={"access_level": "confidential"})
    assert custom_fail.matches(doc_metadata) is False


def test_retrieval_trace():
    trace = RetrievalTrace(
        query="What is the refund period?",
        strategy="hybrid_reranked",
        query_variants=["What is the refund period?", "refund deadline days"],
        filters=MetadataFilter(department="support")
    )

    assert trace.query == "What is the refund period?"
    assert trace.strategy == "hybrid_reranked"
    assert len(trace.query_variants) == 2
    assert trace.filters.department == "support"
    assert len(trace.candidates) == 0
