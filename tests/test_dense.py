"""
Unit and integration tests for DenseRetriever.
Verifies vector similarity search, score normalization, and metadata filtering.
"""

from app.retrieval.dense import DenseRetriever
from app.models.schemas import MetadataFilter


def test_dense_retrieval_basic():
    retriever = DenseRetriever()
    results = retriever.retrieve("What is the refund period?", top_k=3)

    assert len(results) == 3
    for rank, res in enumerate(results, start=1):
        assert res.rank == rank
        assert res.retriever == "dense"
        assert res.dense_score is not None
        assert res.score == res.dense_score
        assert 0.0 <= res.score <= 1.0
        assert len(res.content) > 0

    # Verify scores are sorted descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_retrieval_empty_query():
    retriever = DenseRetriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_dense_retrieval_with_metadata_filter():
    retriever = DenseRetriever()

    # Filter strictly for active version 2.0
    filter_v2 = MetadataFilter(version="2.0")
    results = retriever.retrieve("What is the customer refund policy?", top_k=3, filters=filter_v2)

    assert len(results) > 0
    for res in results:
        assert str(res.metadata.get("version")) == "2.0"
        assert res.metadata.get("document_id") != "refund-policy-v1"


def test_dense_retrieval_with_dict_filter():
    retriever = DenseRetriever()

    # Filter using dictionary for security department
    results = retriever.retrieve("encryption and SOC2 compliance", top_k=3, filters={"department": "security"})

    assert len(results) > 0
    for res in results:
        assert res.metadata.get("department") == "security"
