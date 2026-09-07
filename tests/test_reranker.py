"""
Unit and integration tests for Cross-Encoder Reranker and RerankedRetriever.
"""

from unittest.mock import MagicMock
from app.models.schemas import RetrievalResult, MetadataFilter
from app.retrieval.reranker import Reranker, RerankedRetriever


def test_reranker_reordering_with_mock():
    # Candidate 1 initially rank 1
    doc_1 = RetrievalResult(
        document_id="doc-1",
        content="General company policies",
        rank=1,
        retriever="dense"
    )
    # Candidate 2 initially rank 2, but has exact relevant answer
    doc_2 = RetrievalResult(
        document_id="doc-2",
        content="Customers have a 30-day window to request a refund.",
        rank=2,
        retriever="dense"
    )
    # Candidate 3 initially rank 3
    doc_3 = RetrievalResult(
        document_id="doc-3",
        content="Hardware return RMA guidelines",
        rank=3,
        retriever="dense"
    )

    # Mock cross-encoder where candidate 2 gets highest score
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.15, 0.95, 0.40]

    reranker = Reranker(cross_encoder=mock_model)
    reranked = reranker.rerank(
        query="What is the refund window?",
        documents=[doc_1, doc_2, doc_3],
        top_k=2
    )

    assert len(reranked) == 2

    # Candidate 2 must be promoted to rank 1
    top = reranked[0]
    assert top.document_id == "doc-2"
    assert top.final_rank == 1
    assert top.original_rank == 2
    assert top.rerank_score == 0.95
    assert top.score == 0.95

    # Candidate 3 is rank 2 (0.40 > 0.15)
    second = reranked[1]
    assert second.document_id == "doc-3"
    assert second.final_rank == 2
    assert second.original_rank == 3
    assert second.rerank_score == 0.40


def test_reranker_empty_inputs():
    reranker = Reranker(cross_encoder=MagicMock())
    assert reranker.rerank("", [MagicMock()]) == []
    assert reranker.rerank("query", []) == []


def test_reranked_retriever_end_to_end():
    # End-to-end two-stage test on real knowledge base
    retriever = RerankedRetriever()
    results = retriever.retrieve(
        query="What is the customer refund period?",
        top_k=3,
        candidate_pool_k=10
    )

    assert len(results) == 3
    for rank, res in enumerate(results, start=1):
        assert res.rank == rank
        assert res.final_rank == rank
        assert res.original_rank is not None
        assert res.rerank_score is not None
        assert res.score == res.rerank_score
        assert "reranked" in res.retriever

    # Verify scores are sorted descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_reranked_retriever_with_metadata_filter():
    retriever = RerankedRetriever()
    filter_v2 = MetadataFilter(version="2.0")

    results = retriever.retrieve(
        query="refund policy rules",
        top_k=2,
        filters=filter_v2
    )

    assert len(results) > 0
    for res in results:
        assert str(res.metadata.get("version")) == "2.0"
