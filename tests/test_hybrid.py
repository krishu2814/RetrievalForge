"""
Unit and integration tests for HybridRetriever and Reciprocal Rank Fusion (RRF).
"""

from app.models.schemas import RetrievalResult, MetadataFilter
from app.retrieval.hybrid import reciprocal_rank_fusion, HybridRetriever


def test_reciprocal_rank_fusion_math_and_deduplication():
    # Chunk 1: Present in both dense (#1) and sparse (#1)
    dense_1 = RetrievalResult(
        document_id="doc-1",
        content="Chunk 1 content",
        metadata={"chunk_id": "chunk_1"},
        score=0.9,
        rank=1,
        retriever="dense",
        dense_score=0.9
    )
    # Chunk 2: Present only in dense (#2)
    dense_2 = RetrievalResult(
        document_id="doc-2",
        content="Chunk 2 content",
        metadata={"chunk_id": "chunk_2"},
        score=0.8,
        rank=2,
        retriever="dense",
        dense_score=0.8
    )

    # Chunk 1 in sparse: rank 1
    sparse_1 = RetrievalResult(
        document_id="doc-1",
        content="Chunk 1 content",
        metadata={"chunk_id": "chunk_1"},
        score=15.0,
        rank=1,
        retriever="bm25",
        bm25_score=15.0
    )
    # Chunk 3: Present only in sparse (#2)
    sparse_3 = RetrievalResult(
        document_id="doc-3",
        content="Chunk 3 content",
        metadata={"chunk_id": "chunk_3"},
        score=12.0,
        rank=2,
        retriever="bm25",
        bm25_score=12.0
    )

    fused = reciprocal_rank_fusion(
        dense_results=[dense_1, dense_2],
        sparse_results=[sparse_1, sparse_3],
        rrf_k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        top_k=3
    )

    assert len(fused) == 3

    # Chunk 1 was #1 in BOTH, so it must be rank 1 in fusion
    top = fused[0]
    assert top.metadata["chunk_id"] == "chunk_1"
    assert top.rank == 1
    assert top.retriever == "hybrid"
    assert top.dense_score == 0.9
    assert top.bm25_score == 15.0

    # RRF score calculation check:
    # 0.5 * (1 / (60 + 1)) + 0.5 * (1 / (60 + 1)) = 1 / 61 ≈ 0.016393
    expected_top_score = round(1.0 / 61.0, 6)
    assert abs(top.fusion_score - expected_top_score) < 1e-5


def test_hybrid_retrieval_end_to_end():
    retriever = HybridRetriever()
    results = retriever.retrieve("What is the customer refund policy?", top_k=4)

    assert len(results) == 4
    for rank, res in enumerate(results, start=1):
        assert res.rank == rank
        assert res.retriever == "hybrid"
        assert res.fusion_score is not None
        assert res.score == res.fusion_score

    # Check descending fusion scores
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_retrieval_empty_query():
    retriever = HybridRetriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_hybrid_retrieval_with_metadata_filter():
    retriever = HybridRetriever()

    # Filter for version 2.0
    filter_v2 = MetadataFilter(version="2.0")
    results = retriever.retrieve("customer refund window", top_k=3, filters=filter_v2)

    assert len(results) > 0
    for res in results:
        assert str(res.metadata.get("version")) == "2.0"
        assert res.metadata.get("document_id") != "refund-policy-v1"
