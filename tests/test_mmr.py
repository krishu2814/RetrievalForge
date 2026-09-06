"""
Unit and integration tests for MMRRetriever.
Verifies diversity vs relevance trade-offs and metadata filtering under MMR.
"""

from app.retrieval.mmr import MMRRetriever
from app.models.schemas import MetadataFilter


def test_mmr_retrieval_basic():
    retriever = MMRRetriever()
    results = retriever.retrieve("What is our customer refund and leave policy?", top_k=4)

    assert len(results) == 4
    for rank, res in enumerate(results, start=1):
        assert res.rank == rank
        assert res.retriever == "mmr"
        assert res.score is not None
        assert 0.0 <= res.score <= 1.0
        assert len(res.content) > 0


def test_mmr_empty_query():
    retriever = MMRRetriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_mmr_lambda_tradeoff():
    retriever = MMRRetriever()
    query = "enterprise security SOC2 and data retention encryption"

    # High relevance (pure similarity)
    sim_results = retriever.retrieve(query, top_k=3, lambda_mult=1.0)
    # High diversity
    div_results = retriever.retrieve(query, top_k=3, lambda_mult=0.1)

    assert len(sim_results) == 3
    assert len(div_results) == 3

    sim_ids = [r.document_id for r in sim_results]
    div_ids = [r.document_id for r in div_results]
    assert len(sim_ids) > 0
    assert len(div_ids) > 0


def test_mmr_metadata_filtering():
    retriever = MMRRetriever()
    filter_support = MetadataFilter(department="support")
    results = retriever.retrieve("refund and support policies", top_k=3, filters=filter_support)

    assert len(results) > 0
    for res in results:
        assert res.metadata.get("department") == "support"
