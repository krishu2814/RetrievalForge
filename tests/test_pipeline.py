"""
Unit tests for the RAGPipeline and Strategy Pattern dispatching.
"""

import pytest
from unittest.mock import MagicMock

from app.pipeline.rag_pipeline import (
    RAGPipeline,
    get_pipeline,
    format_context,
    SUPPORTED_STRATEGIES,
)
from app.models.schemas import RetrievalResult, MetadataFilter, RetrievalTrace


def test_supported_strategies_list():
    strategies = RAGPipeline.list_strategies()
    assert "dense" in strategies
    assert "bm25" in strategies
    assert "hybrid" in strategies
    assert "hybrid_reranked" in strategies
    assert "hybrid_reranked_compressed" in strategies


def test_strategy_validation():
    pipeline = RAGPipeline(strategy="hybrid")
    assert pipeline.strategy == "hybrid"

    pipeline.set_strategy("dense")
    assert pipeline.strategy == "dense"

    with pytest.raises(ValueError, match="Unknown strategy"):
        pipeline.set_strategy("invalid_strategy_xyz")


def test_format_context_empty():
    assert format_context([]) == "No relevant documents found."


def test_format_context_with_documents():
    docs = [
        RetrievalResult(
            document_id="doc_1",
            content="First content snippet.",
            metadata={"source": "refund_policy.txt", "department": "Billing"},
            score=0.9,
            rank=1,
            retriever="hybrid",
        ),
        RetrievalResult(
            document_id="doc_2",
            content="Second content snippet.",
            metadata={"source": "security.txt", "department": "Security"},
            score=0.8,
            rank=2,
            retriever="hybrid",
        ),
    ]

    context = format_context(docs)
    assert "[Document 1] Source: refund_policy.txt | Department: Billing | Retriever: hybrid" in context
    assert "First content snippet." in context
    assert "[Document 2] Source: security.txt | Department: Security | Retriever: hybrid" in context
    assert "Second content snippet." in context
    assert "---" in context


def test_pipeline_dispatch_with_mock_retrievers():
    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = [
        RetrievalResult(
            document_id="dense_1",
            content="Dense result content",
            metadata={"source": "test.txt"},
            score=0.95,
            rank=1,
            retriever="dense",
        )
    ]

    mock_bm25 = MagicMock()
    mock_bm25.retrieve.return_value = [
        RetrievalResult(
            document_id="bm25_1",
            content="BM25 result content",
            metadata={"source": "test.txt"},
            score=0.85,
            rank=1,
            retriever="bm25",
        )
    ]

    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = [
        RetrievalResult(
            document_id="hybrid_1",
            content="Hybrid result content",
            metadata={"source": "test.txt"},
            score=0.03,
            rank=1,
            retriever="hybrid",
        )
    ]

    pipeline = RAGPipeline(
        strategy="dense",
        dense_retriever=mock_dense,
        bm25_retriever=mock_bm25,
        hybrid_retriever=mock_hybrid,
    )

    # Test Dense run
    trace = pipeline.run(query="test dense query", top_k=3)
    assert isinstance(trace, RetrievalTrace)
    assert trace.strategy == "dense"
    assert len(trace.candidates) == 1
    assert trace.candidates[0].document_id == "dense_1"
    assert "Dense result content" in trace.final_context
    mock_dense.retrieve.assert_called_once_with(query="test dense query", top_k=3, filters=None)

    # Test BM25 run
    trace_bm25 = pipeline.run(query="test bm25 query", strategy="bm25", top_k=2)
    assert trace_bm25.strategy == "bm25"
    assert trace_bm25.candidates[0].document_id == "bm25_1"
    mock_bm25.retrieve.assert_called_once_with(query="test bm25 query", top_k=2, filters=None)

    # Test Hybrid run
    trace_hybrid = pipeline.run(query="test hybrid query", strategy="hybrid", top_k=5)
    assert trace_hybrid.strategy == "hybrid"
    assert trace_hybrid.candidates[0].document_id == "hybrid_1"
    mock_hybrid.retrieve.assert_called_once_with(query="test hybrid query", top_k=5, filters=None)


def test_pipeline_get_context():
    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = [
        RetrievalResult(
            document_id="d1",
            content="Context snippet for LLM.",
            metadata={"source": "manual.txt", "department": "Ops"},
            score=0.9,
            rank=1,
            retriever="dense",
        )
    ]

    pipeline = RAGPipeline(strategy="dense", dense_retriever=mock_dense)
    context = pipeline.get_context("query about ops")

    assert "[Document 1] Source: manual.txt | Department: Ops | Retriever: dense" in context
    assert "Context snippet for LLM." in context


def test_pipeline_mmr_and_multi_query():
    mock_mmr = MagicMock()
    mock_mmr.retrieve.return_value = [
        RetrievalResult(
            document_id="mmr_1",
            content="MMR content",
            metadata={"source": "doc.txt"},
            score=0.88,
            rank=1,
            retriever="mmr",
        )
    ]

    mock_mq = MagicMock()
    mock_mq.retrieve.return_value = [
        RetrievalResult(
            document_id="mq_1",
            content="Multi-query content",
            metadata={"source": "doc.txt"},
            score=0.92,
            rank=1,
            retriever="multi_query",
            query_variant="expanded query 1",
        )
    ]

    pipeline = RAGPipeline(
        strategy="mmr",
        mmr_retriever=mock_mmr,
        multi_query_retriever=mock_mq,
    )

    trace_mmr = pipeline.run(query="diversity query", top_k=4)
    assert trace_mmr.strategy == "mmr"
    assert trace_mmr.candidates[0].document_id == "mmr_1"

    trace_mq = pipeline.run(query="multi-intent query", strategy="multi_query", top_k=3)
    assert trace_mq.strategy == "multi_query"
    assert trace_mq.candidates[0].document_id == "mq_1"
    assert "expanded query 1" in trace_mq.query_variants


def test_pipeline_hybrid_reranked_and_compressed():
    mock_hybrid = MagicMock()
    candidate_doc = RetrievalResult(
        document_id="c1",
        content="First sentence relevant. Second sentence distractor.",
        metadata={"source": "policy.txt"},
        score=0.5,
        rank=1,
        retriever="hybrid",
    )
    mock_hybrid.retrieve.return_value = [candidate_doc]

    mock_reranker = MagicMock()
    reranked_doc = RetrievalResult(
        document_id="c1",
        content="First sentence relevant. Second sentence distractor.",
        metadata={"source": "policy.txt"},
        score=2.34,
        rank=1,
        retriever="cross_encoder_reranked",
        rerank_score=2.34,
        original_rank=1,
        final_rank=1,
    )
    mock_reranker.rerank.return_value = [reranked_doc]

    mock_compressor = MagicMock()
    compressed_doc = RetrievalResult(
        document_id="c1",
        content="First sentence relevant.",
        metadata={"source": "policy.txt", "compression_ratio": 0.5},
        score=2.34,
        rank=1,
        retriever="cross_encoder_reranked_compressed",
    )
    mock_compressor.compress_documents.return_value = [compressed_doc]

    pipeline = RAGPipeline(
        strategy="hybrid_reranked",
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
        compressor=mock_compressor,
    )

    # Test hybrid_reranked
    trace_reranked = pipeline.run(query="test query", top_k=1)
    assert trace_reranked.strategy == "hybrid_reranked"
    assert trace_reranked.candidates[0].retriever == "cross_encoder_reranked"

    # Test hybrid_reranked_compressed
    trace_compressed = pipeline.run(query="test query", strategy="hybrid_reranked_compressed", top_k=1)
    assert trace_compressed.strategy == "hybrid_reranked_compressed"
    assert trace_compressed.candidates[0].content == "First sentence relevant."
    mock_compressor.compress_documents.assert_called()


def test_pipeline_metadata_filters_passed():
    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = []

    pipeline = RAGPipeline(strategy="dense", dense_retriever=mock_dense)
    filt = MetadataFilter(department="Billing", version="2.0")
    pipeline.run(query="refund window", filters=filt)

    mock_dense.retrieve.assert_called_once_with(query="refund window", top_k=5, filters=filt)


def test_get_pipeline_factory():
    p = get_pipeline("hybrid")
    assert isinstance(p, RAGPipeline)
    assert p.strategy == "hybrid"


def test_pipeline_real_hybrid_integration():
    pipeline = RAGPipeline(strategy="hybrid")
    trace = pipeline.run(query="What is the refund window for enterprise accounts?", top_k=2)
    assert len(trace.candidates) == 2
    assert trace.final_context != ""
    assert "[Document 1]" in trace.final_context
    assert trace.strategy == "hybrid"


