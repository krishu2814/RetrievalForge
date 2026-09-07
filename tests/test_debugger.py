"""
Unit tests for the RetrievalDebugger module.
"""

import pytest
from unittest.mock import MagicMock

from app.debugging.retrieval_debugger import RetrievalDebugger
from app.models.schemas import RetrievalResult, RetrievalTrace, MetadataFilter


def test_debugger_format_trace_text():
    debugger = RetrievalDebugger()

    docs = [
        RetrievalResult(
            document_id="doc_1",
            content="Enterprise accounts receive 24/7 dedicated support.",
            metadata={"source": "support.txt", "department": "Support"},
            score=0.92,
            rank=1,
            dense_score=0.88,
            bm25_score=14.2,
            fusion_score=0.016,
            rerank_score=3.14,
            original_rank=3,
        )
    ]

    trace = RetrievalTrace(
        query="24/7 enterprise support",
        strategy="hybrid_reranked",
        candidates=docs,
    )

    text = debugger.format_trace_text(trace)
    assert "RETRIEVAL TRACE: '24/7 enterprise support'" in text
    assert "Strategy: hybrid_reranked" in text
    assert "support.txt" in text
    assert "#1" in text
    assert "#3" in text
    assert "3.140" in text
    assert "Enterprise accounts receive 24/7 dedicated support." in text


def test_debugger_format_trace_text_empty():
    debugger = RetrievalDebugger()
    trace = RetrievalTrace(query="empty query", strategy="dense", candidates=[])
    text = debugger.format_trace_text(trace)
    assert "Candidates: 0" in text


def test_debugger_format_markdown():
    debugger = RetrievalDebugger()

    docs = [
        RetrievalResult(
            document_id="d1",
            content="Some text",
            metadata={"source": "pricing.txt", "department": "Sales"},
            score=0.85,
            rank=1,
            original_rank=2,
            dense_score=0.8,
            bm25_score=10.0,
            fusion_score=0.015,
            rerank_score=2.5,
        )
    ]

    filt = MetadataFilter(department="Sales")
    trace = RetrievalTrace(
        query="pricing details",
        strategy="hybrid_reranked",
        query_variants=["cost plans"],
        filters=filt,
        candidates=docs,
    )

    md = debugger.format_markdown(trace)
    assert "### Retrieval Trace: `pricing details`" in md
    assert "**Strategy**: `hybrid_reranked`" in md
    assert "`cost plans`" in md
    assert "| Rank | Orig | Source / Doc ID | Dept | Dense | BM25 | RRF | Rerank |" in md
    assert "| #1 | #2 | `pricing.txt` | Sales | 0.800 | 10.00 | 0.0150 | +2.500 |" in md


def test_debugger_compare_text():
    debugger = RetrievalDebugger()

    t1 = RetrievalTrace(
        query="refund time",
        strategy="dense",
        candidates=[
            RetrievalResult(document_id="d1", content="t1", metadata={"source": "refund_v1.txt"}, score=0.9, rank=1)
        ],
    )
    t2 = RetrievalTrace(
        query="refund time",
        strategy="bm25",
        candidates=[
            RetrievalResult(document_id="d2", content="t2", metadata={"source": "refund_v2.txt"}, score=12.0, rank=1)
        ],
    )

    comp = debugger.format_comparison_text([t1, t2])
    assert "STRATEGY COMPARISON: 'refund time'" in comp
    assert "dense" in comp
    assert "bm25" in comp
    assert "refund_v1.txt" in comp
    assert "refund_v2.txt" in comp


def test_debugger_compare_empty():
    debugger = RetrievalDebugger()
    assert debugger.format_comparison_text([]) == "No traces to compare."


def test_debugger_rich_print_calls():
    mock_console = MagicMock()
    debugger = RetrievalDebugger(console=mock_console)

    docs = [
        RetrievalResult(
            document_id="d1",
            content="Some compressed content.",
            metadata={"source": "test.txt", "department": "Engineering", "compression_ratio": 0.6, "original_content": "Longer text here."},
            score=0.95,
            rank=1,
            original_rank=2,
            dense_score=0.9,
            bm25_score=11.5,
            fusion_score=0.016,
            rerank_score=2.8,
        )
    ]

    trace = RetrievalTrace(
        query="test query",
        strategy="hybrid_reranked_compressed",
        candidates=docs,
    )

    debugger.print_trace(trace)
    assert mock_console.print.called

    # Test compare
    debugger.compare_traces([trace])
    assert mock_console.print.called
