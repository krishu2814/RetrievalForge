"""
Unit tests for the RetrievalForge CLI runner (main.py).
"""

import pytest
from unittest.mock import MagicMock, patch

from main import (
    build_arg_parser,
    print_banner,
    print_answer,
    execute_query,
    execute_comparison,
)
from app.models.schemas import RetrievalResult, RetrievalTrace, RAGResponse, MetadataFilter


def test_cli_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.query is None
    assert args.strategy == "hybrid_reranked"
    assert args.top_k == 5
    assert args.compare is False
    assert args.no_generate is False


def test_cli_parser_custom_args():
    parser = build_arg_parser()
    args = parser.parse_args([
        "-q", "What is the refund SLA?",
        "-s", "dense",
        "-k", "3",
        "--compare",
        "--compress",
        "--no-generate",
        "--department", "Billing",
        "--version", "2.0",
    ])
    assert args.query == "What is the refund SLA?"
    assert args.strategy == "dense"
    assert args.top_k == 3
    assert args.compare is True
    assert args.compress is True
    assert args.no_generate is True
    assert args.department == "Billing"
    assert args.version == "2.0"


def test_print_banner_and_answer():
    # Verify these display functions execute without throwing exceptions
    print_banner()

    resp = RAGResponse(
        answer="According to policy, 30 days.",
        query="Refund window?",
        sources=["refund_policy_v2.txt"],
        strategy="hybrid_reranked"
    )
    print_answer(resp)


def test_execute_query():
    mock_pipeline = MagicMock()
    mock_debugger = MagicMock()
    mock_generator = MagicMock()

    candidate = RetrievalResult(
        document_id="doc_1",
        content="Enterprise accounts receive dedicated support.",
        metadata={"source": "support.txt"},
        score=0.95,
        rank=1,
    )
    trace = RetrievalTrace(
        query="support query",
        strategy="dense",
        candidates=[candidate],
        final_context="Enterprise accounts receive dedicated support.",
    )
    mock_pipeline.run.return_value = trace

    expected_resp = RAGResponse(
        answer="Dedicated support is available.",
        query="support query",
        sources=["support.txt"],
        strategy="dense",
    )
    mock_generator.generate_from_trace.return_value = expected_resp

    # Execute with generation
    res = execute_query(
        query="support query",
        pipeline=mock_pipeline,
        debugger=mock_debugger,
        generator=mock_generator,
        strategy="dense",
        top_k=3,
        generate_answer=True,
    )

    mock_pipeline.run.assert_called_once_with(
        query="support query",
        strategy="dense",
        top_k=3,
        filters=None,
        compress=None,
    )
    mock_debugger.print_trace.assert_called_once_with(trace)
    mock_generator.generate_from_trace.assert_called_once_with(trace)
    assert res == expected_resp

    # Execute without generation
    res_no_gen = execute_query(
        query="support query",
        pipeline=mock_pipeline,
        debugger=mock_debugger,
        generator=mock_generator,
        strategy="dense",
        top_k=3,
        generate_answer=False,
    )
    assert "[Generation skipped" in res_no_gen.answer


def test_execute_comparison():
    mock_pipeline = MagicMock()
    mock_debugger = MagicMock()
    mock_generator = MagicMock()

    trace1 = RetrievalTrace(query="test", strategy="dense", candidates=[])
    trace2 = RetrievalTrace(query="test", strategy="bm25", candidates=[])
    mock_pipeline.run.side_effect = [trace1, trace2]

    execute_comparison(
        query="test",
        pipeline=mock_pipeline,
        debugger=mock_debugger,
        generator=mock_generator,
        strategies=["dense", "bm25"],
        generate_answer=False,
    )

    assert mock_pipeline.run.call_count == 2
    mock_debugger.compare_traces.assert_called_once_with([trace1, trace2])
