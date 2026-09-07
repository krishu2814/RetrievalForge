"""
Unit tests for the automated evaluation benchmark script (scripts/evaluate.py).
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from scripts.evaluate import (
    load_dataset,
    benchmark_strategy,
    generate_markdown_summary,
    build_arg_parser,
)
from app.models.schemas import RetrievalResult, RetrievalTrace


def test_evaluate_script_arg_parser():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert "dense" in args.strategies
    assert "hybrid" in args.strategies
    assert args.top_k == 5
    assert args.compress is False


def test_load_dataset_missing():
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_dataset_file.json")


def test_load_dataset_valid():
    from app.config import settings
    dataset = load_dataset(settings.EVAL_DATASET_PATH)
    assert isinstance(dataset, list)
    assert len(dataset) >= 20


def test_benchmark_strategy_with_mock():
    mock_pipeline = MagicMock()
    mock_doc = RetrievalResult(
        document_id="api-documentation",
        content="Error codes",
        metadata={"source": "api_documentation.txt"},
        score=0.9,
        rank=1,
    )
    mock_trace = RetrievalTrace(
        query="test query",
        strategy="dense",
        candidates=[mock_doc],
    )
    mock_pipeline.run.return_value = mock_trace

    mini_dataset = [
        {
            "id": "q1",
            "query": "What is ERR_AUTH_TIMEOUT_504?",
            "category": "exact_keyword",
            "expected_sources": ["api_documentation.txt"],
            "expected_doc_ids": ["api-documentation"],
            "ground_truth_answer": "Timeout error",
            "relevant_keywords": ["504"],
        }
    ]

    res = benchmark_strategy(
        strategy="dense",
        dataset=mini_dataset,
        pipeline=mock_pipeline,
        top_k=3,
        compress=False,
    )

    assert res["strategy"] == "dense"
    assert res["overall"]["mean_hit_at_1"] == 1.0
    assert res["overall"]["mean_mrr_at_5"] == 1.0
    assert "exact_keyword" in res["by_category"]


def test_generate_markdown_summary():
    sample_results = {
        "dense": {
            "overall": {
                "mean_hit_at_1": 0.8,
                "mean_hit_at_3": 0.9,
                "mean_hit_at_5": 0.95,
                "mean_recall_at_5": 0.9,
                "mean_mrr_at_5": 0.85,
                "mean_ndcg_at_5": 0.88,
                "mean_latency_ms": 15.2,
            },
            "by_category": {
                "exact_keyword": {"mrr_at_5": 0.75},
                "conceptual": {"mrr_at_5": 0.95},
            },
        },
        "hybrid": {
            "overall": {
                "mean_hit_at_1": 0.9,
                "mean_hit_at_3": 0.95,
                "mean_hit_at_5": 1.0,
                "mean_recall_at_5": 0.98,
                "mean_mrr_at_5": 0.93,
                "mean_ndcg_at_5": 0.94,
                "mean_latency_ms": 22.4,
            },
            "by_category": {
                "exact_keyword": {"mrr_at_5": 1.0},
                "conceptual": {"mrr_at_5": 0.95},
            },
        },
    }

    md = generate_markdown_summary(sample_results)
    assert "## 🏆 RetrievalForge Benchmark Results" in md
    assert "**dense**" in md
    assert "**hybrid**" in md
    assert "| `exact_keyword` | 0.750 | 1.000 |" in md
