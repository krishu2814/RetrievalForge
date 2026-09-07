"""
Unit tests for Information Retrieval (IR) evaluation metrics.
"""

import pytest
import math

from app.evaluation.metrics import (
    normalize_identifier,
    is_document_match,
    compute_recall_at_k,
    compute_hit_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    evaluate_query_retrieval,
    aggregate_metrics,
    QueryMetrics,
)


def test_normalize_identifier():
    assert normalize_identifier("refund_policy_v2.txt") == "refund-policy-v2"
    assert normalize_identifier("API_DOCUMENTATION.TXT") == "api-documentation"
    assert normalize_identifier("data-privacy") == "data-privacy"


def test_is_document_match():
    assert is_document_match("refund_policy_v2.txt", "refund-policy-v2") is True
    assert is_document_match("refund-policy-v2", "refund_policy_v2.txt") is True
    assert is_document_match("refund_policy_v1.txt", "refund_policy_v2.txt") is False


def test_compute_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    
    # Both expected are in top 5
    assert compute_recall_at_k(retrieved, ["doc1", "doc3"], k=5) == 1.0

    # Only 1 of 2 expected is in top 2
    assert compute_recall_at_k(retrieved, ["doc1", "doc3"], k=2) == 0.5

    # None in top 2
    assert compute_recall_at_k(retrieved, ["doc4", "doc5"], k=2) == 0.0


def test_compute_hit_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    assert compute_hit_at_k(retrieved, ["doc2"], k=1) == 0.0
    assert compute_hit_at_k(retrieved, ["doc2"], k=2) == 1.0
    assert compute_hit_at_k(retrieved, ["doc_missing"], k=3) == 0.0


def test_compute_mrr_at_k():
    retrieved = ["docA", "docB", "docC", "docD"]

    # Rank 1 match -> MRR = 1/1 = 1.0
    assert compute_mrr_at_k(retrieved, ["docA"], k=5) == 1.0

    # Rank 2 match -> MRR = 1/2 = 0.5
    assert compute_mrr_at_k(retrieved, ["docB"], k=5) == 0.5

    # Rank 4 match -> MRR = 1/4 = 0.25
    assert compute_mrr_at_k(retrieved, ["docD"], k=5) == 0.25

    # Rank 4 with cutoff k=3 -> MRR = 0.0
    assert compute_mrr_at_k(retrieved, ["docD"], k=3) == 0.0


def test_compute_ndcg_at_k():
    retrieved = ["docA", "docB", "docC", "docD"]

    # If the only relevant doc is at rank 1, NDCG is 1.0
    assert compute_ndcg_at_k(retrieved, ["docA"], k=3) == 1.0

    # If the relevant doc is at rank 2:
    # DCG = 1 / log2(3) = 0.630929
    # IDCG = 1 / log2(2) = 1.0
    # NDCG = 0.6309
    ndcg = compute_ndcg_at_k(retrieved, ["docB"], k=3)
    assert math.isclose(ndcg, 0.6309, abs_tol=0.001)

    # If no relevant doc retrieved
    assert compute_ndcg_at_k(retrieved, ["missing"], k=3) == 0.0


def test_compute_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    assert compute_precision_at_k(retrieved, ["doc1", "doc2"], k=5) == 0.4
    assert compute_precision_at_k(retrieved, ["doc1"], k=5) == 0.2
    assert compute_precision_at_k(retrieved, ["missing"], k=5) == 0.0


def test_evaluate_query_retrieval():
    metrics = evaluate_query_retrieval(
        query_id="q1",
        query="test query",
        category="exact_keyword",
        retrieved_items=["api_documentation.txt", "other.txt"],
        expected_targets=["api-documentation"],
        latency_ms=45.2,
    )

    assert isinstance(metrics, QueryMetrics)
    assert metrics.hit_at_1 == 1.0
    assert metrics.mrr_at_5 == 1.0
    assert metrics.ndcg_at_5 == 1.0
    assert metrics.latency_ms == 45.2


def test_aggregate_metrics():
    m1 = QueryMetrics(
        query_id="q1",
        query="q1",
        category="exact_keyword",
        hit_at_1=1.0,
        hit_at_3=1.0,
        hit_at_5=1.0,
        recall_at_5=1.0,
        mrr_at_5=1.0,
        ndcg_at_5=1.0,
        precision_at_5=0.2,
        latency_ms=20.0,
    )
    m2 = QueryMetrics(
        query_id="q2",
        query="q2",
        category="conceptual",
        hit_at_1=0.0,
        hit_at_3=1.0,
        hit_at_5=1.0,
        recall_at_5=1.0,
        mrr_at_5=0.5,
        ndcg_at_5=0.6309,
        precision_at_5=0.2,
        latency_ms=30.0,
    )

    agg = aggregate_metrics([m1, m2])
    overall = agg["overall"]

    assert overall["total_queries"] == 2
    assert overall["mean_hit_at_1"] == 0.5
    assert overall["mean_hit_at_3"] == 1.0
    assert overall["mean_mrr_at_5"] == 0.75
    assert overall["mean_latency_ms"] == 25.0
    assert "exact_keyword" in agg["by_category"]
    assert "conceptual" in agg["by_category"]
