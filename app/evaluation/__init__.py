"""Evaluation package for ground-truth benchmark datasets, metrics calculation, and strategy comparison."""

from app.evaluation.metrics import (
    normalize_identifier,
    is_document_match,
    compute_recall_at_k,
    compute_hit_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    QueryMetrics,
    evaluate_query_retrieval,
    aggregate_metrics,
)

__all__ = [
    "normalize_identifier",
    "is_document_match",
    "compute_recall_at_k",
    "compute_hit_at_k",
    "compute_mrr_at_k",
    "compute_ndcg_at_k",
    "compute_precision_at_k",
    "QueryMetrics",
    "evaluate_query_retrieval",
    "aggregate_metrics",
]
