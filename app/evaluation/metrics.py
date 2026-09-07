"""
Information Retrieval (IR) metrics module for RetrievalForge.
Implements Recall@K, Mean Reciprocal Rank (MRR@K), and
Normalized Discounted Cumulative Gain (NDCG@K) for benchmarking retrieval strategies.
"""

import math
from typing import List, Dict, Any, Set, Optional
from pydantic import BaseModel, Field


def normalize_identifier(identifier: str) -> str:
    """
    Normalizes document IDs and source filenames for robust matching.
    Strips file extension, lowercase, and converts underscores to hyphens.
    Example: 'refund_policy_v2.txt' -> 'refund-policy-v2'
    """
    cleaned = identifier.strip().lower()
    if cleaned.endswith(".txt"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.replace("_", "-")
    return cleaned


def is_document_match(retrieved_item: str, target_item: str) -> bool:
    """
    Checks if a retrieved document matches an expected target document,
    taking normalization into account.
    """
    norm_retrieved = normalize_identifier(retrieved_item)
    norm_target = normalize_identifier(target_item)
    return norm_retrieved == norm_target or norm_target in norm_retrieved or norm_retrieved in norm_target


def compute_recall_at_k(
    retrieved_items: List[str],
    expected_items: List[str],
    k: int = 5
) -> float:
    """
    Computes Recall@K: proportion of expected target documents retrieved in the top K.
    Formula: |retrieved[:K] ∩ expected| / |expected|
    """
    if not expected_items:
        return 1.0  # If no document was expected (e.g. negative query), recall is trivially satisfied

    top_k_items = retrieved_items[:k]
    hits = 0

    for expected in expected_items:
        if any(is_document_match(retrieved, expected) for retrieved in top_k_items):
            hits += 1

    return round(hits / len(expected_items), 4)


def compute_hit_at_k(
    retrieved_items: List[str],
    expected_items: List[str],
    k: int = 5
) -> float:
    """
    Computes Hit@K (Binary Recall): Returns 1.0 if at least one expected document
    appears in the top K retrieved results, otherwise 0.0.
    """
    if not expected_items:
        return 1.0

    top_k_items = retrieved_items[:k]
    for expected in expected_items:
        if any(is_document_match(retrieved, expected) for retrieved in top_k_items):
            return 1.0
    return 0.0


def compute_mrr_at_k(
    retrieved_items: List[str],
    expected_items: List[str],
    k: int = 5
) -> float:
    """
    Computes Reciprocal Rank (RR@K) for a single query.
    Formula: 1 / rank of the first relevant document in top K (0.0 if not found).
    """
    if not expected_items:
        return 1.0

    top_k_items = retrieved_items[:k]
    for rank_idx, item in enumerate(top_k_items, start=1):
        if any(is_document_match(item, expected) for expected in expected_items):
            return round(1.0 / rank_idx, 4)

    return 0.0


def compute_ndcg_at_k(
    retrieved_items: List[str],
    expected_items: List[str],
    k: int = 5
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.
    DCG@K = sum(rel_i / log2(i + 2))
    IDCG@K = ideal DCG where all relevant documents are ranked at the top.
    NDCG@K = DCG@K / IDCG@K
    """
    if not expected_items:
        return 1.0

    top_k_items = retrieved_items[:k]

    # Calculate DCG@K
    dcg = 0.0
    for idx, item in enumerate(top_k_items):
        if any(is_document_match(item, expected) for expected in expected_items):
            dcg += 1.0 / math.log2(idx + 2)  # idx=0 -> log2(2) = 1.0

    # Calculate Ideal DCG (IDCG@K)
    num_relevant = min(len(expected_items), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

    if idcg == 0.0:
        return 0.0

    return round(dcg / idcg, 4)


def compute_precision_at_k(
    retrieved_items: List[str],
    expected_items: List[str],
    k: int = 5
) -> float:
    """
    Computes Precision@K: proportion of top K retrieved documents that are relevant.
    Formula: |retrieved[:K] ∩ expected| / K
    """
    if k <= 0:
        return 0.0

    if not expected_items:
        return 0.0

    top_k_items = retrieved_items[:k]
    matched = 0
    for retrieved in top_k_items:
        if any(is_document_match(retrieved, expected) for expected in expected_items):
            matched += 1

    return round(matched / k, 4)


class QueryMetrics(BaseModel):
    """
    Detailed evaluation metrics for a single query.
    """
    query_id: str
    query: str
    category: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    recall_at_5: float
    mrr_at_5: float
    ndcg_at_5: float
    precision_at_5: float
    latency_ms: float = 0.0


def evaluate_query_retrieval(
    query_id: str,
    query: str,
    category: str,
    retrieved_items: List[str],
    expected_targets: List[str],
    latency_ms: float = 0.0,
) -> QueryMetrics:
    """
    Evaluates a single query run and returns a structured QueryMetrics object.
    """
    return QueryMetrics(
        query_id=query_id,
        query=query,
        category=category,
        hit_at_1=compute_hit_at_k(retrieved_items, expected_targets, k=1),
        hit_at_3=compute_hit_at_k(retrieved_items, expected_targets, k=3),
        hit_at_5=compute_hit_at_k(retrieved_items, expected_targets, k=5),
        recall_at_5=compute_recall_at_k(retrieved_items, expected_targets, k=5),
        mrr_at_5=compute_mrr_at_k(retrieved_items, expected_targets, k=5),
        ndcg_at_5=compute_ndcg_at_k(retrieved_items, expected_targets, k=5),
        precision_at_5=compute_precision_at_k(retrieved_items, expected_targets, k=5),
        latency_ms=round(latency_ms, 2),
    )


def aggregate_metrics(metrics_list: List[QueryMetrics]) -> Dict[str, Any]:
    """
    Aggregates metrics across all evaluated queries, computing overall means
    as well as category-by-category performance breakdowns.
    """
    if not metrics_list:
        return {}

    # Separate positive queries from negative queries for retrieval accuracy
    positive_queries = [m for m in metrics_list if m.category != "negative"]
    eval_pool = positive_queries if positive_queries else metrics_list

    n = len(eval_pool)
    overall = {
        "total_queries": len(metrics_list),
        "evaluated_positive_queries": n,
        "mean_hit_at_1": round(sum(m.hit_at_1 for m in eval_pool) / n, 4),
        "mean_hit_at_3": round(sum(m.hit_at_3 for m in eval_pool) / n, 4),
        "mean_hit_at_5": round(sum(m.hit_at_5 for m in eval_pool) / n, 4),
        "mean_recall_at_5": round(sum(m.recall_at_5 for m in eval_pool) / n, 4),
        "mean_mrr_at_5": round(sum(m.mrr_at_5 for m in eval_pool) / n, 4),
        "mean_ndcg_at_5": round(sum(m.ndcg_at_5 for m in eval_pool) / n, 4),
        "mean_precision_at_5": round(sum(m.precision_at_5 for m in eval_pool) / n, 4),
        "mean_latency_ms": round(sum(m.latency_ms for m in metrics_list) / len(metrics_list), 2),
    }

    # Category breakdowns
    by_category: Dict[str, Dict[str, float]] = {}
    categories = set(m.category for m in metrics_list)

    for cat in sorted(categories):
        cat_metrics = [m for m in metrics_list if m.category == cat]
        cat_n = len(cat_metrics)
        by_category[cat] = {
            "count": cat_n,
            "hit_at_1": round(sum(m.hit_at_1 for m in cat_metrics) / cat_n, 4),
            "hit_at_5": round(sum(m.hit_at_5 for m in cat_metrics) / cat_n, 4),
            "mrr_at_5": round(sum(m.mrr_at_5 for m in cat_metrics) / cat_n, 4),
            "ndcg_at_5": round(sum(m.ndcg_at_5 for m in cat_metrics) / cat_n, 4),
        }

    return {
        "overall": overall,
        "by_category": by_category,
    }
