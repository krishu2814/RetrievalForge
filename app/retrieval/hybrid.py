"""
Hybrid search module combining Dense vector search and Sparse BM25 search
using Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import List, Optional, Union, Dict, Any

from app.config import settings
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever
from app.models.schemas import RetrievalResult, MetadataFilter

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: List[RetrievalResult],
    sparse_results: List[RetrievalResult],
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    top_k: int = 5
) -> List[RetrievalResult]:
    """
    Combines two ranked lists of RetrievalResult using Reciprocal Rank Fusion (RRF).

    Formula for each document chunk d:
        RRF_Score(d) = dense_weight * (1 / (rrf_k + dense_rank)) +
                       sparse_weight * (1 / (rrf_k + sparse_rank))

    Rank-based fusion prevents score scale mismatch between bounded cosine
    distances and unbounded BM25 scores.
    """
    fused_scores: Dict[str, float] = {}
    doc_store: Dict[str, RetrievalResult] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}

    # 1. Process dense candidates
    for rank, res in enumerate(dense_results, start=1):
        chunk_key = res.metadata.get("chunk_id") or f"{res.document_id}_{res.content[:30]}"
        dense_ranks[chunk_key] = rank
        rrf_val = dense_weight * (1.0 / (rrf_k + rank))
        fused_scores[chunk_key] = fused_scores.get(chunk_key, 0.0) + rrf_val
        doc_store[chunk_key] = res

    # 2. Process sparse candidates
    for rank, res in enumerate(sparse_results, start=1):
        chunk_key = res.metadata.get("chunk_id") or f"{res.document_id}_{res.content[:30]}"
        sparse_ranks[chunk_key] = rank
        rrf_val = sparse_weight * (1.0 / (rrf_k + rank))
        fused_scores[chunk_key] = fused_scores.get(chunk_key, 0.0) + rrf_val

        # If already present in dense results, merge the bm25 score
        if chunk_key in doc_store:
            doc_store[chunk_key].bm25_score = res.bm25_score
        else:
            doc_store[chunk_key] = res

    # 3. Sort chunks by descending fusion score
    sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)

    final_results: List[RetrievalResult] = []
    for final_rank, chunk_key in enumerate(sorted_keys[:top_k], start=1):
        item = doc_store[chunk_key]
        score = round(fused_scores[chunk_key], 6)

        merged = RetrievalResult(
            document_id=item.document_id,
            content=item.content,
            metadata=item.metadata,
            score=score,
            rank=final_rank,
            retriever="hybrid",
            dense_score=item.dense_score,
            bm25_score=item.bm25_score,
            fusion_score=score,
            original_rank=dense_ranks.get(chunk_key) or sparse_ranks.get(chunk_key),
            final_rank=final_rank
        )
        final_results.append(merged)

    return final_results


class HybridRetriever:
    """
    Hybrid retriever that runs Dense and BM25 search in tandem,
    then combines and deduplicates candidates using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
    ):
        """
        Initializes the hybrid retriever with dense and sparse instances.
        Supports either sparse_retriever or bm25_retriever parameter.
        """
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or bm25_retriever or BM25Retriever()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Union[MetadataFilter, dict]] = None,
        rrf_k: Optional[int] = None,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Executes hybrid retrieval:
        1. Queries Dense retriever for candidate_pool_k chunks
        2. Queries Sparse BM25 retriever for candidate_pool_k chunks
        3. Applies Reciprocal Rank Fusion (RRF)
        4. Returns top_k deduplicated, re-ranked candidates
        """
        if not query or not query.strip():
            return []

        k = top_k or settings.DEFAULT_TOP_K
        pool_k = settings.CANDIDATE_POOL_K
        r_k = rrf_k or settings.RRF_K
        d_weight = dense_weight if dense_weight is not None else settings.HYBRID_DENSE_WEIGHT
        s_weight = sparse_weight if sparse_weight is not None else settings.HYBRID_SPARSE_WEIGHT

        # Retrieve wider candidate pools from both engines
        dense_candidates = self.dense_retriever.retrieve(query=query, top_k=pool_k, filters=filters)
        sparse_candidates = self.sparse_retriever.retrieve(query=query, top_k=pool_k, filters=filters)

        # Merge with Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            rrf_k=r_k,
            dense_weight=d_weight,
            sparse_weight=s_weight,
            top_k=k
        )

        return fused_results
