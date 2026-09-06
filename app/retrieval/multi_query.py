"""
Multi-query retriever module.
Generates multiple distinct queries from a single user prompt,
retrieves candidate pools for each query, deduplicates, and tracks provenance.
"""

import logging
from typing import List, Optional, Union, Dict, Any

from app.config import settings
from app.retrieval.dense import DenseRetriever
from app.retrieval.query_expansion import QueryExpander
from app.models.schemas import RetrievalResult, MetadataFilter

logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """
    Retriever that expands a user query into multiple perspectives,
    runs retrieval for each perspective, deduplicates candidates,
    and records which query variant retrieved each document.
    """

    def __init__(
        self,
        base_retriever=None,
        expander: Optional[QueryExpander] = None
    ):
        """
        Initializes MultiQueryRetriever with an underlying retriever (default: DenseRetriever)
        and a QueryExpander instance.
        """
        self.base_retriever = base_retriever or DenseRetriever()
        self.expander = expander or QueryExpander()
        self.last_generated_queries: List[str] = []

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        num_queries: Optional[int] = None,
        filters: Optional[Union[MetadataFilter, dict]] = None
    ) -> List[RetrievalResult]:
        """
        Executes multi-query retrieval:
        1. Generates query variations using QueryExpander
        2. Retrieves candidate pools for each variation
        3. Deduplicates results by chunk_id
        4. Tracks query_variant provenance
        5. Returns top_k ranked results
        """
        if not query or not query.strip():
            self.last_generated_queries = []
            return []

        k = top_k or settings.DEFAULT_TOP_K
        n_queries = num_queries or settings.MULTI_QUERY_COUNT

        # Generate query variants (guaranteed to include original query at index 0)
        query_variants = self.expander.expand(query, num_queries=n_queries)
        self.last_generated_queries = query_variants

        logger.info(f"MultiQuery generated {len(query_variants)} queries: {query_variants}")

        # Map unique chunk key -> (RetrievalResult, highest_score, list_of_queries)
        merged_candidates: Dict[str, RetrievalResult] = {}
        candidate_scores: Dict[str, float] = {}
        query_provenance: Dict[str, List[str]] = {}

        # Retrieve for each query variant
        for q in query_variants:
            candidates = self.base_retriever.retrieve(query=q, top_k=k, filters=filters)

            for cand in candidates:
                chunk_key = cand.metadata.get("chunk_id") or f"{cand.document_id}_{cand.content[:30]}"
                cand_score = cand.score if cand.score is not None else 0.0

                if chunk_key not in merged_candidates:
                    merged_candidates[chunk_key] = cand
                    candidate_scores[chunk_key] = cand_score
                    query_provenance[chunk_key] = [q]
                else:
                    # Keep highest score match
                    if cand_score > candidate_scores[chunk_key]:
                        candidate_scores[chunk_key] = cand_score
                        merged_candidates[chunk_key] = cand
                    if q not in query_provenance[chunk_key]:
                        query_provenance[chunk_key].append(q)

        # Sort deduplicated candidates by score descending
        sorted_keys = sorted(candidate_scores.keys(), key=lambda ck: candidate_scores[ck], reverse=True)

        final_results: List[RetrievalResult] = []
        for rank, chunk_key in enumerate(sorted_keys[:k], start=1):
            item = merged_candidates[chunk_key]
            queries_matched = query_provenance[chunk_key]

            result = RetrievalResult(
                document_id=item.document_id,
                content=item.content,
                metadata=item.metadata,
                score=candidate_scores[chunk_key],
                rank=rank,
                retriever="multi_query",
                dense_score=item.dense_score,
                bm25_score=item.bm25_score,
                fusion_score=item.fusion_score,
                original_rank=item.rank,
                final_rank=rank,
                query_variant=" | ".join(queries_matched)
            )
            final_results.append(result)

        return final_results
