"""
Reranking module using Cross-Encoder models.
Implements two-stage retrieval: Retrieve Many (Top 20) -> Rerank Carefully (Top 5).
"""

import logging
from typing import List, Optional, Union
import numpy as np

from app.config import settings
from app.retrieval.hybrid import HybridRetriever
from app.models.schemas import RetrievalResult, MetadataFilter

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-Encoder reranker.
    Performs full cross-attention between query and candidate documents
    to compute fine-grained relevance scores.
    """

    def __init__(self, model_name: Optional[str] = None, cross_encoder=None):
        """
        Initializes the reranker.
        Allows passing a custom or mock cross_encoder instance.
        """
        self.model_name = model_name or settings.RERANKER_MODEL
        self._cross_encoder = cross_encoder

    def _get_model(self):
        """
        Lazy-loads the CrossEncoder model when first needed.
        """
        if self._cross_encoder is None:
            logger.info(f"Loading CrossEncoder model: {self.model_name}...")
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self.model_name)
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        documents: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Reranks a list of candidate documents against the query.
        Updates original_rank, rerank_score, score, and final_rank on each result.
        """
        if not query or not query.strip() or not documents:
            return []

        k = top_k or settings.RERANK_TOP_K
        model = self._get_model()

        # Prepare (query, document_content) pairs for cross-attention
        pairs = [[query, doc.content] for doc in documents]

        # Compute cross-encoder relevance scores
        raw_scores = model.predict(pairs)

        # Handle scalar output for single document or list/array
        if isinstance(raw_scores, (int, float, np.floating)):
            scores = [float(raw_scores)]
        else:
            scores = [float(s) for s in raw_scores]

        # Pair each document with its rerank score and original rank
        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_docs.append((doc, score))

        # Sort documents by rerank score descending
        scored_docs.sort(key=lambda item: item[1], reverse=True)

        reranked_results: List[RetrievalResult] = []
        for new_rank, (doc, score) in enumerate(scored_docs[:k], start=1):
            rerank_score_val = round(score, 4)

            # Build a new clean result with updated diagnostic ranks & scores
            updated_doc = RetrievalResult(
                document_id=doc.document_id,
                content=doc.content,
                metadata=doc.metadata,
                score=rerank_score_val,
                rank=new_rank,
                retriever=f"{doc.retriever}_reranked" if not doc.retriever.endswith("_reranked") else doc.retriever,
                dense_score=doc.dense_score,
                bm25_score=doc.bm25_score,
                fusion_score=doc.fusion_score,
                rerank_score=rerank_score_val,
                original_rank=doc.original_rank or doc.rank,
                final_rank=new_rank,
                query_variant=doc.query_variant
            )
            reranked_results.append(updated_doc)

        return reranked_results


class RerankedRetriever:
    """
    Complete Two-Stage Retrieval Pipeline:
    Stage 1: Retrieve candidate pool (default: 20 candidates) via base retriever
    Stage 2: Cross-Encoder reranking down to top_k (default: 5 candidates)
    """

    def __init__(
        self,
        base_retriever=None,
        reranker: Optional[Reranker] = None
    ):
        """
        Initializes two-stage retriever with a base retriever (default: HybridRetriever)
        and a Reranker instance.
        """
        self.base_retriever = base_retriever or HybridRetriever()
        self.reranker = reranker or Reranker()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_pool_k: Optional[int] = None,
        filters: Optional[Union[MetadataFilter, dict]] = None
    ) -> List[RetrievalResult]:
        """
        Executes two-stage retrieval:
        1. Retrieves wide candidate pool of size candidate_pool_k
        2. Reranks pool using Cross-Encoder and returns top_k
        """
        if not query or not query.strip():
            return []

        k = top_k or settings.RERANK_TOP_K
        pool_k = candidate_pool_k or settings.CANDIDATE_POOL_K

        # Stage 1: Candidate pool retrieval
        candidates = self.base_retriever.retrieve(
            query=query,
            top_k=pool_k,
            filters=filters
        )

        if not candidates:
            return []

        # Stage 2: Cross-Encoder reranking
        reranked = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=k
        )

        return reranked


# Alias for clarity in pipelines and documentation
CrossEncoderReranker = Reranker

