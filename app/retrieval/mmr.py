"""
Maximal Marginal Relevance (MMR) retriever module.
Balances query relevance and document diversity to prevent redundant chunks.
"""

import logging
from typing import List, Optional, Union
from app.config import settings
from app.ingestion.indexing import load_dense_index
from app.models.schemas import RetrievalResult, MetadataFilter

logger = logging.getLogger(__name__)


class MMRRetriever:
    """
    Retriever using Maximal Marginal Relevance (MMR) over FAISS vector store.
    Trades off semantic relevance vs candidate diversity using lambda_mult.
    """

    def __init__(self, vector_store=None, index_path: Optional[str] = None):
        """
        Initializes the retriever with an existing vector store or loads from disk.
        """
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = load_dense_index(index_path)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filters: Optional[Union[MetadataFilter, dict]] = None
    ) -> List[RetrievalResult]:
        """
        Performs MMR search balancing semantic relevance with candidate diversity.
        
        Args:
            query: The user query string.
            top_k: Number of final diverse documents to return.
            fetch_k: Initial candidate pool size to fetch before selecting diverse chunks.
            lambda_mult: Diversity factor (1.0 = pure similarity, 0.0 = maximal diversity).
            filters: Optional metadata filters.
        """
        if not query or not query.strip():
            return []

        k = top_k or settings.DEFAULT_TOP_K
        f_k = fetch_k or settings.MMR_FETCH_K
        l_mult = lambda_mult if lambda_mult is not None else settings.MMR_LAMBDA

        # Ensure fetch_k is at least as large as k
        if f_k < k:
            f_k = k * 2

        # Normalize dictionary filter into a MetadataFilter object
        if isinstance(filters, dict):
            filter_obj = MetadataFilter(custom=filters)
        else:
            filter_obj = filters

        # Fetch extra candidates if filtering is active
        actual_fetch_k = f_k * 2 if filter_obj else f_k
        query_k = actual_fetch_k if filter_obj else k

        # FAISS built-in MMR search
        docs = self.vector_store.max_marginal_relevance_search(
            query=query,
            k=query_k,
            fetch_k=actual_fetch_k,
            lambda_mult=l_mult
        )

        results: List[RetrievalResult] = []
        rank = 1

        for doc in docs:
            metadata = doc.metadata or {}

            # Apply metadata filter
            if filter_obj and not filter_obj.matches(metadata):
                continue

            doc_id = metadata.get("document_id") or metadata.get("chunk_id", f"chunk_{rank}")

            # Assign a rank-proportional relative score for ranking consistency
            score = round(max(0.1, 1.0 - (rank - 1) * (0.5 / max(k, 1))), 4)

            result = RetrievalResult(
                document_id=doc_id,
                content=doc.page_content,
                metadata=metadata,
                score=score,
                rank=rank,
                retriever="mmr",
                original_rank=rank
            )
            results.append(result)
            rank += 1

            if len(results) >= k:
                break

        return results
