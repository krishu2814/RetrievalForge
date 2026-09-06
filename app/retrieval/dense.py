"""
Dense vector retriever module.
Performs semantic similarity search over FAISS vector index with metadata filtering.
"""

import logging
from typing import List, Optional, Union
from app.config import settings
from app.ingestion.indexing import load_dense_index
from app.models.schemas import RetrievalResult, MetadataFilter

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense vector retriever using FAISS and sentence embeddings.
    Provides semantic similarity search and post-retrieval metadata filtering.
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
        filters: Optional[Union[MetadataFilter, dict]] = None
    ) -> List[RetrievalResult]:
        """
        Performs dense vector similarity search for the given query string.
        Applies metadata filtering if provided and returns standardized RetrievalResult objects.
        """
        if not query or not query.strip():
            return []

        k = top_k or settings.DEFAULT_TOP_K

        # Normalize dictionary filter into a MetadataFilter object
        if isinstance(filters, dict):
            filter_obj = MetadataFilter(custom=filters)
        else:
            filter_obj = filters

        # Fetch extra candidates if filtering so we have enough matching results
        fetch_k = max(k * 4, settings.CANDIDATE_POOL_K) if filter_obj else k

        # FAISS search returning (Document, distance)
        docs_and_distances = self.vector_store.similarity_search_with_score(query, k=fetch_k)

        results: List[RetrievalResult] = []
        rank = 1

        for doc, distance in docs_and_distances:
            metadata = doc.metadata or {}

            # Check metadata filter
            if filter_obj and not filter_obj.matches(metadata):
                continue

            # Convert L2 distance to normalized similarity score [0.0, 1.0]
            # Higher score = more semantically similar
            similarity_score = round(1.0 / (1.0 + float(distance)), 4)

            doc_id = metadata.get("document_id") or metadata.get("chunk_id", f"chunk_{rank}")

            result = RetrievalResult(
                document_id=doc_id,
                content=doc.page_content,
                metadata=metadata,
                score=similarity_score,
                rank=rank,
                retriever="dense",
                dense_score=similarity_score,
                original_rank=rank
            )
            results.append(result)
            rank += 1

            if len(results) >= k:
                break

        return results
