"""
Sparse lexical retriever module using BM25.
Performs keyword-based inverted index retrieval with metadata filtering.
"""

import logging
from typing import List, Optional, Union, Dict, Any
import numpy as np

from app.config import settings
from app.ingestion.indexing import load_sparse_index, tokenize_text
from app.models.schemas import RetrievalResult, MetadataFilter

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Sparse lexical retriever using BM25Okapi.
    Excels at exact keyword matching, error codes, IDs, and technical terms.
    """

    def __init__(self, index_data: Optional[Dict[str, Any]] = None, index_path: Optional[str] = None):
        """
        Initializes the retriever with an existing index dictionary or loads from disk.
        """
        if index_data is not None:
            self.bm25 = index_data["bm25"]
            self.chunks = index_data["chunks"]
        else:
            loaded = load_sparse_index(index_path)
            self.bm25 = loaded["bm25"]
            self.chunks = loaded["chunks"]

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Union[MetadataFilter, dict]] = None
    ) -> List[RetrievalResult]:
        """
        Performs BM25 lexical search for the given query string.
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

        # Tokenize query into lowercase words
        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        # Calculate BM25 scores across all corpus chunks
        scores = self.bm25.get_scores(tokenized_query)

        # Sort indices in descending order of score
        ranked_indices = np.argsort(scores)[::-1]

        results: List[RetrievalResult] = []
        rank = 1

        for idx in ranked_indices:
            score = float(scores[idx])

            # Stop when score is 0.0 (no overlapping words)
            if score <= 0.0:
                break

            chunk = self.chunks[idx]
            metadata = chunk.get("metadata", {})

            # Apply metadata filter
            if filter_obj and not filter_obj.matches(metadata):
                continue

            doc_id = metadata.get("document_id") or metadata.get("chunk_id", f"chunk_{rank}")

            result = RetrievalResult(
                document_id=doc_id,
                content=chunk["content"],
                metadata=metadata,
                score=round(score, 4),
                rank=rank,
                retriever="bm25",
                bm25_score=round(score, 4),
                original_rank=rank
            )
            results.append(result)
            rank += 1

            if len(results) >= k:
                break

        return results
