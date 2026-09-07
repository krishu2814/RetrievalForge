"""
Contextual compression module.
Extracts only the query-relevant sentences/passages from retrieved chunks,
reducing token bloat and eliminating distractor content before sending context to the LLM.
"""

import re
import logging
from typing import List, Optional, Tuple
import numpy as np

from app.config import settings
from app.ingestion.indexing import get_embedding_model
from app.models.schemas import RetrievalResult

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Splits a text block into individual sentences and bullet lines.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences = []
    for line in lines:
        parts = re.split(r'(?<=[.!?])\s+', line)
        for p in parts:
            p_clean = p.strip()
            if p_clean:
                sentences.append(p_clean)
    return sentences if sentences else [text.strip()]


class ContextualCompressor:
    """
    Compresses retrieved document chunks by extracting only query-relevant sentences.
    Uses sentence-level embedding similarity to prune distractor text.
    """

    def __init__(self, embedding_model=None, similarity_threshold: Optional[float] = None):
        """
        Initializes ContextualCompressor.
        Allows injecting an existing embedding model or custom similarity threshold.
        """
        self._embedding_model = embedding_model
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(settings, "COMPRESSION_SIMILARITY_THRESHOLD", 0.4)
        )

    def _get_embedding_model(self):
        """
        Lazy-loads embedding model when needed.
        """
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def compress_text(self, query: str, text: str) -> Tuple[str, float]:
        """
        Compresses a single chunk of text against the query.
        Returns a tuple of (compressed_text, compression_ratio).
        """
        sentences = split_into_sentences(text)
        if len(sentences) <= 1:
            return text, 1.0

        embedder = self._get_embedding_model()

        # Embed query and all candidate sentences
        query_vec = np.array(embedder.embed_query(query))
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        sentence_vecs = np.array(embedder.embed_documents(sentences))
        norms = np.linalg.norm(sentence_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        sentence_vecs = sentence_vecs / norms

        # Cosine similarity between query and each sentence
        sim_scores = np.dot(sentence_vecs, query_vec)

        # Retain sentences that meet or exceed the similarity threshold
        selected_indices = [
            i for i, score in enumerate(sim_scores)
            if score >= self.similarity_threshold
        ]

        # If none met the threshold, fallback to keeping the top 2 highest scoring sentences
        if not selected_indices:
            top_indices = np.argsort(sim_scores)[::-1][:2]
            selected_indices = sorted(top_indices.tolist())
        else:
            selected_indices.sort()

        selected_sentences = [sentences[i] for i in selected_indices]
        compressed_text = " ".join(selected_sentences)

        original_len = max(len(text), 1)
        ratio = round(len(compressed_text) / original_len, 3)

        return compressed_text, ratio

    def compress_documents(
        self,
        query: str,
        documents: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Compresses each document chunk in the candidate list.
        Preserves original_content and compression_ratio in metadata for debugging.
        """
        if not query or not query.strip() or not documents:
            return []

        compressed_results: List[RetrievalResult] = []

        for doc in documents:
            compressed_text, ratio = self.compress_text(query, doc.content)

            updated_metadata = dict(doc.metadata or {})
            updated_metadata["original_content"] = doc.content
            updated_metadata["compression_ratio"] = ratio

            compressed_doc = RetrievalResult(
                document_id=doc.document_id,
                content=compressed_text,
                metadata=updated_metadata,
                score=doc.score,
                rank=doc.rank,
                retriever=f"{doc.retriever}_compressed" if not doc.retriever.endswith("_compressed") else doc.retriever,
                dense_score=doc.dense_score,
                bm25_score=doc.bm25_score,
                fusion_score=doc.fusion_score,
                rerank_score=doc.rerank_score,
                original_rank=doc.original_rank,
                final_rank=doc.final_rank,
                query_variant=doc.query_variant
            )
            compressed_results.append(compressed_doc)

        return compressed_results
