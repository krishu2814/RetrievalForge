"""
Unified RAG Pipeline module implementing the Strategy Pattern.
Provides a single, cohesive interface to execute, benchmark, and trace
all retrieval strategies across the RetrievalForge lab.
"""

import logging
from typing import List, Optional, Dict, Any, Union

from app.config import settings
from app.models.schemas import RetrievalResult, MetadataFilter, RetrievalTrace
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import BM25Retriever
from app.retrieval.mmr import MMRRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.multi_query import MultiQueryRetriever
from app.retrieval.reranker import CrossEncoderReranker, RerankedRetriever
from app.retrieval.compression import ContextualCompressor

logger = logging.getLogger(__name__)

SUPPORTED_STRATEGIES = [
    "dense",
    "bm25",
    "sparse",
    "mmr",
    "hybrid",
    "multi_query",
    "dense_reranked",
    "bm25_reranked",
    "hybrid_reranked",
    "multi_query_reranked",
    "hybrid_reranked_compressed",
]


def format_context(candidates: List[RetrievalResult]) -> str:
    """
    Formats a list of retrieved documents into a clean, structured context block
    suitable for feeding directly into LLM prompts.
    """
    if not candidates:
        return "No relevant documents found."

    sections = []
    for i, doc in enumerate(candidates, start=1):
        source = doc.metadata.get("source", doc.document_id)
        department = doc.metadata.get("department", "General")
        header = f"[Document {i}] Source: {source} | Department: {department} | Retriever: {doc.retriever}"
        sections.append(f"{header}\n{doc.content.strip()}")

    return "\n\n---\n\n".join(sections)


class RAGPipeline:
    """
    Unified RAG Pipeline employing the Strategy Pattern.
    Allows seamlessly switching between dense, sparse, MMR, hybrid,
    multi-query, reranked, and compressed retrieval strategies.
    """

    def __init__(
        self,
        strategy: str = "hybrid_reranked",
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        mmr_retriever: Optional[MMRRetriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        multi_query_retriever: Optional[MultiQueryRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        compressor: Optional[ContextualCompressor] = None,
    ):
        """
        Initializes the pipeline with a default strategy and optional dependency injection.
        Retriever instances are lazy-loaded on demand if not injected.
        """
        self.set_strategy(strategy)

        # Injected or lazily-loaded components
        self._dense_retriever = dense_retriever
        self._bm25_retriever = bm25_retriever
        self._mmr_retriever = mmr_retriever
        self._hybrid_retriever = hybrid_retriever
        self._multi_query_retriever = multi_query_retriever
        self._reranker = reranker
        self._compressor = compressor

    @staticmethod
    def list_strategies() -> List[str]:
        """Returns the list of all supported retrieval strategies."""
        return list(SUPPORTED_STRATEGIES)

    def set_strategy(self, strategy: str) -> None:
        """Sets the active retrieval strategy."""
        normalized = strategy.strip().lower()
        if normalized not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Supported strategies: {SUPPORTED_STRATEGIES}"
            )
        self.strategy = normalized

    # =========================================================================
    # Lazy Loaders for Components
    # =========================================================================

    def get_dense_retriever(self) -> DenseRetriever:
        if self._dense_retriever is None:
            self._dense_retriever = DenseRetriever()
        return self._dense_retriever

    def get_bm25_retriever(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever()
        return self._bm25_retriever

    def get_mmr_retriever(self) -> MMRRetriever:
        if self._mmr_retriever is None:
            self._mmr_retriever = MMRRetriever()
        return self._mmr_retriever

    def get_hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                dense_retriever=self.get_dense_retriever(),
                bm25_retriever=self.get_bm25_retriever(),
            )
        return self._hybrid_retriever

    def get_multi_query_retriever(self) -> MultiQueryRetriever:
        if self._multi_query_retriever is None:
            self._multi_query_retriever = MultiQueryRetriever(
                base_retriever=self.get_hybrid_retriever()
            )
        return self._multi_query_retriever

    def get_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def get_compressor(self) -> ContextualCompressor:
        if self._compressor is None:
            self._compressor = ContextualCompressor()
        return self._compressor

    # =========================================================================
    # Strategy Dispatcher
    # =========================================================================

    def _execute_strategy(
        self,
        strategy: str,
        query: str,
        top_k: int,
        filters: Optional[MetadataFilter] = None,
    ) -> List[RetrievalResult]:
        """
        Dispatches query retrieval to the appropriate strategy implementation.
        """
        if strategy == "dense":
            return self.get_dense_retriever().retrieve(query=query, top_k=top_k, filters=filters)

        elif strategy in ("bm25", "sparse"):
            return self.get_bm25_retriever().retrieve(query=query, top_k=top_k, filters=filters)

        elif strategy == "mmr":
            return self.get_mmr_retriever().retrieve(query=query, top_k=top_k, filters=filters)

        elif strategy == "hybrid":
            return self.get_hybrid_retriever().retrieve(query=query, top_k=top_k, filters=filters)

        elif strategy == "multi_query":
            return self.get_multi_query_retriever().retrieve(query=query, top_k=top_k, filters=filters)

        elif strategy == "dense_reranked":
            pipeline = RerankedRetriever(
                base_retriever=self.get_dense_retriever(),
                reranker=self.get_reranker(),
            )
            return pipeline.retrieve(
                query=query,
                top_k=top_k,
                candidate_pool_k=settings.CANDIDATE_POOL_K,
                filters=filters,
            )

        elif strategy == "bm25_reranked":
            pipeline = RerankedRetriever(
                base_retriever=self.get_bm25_retriever(),
                reranker=self.get_reranker(),
            )
            return pipeline.retrieve(
                query=query,
                top_k=top_k,
                candidate_pool_k=settings.CANDIDATE_POOL_K,
                filters=filters,
            )

        elif strategy == "hybrid_reranked":
            pipeline = RerankedRetriever(
                base_retriever=self.get_hybrid_retriever(),
                reranker=self.get_reranker(),
            )
            return pipeline.retrieve(
                query=query,
                top_k=top_k,
                candidate_pool_k=settings.CANDIDATE_POOL_K,
                filters=filters,
            )

        elif strategy == "multi_query_reranked":
            pipeline = RerankedRetriever(
                base_retriever=self.get_multi_query_retriever(),
                reranker=self.get_reranker(),
            )
            return pipeline.retrieve(
                query=query,
                top_k=top_k,
                candidate_pool_k=settings.CANDIDATE_POOL_K,
                filters=filters,
            )

        elif strategy == "hybrid_reranked_compressed":
            pipeline = RerankedRetriever(
                base_retriever=self.get_hybrid_retriever(),
                reranker=self.get_reranker(),
            )
            reranked_docs = pipeline.retrieve(
                query=query,
                top_k=top_k,
                candidate_pool_k=settings.CANDIDATE_POOL_K,
                filters=filters,
            )
            return self.get_compressor().compress_documents(query=query, documents=reranked_docs)

        else:
            raise ValueError(f"Unhandled strategy: {strategy}")

    # =========================================================================
    # Public Execution & Diagnostic Trace
    # =========================================================================

    def run(
        self,
        query: str,
        strategy: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[MetadataFilter] = None,
        compress: Optional[bool] = None,
    ) -> RetrievalTrace:
        """
        Runs retrieval using the configured or overridden strategy.
        Returns an end-to-end RetrievalTrace containing candidates,
        query variants, and the formatted context string.
        """
        active_strategy = strategy.strip().lower() if strategy else self.strategy
        if active_strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{active_strategy}'. Supported strategies: {SUPPORTED_STRATEGIES}"
            )

        k = top_k if top_k is not None else settings.DEFAULT_TOP_K

        # Execute the chosen retrieval strategy
        candidates = self._execute_strategy(
            strategy=active_strategy,
            query=query,
            top_k=k,
            filters=filters,
        )

        # Apply optional contextual compression if requested explicitly and not already compressed
        if compress is True and not active_strategy.endswith("_compressed"):
            candidates = self.get_compressor().compress_documents(query=query, documents=candidates)

        # Collect any query variants tracked in candidates
        variants = []
        for doc in candidates:
            if doc.query_variant and doc.query_variant not in variants:
                variants.append(doc.query_variant)

        final_context = format_context(candidates)

        return RetrievalTrace(
            query=query,
            strategy=active_strategy,
            query_variants=variants,
            filters=filters,
            candidates=candidates,
            final_context=final_context,
        )

    def get_context(
        self,
        query: str,
        strategy: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> str:
        """
        Convenience method that returns just the formatted context string
        ready for insertion into an LLM prompt.
        """
        trace = self.run(query=query, strategy=strategy, top_k=top_k, filters=filters)
        return trace.final_context


def get_pipeline(strategy: str = "hybrid_reranked") -> RAGPipeline:
    """
    Factory function to obtain a configured RAGPipeline instance.
    """
    return RAGPipeline(strategy=strategy)
