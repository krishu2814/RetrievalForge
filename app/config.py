"""
Configuration Module for RetrievalForge.

================================================================================
WHAT IS THIS?
Centralized, strongly-typed configuration system powered by Pydantic Settings.
It reads from environment variables, `.env` files, and sets battle-tested defaults.

WHY IS IT NEEDED?
1. Prevents "magic numbers" scattered across retrievers, chunkers, and evaluators.
2. Ensures reproducible experiments: every retrieval hyperparameter (chunk size,
   overlap, k, lambda, fusion weights, reranker top-k) is defined in one place.
3. Supports switching models (e.g., local HuggingFace embeddings vs OpenAI,
   or different cross-encoders) without touching business logic.
4. Validates inputs at application boot time rather than failing midway through
   a benchmark.

WHAT HAPPENS INTERNALLY?
Pydantic parses environment variables (or `.env`), coerces types (e.g. float, int,
Path), and performs validation checks before providing an immutable `Settings` object.
================================================================================
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --------------------------------------------------------------------------
    # LLM & Generation Configuration
    # --------------------------------------------------------------------------
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key for OpenAI or OpenAI-compatible inference endpoints.",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI-compatible endpoints (e.g. Ollama, vLLM, Groq).",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model identifier used for generation and query transformation.",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Sampling temperature. 0.0 enforces deterministic, grounded answers.",
    )

    # --------------------------------------------------------------------------
    # Embedding Configuration
    # --------------------------------------------------------------------------
    embedding_provider: Literal["huggingface", "openai"] = Field(
        default="huggingface",
        description="Provider for dense vector embeddings ('huggingface' or 'openai').",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name/path for vector embeddings.",
    )

    # --------------------------------------------------------------------------
    # Reranker Configuration
    # --------------------------------------------------------------------------
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model identifier for deep semantic reranking.",
    )

    # --------------------------------------------------------------------------
    # Document Ingestion & Chunking Hyperparameters
    # --------------------------------------------------------------------------
    chunk_size: int = Field(
        default=500,
        gt=50,
        description="Maximum token/character length of each text chunk.",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        description="Overlap between consecutive chunks to maintain contextual boundaries.",
    )

    # --------------------------------------------------------------------------
    # Retrieval Hyperparameters
    # --------------------------------------------------------------------------
    default_top_k: int = Field(
        default=5,
        gt=0,
        description="Default number of documents to return to the generation layer.",
    )
    candidate_pool_k: int = Field(
        default=20,
        gt=0,
        description="Initial wide pool of candidates retrieved before fusion/reranking.",
    )

    # --------------------------------------------------------------------------
    # MMR (Maximal Marginal Relevance) Hyperparameters
    # --------------------------------------------------------------------------
    mmr_fetch_k: int = Field(
        default=20,
        gt=0,
        description="Number of candidate documents to fetch before applying MMR diversification.",
    )
    mmr_lambda: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Relevance vs diversity trade-off (1.0 = pure similarity, 0.0 = maximal diversity).",
    )

    # --------------------------------------------------------------------------
    # Hybrid Search & Reciprocal Rank Fusion (RRF) Hyperparameters
    # --------------------------------------------------------------------------
    rrf_k: int = Field(
        default=60,
        gt=0,
        description="RRF smoothing constant preventing high-ranking outliers from dominating.",
    )
    hybrid_dense_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight for dense vector retrieval in score-weighted fusion.",
    )
    hybrid_sparse_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight for sparse BM25 retrieval in score-weighted fusion.",
    )

    # --------------------------------------------------------------------------
    # Post-Retrieval Reranking & Compression Hyperparameters
    # --------------------------------------------------------------------------
    rerank_top_k: int = Field(
        default=5,
        gt=0,
        description="Number of top candidates to keep after cross-encoder reranking.",
    )
    compression_enabled: bool = Field(
        default=False,
        description="Whether to perform LLM/passage-based contextual compression on retrieved chunks.",
    )
    compression_similarity_threshold: float = Field(
        default=0.7,
        description="Threshold score for contextual sentence/passage extraction.",
    )

    # --------------------------------------------------------------------------
    # Query Transformation Hyperparameters
    # --------------------------------------------------------------------------
    multi_query_count: int = Field(
        default=3,
        gt=0,
        description="Number of alternative query formulations generated by MultiQueryRetriever.",
    )
    query_expansion_count: int = Field(
        default=3,
        gt=0,
        description="Number of expanded synonym/keyword queries generated during query expansion.",
    )

    # --------------------------------------------------------------------------
    # Storage & File Paths (Resolved relative to PROJECT_ROOT)
    # --------------------------------------------------------------------------
    documents_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "documents",
        description="Directory containing raw enterprise knowledge base documents.",
    )
    faiss_index_path: Path = Field(
        default=PROJECT_ROOT / "data" / "storage" / "faiss_index",
        description="Directory path where persistent FAISS vector store is saved.",
    )
    bm25_index_path: Path = Field(
        default=PROJECT_ROOT / "data" / "storage" / "bm25_index.pkl",
        description="File path where serialized BM25 sparse index is saved.",
    )
    eval_dataset_path: Path = Field(
        default=PROJECT_ROOT / "data" / "evaluation" / "eval_dataset.json",
        description="Path to evaluation ground-truth dataset.",
    )
    eval_results_path: Path = Field(
        default=PROJECT_ROOT / "data" / "evaluation" / "results.json",
        description="Path where evaluation benchmark results are written.",
    )

    # --------------------------------------------------------------------------
    # Observability & Logging
    # --------------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, overlap: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 500)
        if overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})")
        return overlap

    def ensure_directories_exist(self) -> None:
        """Ensures all configured data, storage, and evaluation directories exist."""
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Returns a singleton cached instance of the application settings."""
    settings = Settings()
    settings.ensure_directories_exist()
    return settings
