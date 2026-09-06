import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables or .env file.
    Uses Pydantic BaseSettings for type validation and default values.
    """
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM Settings (OpenAI or any OpenAI-compatible API like Groq, Ollama)
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0

    # Embedding & Reranking Models
    EMBEDDING_PROVIDER: str = "huggingface"  # "huggingface" or "openai"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Document Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # Retrieval Defaults
    DEFAULT_TOP_K: int = 5
    CANDIDATE_POOL_K: int = 20

    # MMR (Maximal Marginal Relevance)
    MMR_FETCH_K: int = 20
    MMR_LAMBDA: float = 0.7

    # Hybrid Search & Reciprocal Rank Fusion (RRF)
    RRF_K: int = 60
    HYBRID_DENSE_WEIGHT: float = 0.5
    HYBRID_SPARSE_WEIGHT: float = 0.5

    # Reranking & Compression
    RERANK_TOP_K: int = 5
    COMPRESSION_ENABLED: bool = False

    # Query Transformation
    MULTI_QUERY_COUNT: int = 3
    QUERY_EXPANSION_COUNT: int = 3

    # Storage Paths
    DOCUMENTS_DIR: str = str(BASE_DIR / "data" / "documents")
    FAISS_INDEX_PATH: str = str(BASE_DIR / "data" / "storage" / "faiss_index")
    BM25_INDEX_PATH: str = str(BASE_DIR / "data" / "storage" / "bm25_index.pkl")
    EVAL_DATASET_PATH: str = str(BASE_DIR / "data" / "evaluation" / "eval_dataset.json")
    EVAL_RESULTS_PATH: str = str(BASE_DIR / "data" / "evaluation" / "results.json")

    # Logging
    LOG_LEVEL: str = "INFO"

    # Convenience lowercase properties for flexibility
    @property
    def chunk_size(self) -> int:
        return self.CHUNK_SIZE

    @property
    def chunk_overlap(self) -> int:
        return self.CHUNK_OVERLAP

    @property
    def default_top_k(self) -> int:
        return self.DEFAULT_TOP_K

    @property
    def candidate_pool_k(self) -> int:
        return self.CANDIDATE_POOL_K

    @property
    def mmr_fetch_k(self) -> int:
        return self.MMR_FETCH_K

    @property
    def mmr_lambda(self) -> float:
        return self.MMR_LAMBDA

    @property
    def rrf_k(self) -> int:
        return self.RRF_K

    @property
    def hybrid_dense_weight(self) -> float:
        return self.HYBRID_DENSE_WEIGHT

    @property
    def hybrid_sparse_weight(self) -> float:
        return self.HYBRID_SPARSE_WEIGHT

    @property
    def llm_model(self) -> str:
        return self.LLM_MODEL

    @property
    def embedding_provider(self) -> str:
        return self.EMBEDDING_PROVIDER

    @property
    def documents_dir(self) -> Path:
        return Path(self.DOCUMENTS_DIR)

    @property
    def faiss_index_path(self) -> Path:
        return Path(self.FAISS_INDEX_PATH)

    @property
    def bm25_index_path(self) -> Path:
        return Path(self.BM25_INDEX_PATH)

    @property
    def eval_dataset_path(self) -> Path:
        return Path(self.EVAL_DATASET_PATH)

    @property
    def eval_results_path(self) -> Path:
        return Path(self.EVAL_RESULTS_PATH)


# Create a single global instance for use across the application
settings = Settings()


def get_settings() -> Settings:
    """Helper function to get current settings."""
    return settings
