"""Pipeline package implementing the Strategy Pattern for plug-and-play retrieval modes."""

from app.pipeline.rag_pipeline import (
    RAGPipeline,
    get_pipeline,
    format_context,
    SUPPORTED_STRATEGIES,
)

__all__ = [
    "RAGPipeline",
    "get_pipeline",
    "format_context",
    "SUPPORTED_STRATEGIES",
]
