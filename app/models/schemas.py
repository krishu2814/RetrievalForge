"""
Standardized Pydantic schemas for RetrievalForge.
Defines data models for retrieval results, metadata filtering, and pipeline traces.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """
    Standard result object returned by all retrievers in the system.
    Keeps track of document content, metadata, and diagnostic scores
    across each stage of the retrieval pipeline.
    """
    document_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
    rank: int = 1
    retriever: str = "dense"

    # Diagnostic fields for inspecting pipeline stages
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None
    original_rank: Optional[int] = None
    final_rank: Optional[int] = None
    query_variant: Optional[str] = None


class MetadataFilter(BaseModel):
    """
    Filter parameters for metadata filtering.
    Supports matching on common fields (department, version, document_type, etc.)
    as well as custom key-value pairs.
    """
    department: Optional[str] = None
    document_type: Optional[str] = None
    version: Optional[str] = None
    access_level: Optional[str] = None
    topic: Optional[str] = None
    custom: Dict[str, Any] = Field(default_factory=dict)

    def matches(self, metadata: Dict[str, Any]) -> bool:
        """
        Returns True if the given metadata dictionary satisfies all active filter conditions.
        """
        if self.department and metadata.get("department") != self.department:
            return False
        if self.document_type and metadata.get("document_type") != self.document_type:
            return False
        if self.version and str(metadata.get("version")) != str(self.version):
            return False
        if self.access_level and metadata.get("access_level") != self.access_level:
            return False
        if self.topic and metadata.get("topic") != self.topic:
            return False

        # Check any additional custom filters
        for key, expected_value in self.custom.items():
            if str(metadata.get(key)) != str(expected_value):
                return False

        return True


class RetrievalTrace(BaseModel):
    """
    Holds the complete end-to-end diagnostic trace for a query.
    Used by the Retrieval Debugger to show why a document reached the final context.
    """
    query: str
    strategy: str = "dense"
    query_variants: List[str] = Field(default_factory=list)
    filters: Optional[MetadataFilter] = None
    candidates: List[RetrievalResult] = Field(default_factory=list)
    final_context: str = ""


class RAGResponse(BaseModel):
    """
    Standard response object returned by the RAG generation layer.
    Contains the synthesized answer, original query, cited sources,
    and optional diagnostic retrieval trace.
    """
    answer: str
    query: str
    sources: List[str] = Field(default_factory=list)
    strategy: str = "hybrid_reranked"
    trace: Optional[RetrievalTrace] = None

