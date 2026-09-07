"""
Answer synthesis module for RetrievalForge.
Interacts with OpenAI-compatible Chat APIs (OpenAI, Groq, Ollama, vLLM)
to generate strictly grounded, cited responses from retrieved context.
"""

import re
import logging
from typing import List, Optional, Dict, Any

from app.config import settings
from app.models.schemas import RetrievalResult, RetrievalTrace, RAGResponse
from app.generation.prompts import build_rag_messages

logger = logging.getLogger(__name__)


def extract_sources(
    answer: str,
    candidates: Optional[List[RetrievalResult]] = None
) -> List[str]:
    """
    Extracts cited document sources from the generated answer text.
    If no inline citations are matched, falls back to the unique sources
    present in the candidate documents.
    """
    # Look for [Document X: <source_name>] pattern
    cited_pattern = re.findall(r'\[Document\s+\d+:\s*([^\]]+)\]', answer, re.IGNORECASE)
    if cited_pattern:
        # Deduplicate while preserving order
        unique_cited = []
        for src in cited_pattern:
            cleaned = src.strip()
            if cleaned and cleaned not in unique_cited:
                unique_cited.append(cleaned)
        if unique_cited:
            return unique_cited

    # Fallback: extract from candidates metadata if available
    if candidates:
        unique_sources = []
        for doc in candidates:
            src = doc.metadata.get("source") or doc.document_id
            if src and src not in unique_sources:
                unique_sources.append(src)
        return unique_sources

    return []


class RAGGenerator:
    """
    Synthesizes grounded answers using an LLM.
    Supports any OpenAI-compatible API endpoint and includes a deterministic
    offline fallback when no API key is configured.
    """

    def __init__(
        self,
        llm=None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self._llm = llm

    def _get_llm(self):
        """
        Initializes the ChatOpenAI client on demand if an API key is available.
        """
        if self._llm is not None:
            return self._llm

        if self.api_key:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                return self._llm
            except Exception as e:
                logger.warning(f"Could not initialize ChatOpenAI: {e}")
                return None
        return None

    def _offline_fallback_generate(
        self,
        query: str,
        context: str,
        candidates: Optional[List[RetrievalResult]] = None
    ) -> str:
        """
        Deterministic offline fallback when no LLM API key is configured.
        Extracts key sentences from the top context passages.
        """
        if not context or context.strip() == "No relevant documents found.":
            return "I don't have enough information in the provided documents to answer this question."

        primary_source = "retrieved context"
        if candidates and len(candidates) > 0:
            doc = candidates[0]
            src = doc.metadata.get("source", doc.document_id)
            primary_source = f"Document 1: {src}"

        # Provide a structured offline extraction
        return (
            f"[Offline Mode: No OPENAI_API_KEY detected]\n"
            f"Based on [{primary_source}], the relevant facts retrieved for '{query}' are:\n\n"
            f"{context.strip()}"
        )

    def generate(
        self,
        query: str,
        context: str,
        strategy: str = "custom",
        candidates: Optional[List[RetrievalResult]] = None,
        trace: Optional[RetrievalTrace] = None,
    ) -> RAGResponse:
        """
        Generates a grounded answer from query and context string.
        """
        if not query or not query.strip():
            return RAGResponse(
                answer="Please provide a valid question.",
                query="",
                sources=[],
                strategy=strategy,
                trace=trace,
            )

        llm = self._get_llm()

        if llm is None:
            answer = self._offline_fallback_generate(query, context, candidates)
        else:
            try:
                messages = build_rag_messages(query=query, context=context)
                response = llm.invoke(messages)
                answer = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}. Falling back to offline synthesis.")
                answer = self._offline_fallback_generate(query, context, candidates)

        sources = extract_sources(answer, candidates=candidates)

        return RAGResponse(
            answer=answer,
            query=query,
            sources=sources,
            strategy=strategy,
            trace=trace,
        )

    def generate_from_trace(self, trace: RetrievalTrace) -> RAGResponse:
        """
        Convenience method to synthesize an answer directly from a RetrievalTrace.
        """
        return self.generate(
            query=trace.query,
            context=trace.final_context,
            strategy=trace.strategy,
            candidates=trace.candidates,
            trace=trace,
        )
