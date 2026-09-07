"""
Unit tests for the RAGGenerator answer synthesis module.
"""

import pytest
from unittest.mock import MagicMock

from app.generation.generator import RAGGenerator, extract_sources
from app.models.schemas import RetrievalResult, RetrievalTrace, RAGResponse


def test_extract_sources_from_citations():
    text = (
        "According to [Document 1: refund_policy_v2.txt], refunds are permitted within 30 days. "
        "Also, [Document 2: terms_of_service.txt] covers accounts."
    )
    sources = extract_sources(text)
    assert len(sources) == 2
    assert "refund_policy_v2.txt" in sources
    assert "terms_of_service.txt" in sources


def test_extract_sources_fallback_to_candidates():
    text = "Refunds are processed within 30 days."
    candidates = [
        RetrievalResult(
            document_id="doc1",
            content="Content",
            metadata={"source": "refund_policy_v2.txt"},
            score=0.9,
            rank=1,
        ),
        RetrievalResult(
            document_id="doc2",
            content="Content",
            metadata={"source": "faq.txt"},
            score=0.8,
            rank=2,
        ),
    ]
    sources = extract_sources(text, candidates=candidates)
    assert len(sources) == 2
    assert "refund_policy_v2.txt" in sources
    assert "faq.txt" in sources


def test_extract_sources_empty():
    assert extract_sources("") == []
    assert extract_sources("Some answer without citations", candidates=[]) == []


def test_generator_with_mock_llm():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "According to [Document 1: refund.txt], refunds are 14 days."
    mock_llm.invoke.return_value = mock_response

    generator = RAGGenerator(llm=mock_llm)
    response = generator.generate(
        query="What is the refund window?",
        context="[Document 1: refund.txt]\nRefund window is 14 days.",
        strategy="hybrid_reranked"
    )

    assert isinstance(response, RAGResponse)
    assert response.query == "What is the refund window?"
    assert response.strategy == "hybrid_reranked"
    assert "refund.txt" in response.sources
    assert "According to [Document 1: refund.txt]" in response.answer
    assert mock_llm.invoke.called


def test_generator_offline_fallback():
    # Pass api_key="" to force offline mode
    generator = RAGGenerator(llm=None, api_key="")
    context = "[Document 1: pricing.txt]\nStarter tier costs $29/mo."
    response = generator.generate(
        query="How much is Starter tier?",
        context=context,
        strategy="dense"
    )

    assert isinstance(response, RAGResponse)
    assert "[Offline Mode" in response.answer
    assert "Starter tier costs $29/mo." in response.answer


def test_generator_from_trace():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "The answer is grounded in [Document 1: security.txt]."
    mock_llm.invoke.return_value = mock_response

    generator = RAGGenerator(llm=mock_llm)

    candidate = RetrievalResult(
        document_id="sec_1",
        content="2FA is required for all admins.",
        metadata={"source": "security.txt", "department": "Security"},
        score=0.95,
        rank=1,
    )
    trace = RetrievalTrace(
        query="Is 2FA required?",
        strategy="hybrid_reranked",
        candidates=[candidate],
        final_context="[Document 1] Source: security.txt\n2FA is required for all admins.",
    )

    response = generator.generate_from_trace(trace)
    assert response.query == "Is 2FA required?"
    assert response.strategy == "hybrid_reranked"
    assert response.trace == trace
    assert "security.txt" in response.sources


def test_generator_empty_query():
    generator = RAGGenerator()
    response = generator.generate(query="", context="some context")
    assert response.answer == "Please provide a valid question."
