"""
Unit tests for QueryExpander.
Verifies query reformulation, mock LLM parsing, and offline fallback behavior.
"""

from unittest.mock import MagicMock
from app.retrieval.query_expansion import QueryExpander


def test_query_expansion_basic_fallback():
    expander = QueryExpander()
    variants = expander.expand("What is the refund period?", num_queries=3)

    assert len(variants) >= 2
    # Original query must always be at index 0
    assert variants[0] == "What is the refund period?"
    for v in variants[1:]:
        assert len(v) > 0
        assert v != variants[0]


def test_query_expansion_empty():
    expander = QueryExpander()
    assert expander.expand("") == []
    assert expander.expand("   ") == []


def test_query_expansion_with_mock_llm():
    # Mock LLM returning 3 alternative queries
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = """1. customer refund eligibility timeline
2. money back guarantee deadline
3. how many days to request a refund"""
    mock_llm.invoke.return_value = mock_response

    expander = QueryExpander(llm=mock_llm)
    variants = expander.expand("refund policy", num_queries=3)

    assert len(variants) == 4  # 1 original + 3 generated
    assert variants[0] == "refund policy"
    assert variants[1] == "customer refund eligibility timeline"
    assert variants[2] == "money back guarantee deadline"
    assert variants[3] == "how many days to request a refund"


def test_query_expansion_leave_policy():
    expander = QueryExpander()
    variants = expander.expand("How many vacation days do employees get?", num_queries=2)

    assert variants[0] == "How many vacation days do employees get?"
    assert len(variants) >= 2
