"""
Unit tests for enterprise prompt engineering and security guardrails.
"""

import pytest

from app.generation.prompts import (
    build_rag_system_prompt,
    build_rag_user_prompt,
    build_rag_prompt,
    build_rag_messages,
    build_query_expansion_prompt,
    build_multi_query_prompt,
)


def test_rag_system_prompt_rules():
    prompt = build_rag_system_prompt()
    # Verify strict grounding
    assert "SOLELY on the provided retrieved documents" in prompt
    assert "Missing Information" in prompt
    assert "I don't have enough information" in prompt
    # Verify citation protocol
    assert "[Document <num>: <source>]" in prompt
    # Verify prompt injection defense
    assert "PASSIVE UNTRUSTED DATA" in prompt
    assert "ignore previous instructions" in prompt


def test_rag_user_prompt_structure():
    context = "[Document 1: refund.txt]\nRefunds must be requested within 14 days."
    query = "What is the refund window?"

    user_prompt = build_rag_user_prompt(query=query, context=context)
    assert "<context>" in user_prompt
    assert "</context>" in user_prompt
    assert context in user_prompt
    assert "User Question: What is the refund window?" in user_prompt


def test_rag_user_prompt_empty_context():
    user_prompt = build_rag_user_prompt(query="Any question", context="")
    assert "No context documents provided." in user_prompt


def test_rag_prompt_combined():
    prompt = build_rag_prompt(query="hello", context="some context")
    assert "RetrievalForge" in prompt
    assert "<context>" in prompt
    assert "User Question: hello" in prompt


def test_rag_messages_format():
    messages = build_rag_messages(query="Test Question", context="Test Context")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "RetrievalForge" in messages[0]["content"]
    assert "Test Question" in messages[1]["content"]


def test_query_expansion_and_multi_query_prompts():
    q_exp = build_query_expansion_prompt(query="database error", num_queries=4)
    assert "generate 4 diverse search query" in q_exp
    assert "Query: database error" in q_exp

    m_q = build_multi_query_prompt(query="API auth timeout", num_queries=3)
    assert "Generate 3 distinct versions" in m_q
    assert "Query: API auth timeout" in m_q
