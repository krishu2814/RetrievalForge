"""
Unit and integration tests for Contextual Compressor.
"""

import pytest
from unittest.mock import MagicMock
import numpy as np

from app.retrieval.compression import ContextualCompressor, split_into_sentences
from app.models.schemas import RetrievalResult


def test_split_into_sentences():
    text = (
        "Enterprise accounts receive 24/7 dedicated support. "
        "Standard accounts receive response within 24 hours. "
        "Free tier is community support only!"
    )
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Enterprise accounts receive 24/7 dedicated support."
    assert sentences[1] == "Standard accounts receive response within 24 hours."
    assert sentences[2] == "Free tier is community support only!"


def test_split_into_sentences_single_line():
    text = "Short single sentence without punctuation"
    sentences = split_into_sentences(text)
    assert len(sentences) == 1
    assert sentences[0] == text


def test_compress_text_single_sentence():
    compressor = ContextualCompressor()
    text = "Only one sentence here."
    compressed, ratio = compressor.compress_text("query", text)
    assert compressed == text
    assert ratio == 1.0


def test_compress_text_with_mock_embeddings():
    # Setup mock embedder:
    # Query vector = [1, 0]
    # Sentence 1 vector = [1, 0] (cosine sim = 1.0 -> keep)
    # Sentence 2 vector = [0, 1] (cosine sim = 0.0 -> drop)
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [1.0, 0.0]
    mock_embedder.embed_documents.return_value = [
        [1.0, 0.0],
        [0.0, 1.0]
    ]

    compressor = ContextualCompressor(embedding_model=mock_embedder, similarity_threshold=0.5)
    text = "Enterprise plan costs $500 per month. Python 3.12 is the active runtime."
    compressed, ratio = compressor.compress_text("enterprise cost", text)

    assert "Enterprise plan costs $500 per month." in compressed
    assert "Python 3.12" not in compressed
    assert ratio < 1.0


def test_compress_documents():
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [1.0, 0.0]
    mock_embedder.embed_documents.return_value = [
        [1.0, 0.0],
        [0.1, 0.9]
    ]

    compressor = ContextualCompressor(embedding_model=mock_embedder, similarity_threshold=0.6)

    doc = RetrievalResult(
        document_id="doc_123",
        content="Refunds must be requested within 14 days. We also offer cloud backups.",
        metadata={"category": "billing"},
        score=0.88,
        rank=1,
        retriever="hybrid"
    )

    compressed_docs = compressor.compress_documents("refund window", [doc])
    assert len(compressed_docs) == 1
    res = compressed_docs[0]

    assert res.retriever == "hybrid_compressed"
    assert "Refunds must be requested within 14 days." in res.content
    assert res.metadata["original_content"] == doc.content
    assert "compression_ratio" in res.metadata
    assert res.metadata["compression_ratio"] <= 1.0


def test_compress_documents_empty():
    compressor = ContextualCompressor()
    assert compressor.compress_documents("", []) == []
    assert compressor.compress_documents("valid query", []) == []
