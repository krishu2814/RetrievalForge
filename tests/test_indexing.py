"""
Unit tests for indexing module (FAISS and BM25 index building and loading).
"""

from pathlib import Path
from app.ingestion.indexing import (
    tokenize_text,
    build_sparse_index,
    load_sparse_index,
    build_dense_index,
    load_dense_index
)


def test_tokenize_text():
    text = "Refund policy for NovaAI! Error code: ERR_AUTH_504."
    tokens = tokenize_text(text)

    assert "refund" in tokens
    assert "policy" in tokens
    assert "novaai" in tokens
    assert "err_auth_504" in tokens


def test_sparse_index_build_and_load(tmp_path):
    chunks = [
        {"content": "Customer refund policy details.", "metadata": {"doc_id": "refund"}},
        {"content": "Technical architecture and microservices.", "metadata": {"doc_id": "tech"}}
    ]

    save_path = tmp_path / "bm25_test.pkl"
    index_data = build_sparse_index(chunks, save_path=str(save_path))

    assert "bm25" in index_data
    assert len(index_data["chunks"]) == 2
    assert save_path.exists()

    # Test loading
    loaded_data = load_sparse_index(str(save_path))
    assert "bm25" in loaded_data
    assert len(loaded_data["chunks"]) == 2


def test_dense_index_build_and_load(tmp_path):
    chunks = [
        {"content": "NovaAI platform technical overview.", "metadata": {"doc_id": "nova"}},
        {"content": "Billing grace period and payment terms.", "metadata": {"doc_id": "billing"}}
    ]

    save_path = tmp_path / "faiss_test"
    vector_store = build_dense_index(chunks, save_path=str(save_path))

    assert vector_store is not None
    assert (save_path / "index.faiss").exists()
    assert (save_path / "index.pkl").exists()

    # Test loading
    loaded_store = load_dense_index(str(save_path))
    assert loaded_store is not None
