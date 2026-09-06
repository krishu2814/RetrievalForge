"""
Unit tests for app.config.Settings.
Verifies default values, directory creation, and hyperparameter validation rules.
"""

import pytest
from pydantic import ValidationError
from app.config import Settings, get_settings


def test_default_settings_initialization():
    settings = Settings()
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 100
    assert settings.default_top_k == 5
    assert settings.candidate_pool_k == 20
    assert settings.mmr_lambda == 0.7
    assert settings.rrf_k == 60
    assert settings.hybrid_dense_weight == 0.5
    assert settings.hybrid_sparse_weight == 0.5
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.embedding_provider == "huggingface"


def test_chunk_overlap_validation():
    # Overlap strictly less than size should succeed
    valid_settings = Settings(chunk_size=400, chunk_overlap=200)
    assert valid_settings.chunk_overlap == 200

    # Overlap greater than or equal to size should raise ValidationError
    with pytest.raises(ValidationError):
        Settings(chunk_size=300, chunk_overlap=300)

    with pytest.raises(ValidationError):
        Settings(chunk_size=300, chunk_overlap=350)


def test_directories_creation(tmp_path):
    custom_docs = tmp_path / "docs"
    custom_faiss = tmp_path / "storage" / "faiss_index"
    custom_eval = tmp_path / "eval" / "eval.json"

    settings = Settings(
        documents_dir=custom_docs,
        faiss_index_path=custom_faiss,
        eval_dataset_path=custom_eval,
    )
    settings.ensure_directories_exist()

    assert custom_docs.exists()
    assert custom_faiss.parent.exists()
    assert custom_eval.parent.exists()


def test_get_settings_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
