"""
Unit tests for app.config.
Checks default settings values and basic config loading.
"""

import pytest
from app.config import Settings, settings, get_settings


def test_default_settings():
    assert settings.CHUNK_SIZE == 500
    assert settings.CHUNK_OVERLAP == 100
    assert settings.DEFAULT_TOP_K == 5
    assert settings.CANDIDATE_POOL_K == 20
    assert settings.MMR_LAMBDA == 0.7
    assert settings.RRF_K == 60
    assert settings.HYBRID_DENSE_WEIGHT == 0.5
    assert settings.HYBRID_SPARSE_WEIGHT == 0.5
    assert settings.LLM_MODEL == "gpt-4o-mini"
    assert settings.EMBEDDING_PROVIDER == "huggingface"


def test_custom_settings():
    custom = Settings(CHUNK_SIZE=300, CHUNK_OVERLAP=50)
    assert custom.CHUNK_SIZE == 300
    assert custom.CHUNK_OVERLAP == 50
    assert custom.chunk_size == 300
    assert custom.chunk_overlap == 50


def test_get_settings():
    s = get_settings()
    assert s is not None
    assert s.DEFAULT_TOP_K == 5
