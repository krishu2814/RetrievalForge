"""
Indexing module for building and persisting dense (FAISS) and sparse (BM25) indexes.
"""

import os
import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from app.config import settings

logger = logging.getLogger(__name__)


def get_embedding_model():
    """
    Returns the configured embedding model.
    Defaults to sentence-transformers/all-MiniLM-L6-v2 via langchain_huggingface.
    """
    if settings.EMBEDDING_PROVIDER.lower() == "openai" and settings.OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model="text-embedding-3-small"
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def tokenize_text(text: str) -> List[str]:
    """
    Simple word tokenizer for BM25.
    Lowercases text and extracts alphanumeric words.
    """
    return re.findall(r"\w+", text.lower())


def build_dense_index(chunks: List[Dict[str, Any]], save_path: Optional[str] = None) -> FAISS:
    """
    Builds a FAISS dense vector store from document chunks.
    Optionally persists the index to disk.
    """
    if not chunks:
        raise ValueError("Cannot build dense index from empty chunks list.")

    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    logger.info(f"Generating embeddings and building FAISS index for {len(texts)} chunks...")
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        vector_store.save_local(save_path)
        logger.info(f"FAISS index successfully saved to {save_path}")

    return vector_store


def load_dense_index(index_path: Optional[str] = None) -> FAISS:
    """
    Loads a persisted FAISS vector store from disk.
    """
    path = index_path or settings.FAISS_INDEX_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"FAISS index not found at: {path}. Run 'python scripts/ingest.py' first.")

    embedding_model = get_embedding_model()
    return FAISS.load_local(
        str(path),
        embedding_model,
        allow_dangerous_deserialization=True
    )


def build_sparse_index(chunks: List[Dict[str, Any]], save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Builds a BM25 sparse index using rank-bm25.
    Optionally persists the index dictionary to a pickle file.
    """
    if not chunks:
        raise ValueError("Cannot build sparse index from empty chunks list.")

    logger.info(f"Tokenizing {len(chunks)} chunks and building BM25 index...")
    tokenized_corpus = [tokenize_text(chunk["content"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    index_data = {
        "bm25": bm25,
        "chunks": chunks
    }

    if save_path:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "wb") as f:
            pickle.dump(index_data, f)
        logger.info(f"BM25 index successfully saved to {save_path}")

    return index_data


def load_sparse_index(index_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads a persisted BM25 sparse index from disk.
    """
    path = index_path or settings.BM25_INDEX_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"BM25 index not found at: {path}. Run 'python scripts/ingest.py' first.")

    with open(path, "rb") as f:
        index_data = pickle.load(f)

    return index_data
