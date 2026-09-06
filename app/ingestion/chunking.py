"""
Document chunking module.
Splits documents into overlapping chunks using RecursiveCharacterTextSplitter
and attaches enriched metadata to each chunk.
"""

from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.ingestion.metadata import enrich_chunk_metadata


def chunk_document(
    document: Dict[str, Any],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Chunks a single document dictionary into smaller overlapping chunks.
    Preserves all document metadata and adds chunk_id, chunk_index, and total_chunks.
    """
    size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP

    # LangChain text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    raw_text = document["content"]
    doc_metadata = document["metadata"]

    text_chunks = text_splitter.split_text(raw_text)
    total_chunks = len(text_chunks)

    chunks = []
    for index, chunk_text in enumerate(text_chunks):
        chunk_meta = enrich_chunk_metadata(doc_metadata, chunk_index=index, total_chunks=total_chunks)
        chunks.append({
            "content": chunk_text,
            "metadata": chunk_meta
        })

    return chunks


def chunk_all_documents(
    documents: List[Dict[str, Any]],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Chunks a list of documents and returns a flat list of all chunks.
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(doc_chunks)
    return all_chunks
