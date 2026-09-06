"""
Metadata extraction and management module.
Extracts document-level metadata from headers and attaches chunk-level metadata.
"""

from pathlib import Path
from typing import Tuple


def parse_metadata_header(text: str, default_source: str = "unknown.txt") -> Tuple[dict, str]:
    """
    Parses a document starting with '---' frontmatter header.
    Returns a tuple of (metadata_dict, cleaned_content).
    
    Example header format:
    ---
    document_id: refund-policy-v2
    source: refund_policy_v2.txt
    department: support
    document_type: policy
    version: 2.0
    date: 2026-01-15
    access_level: public
    topic: refunds
    ---
    """
    metadata = {
        "document_id": Path(default_source).stem,
        "source": default_source,
        "department": "general",
        "document_type": "document",
        "version": "1.0",
        "date": "2026-01-01",
        "access_level": "public",
        "topic": "general"
    }

    cleaned_content = text.strip()

    # Check if file has frontmatter separated by '---'
    if cleaned_content.startswith("---"):
        parts = cleaned_content.split("---", 2)
        if len(parts) >= 3:
            header_text = parts[1].strip()
            cleaned_content = parts[2].strip()

            # Parse line by line: key: value
            for line in header_text.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()

    return metadata, cleaned_content


def enrich_chunk_metadata(doc_metadata: dict, chunk_index: int, total_chunks: int) -> dict:
    """
    Creates metadata for an individual chunk by copying document metadata
    and attaching chunk-specific fields (chunk_index, chunk_id, total_chunks).
    """
    chunk_meta = dict(doc_metadata)
    doc_id = doc_metadata.get("document_id", "doc")
    chunk_meta["chunk_index"] = chunk_index
    chunk_meta["total_chunks"] = total_chunks
    chunk_meta["chunk_id"] = f"{doc_id}_chunk_{chunk_index}"
    return chunk_meta
