"""
Unit tests for metadata parsing and chunk enrichment.
"""

from app.ingestion.metadata import parse_metadata_header, enrich_chunk_metadata


def test_parse_metadata_with_header():
    sample_text = """---
document_id: refund-policy-v2
department: support
version: 2.0
topic: refunds
---
# Refund Policy
This is the active refund policy content.
"""
    meta, content = parse_metadata_header(sample_text, default_source="refund.txt")

    assert meta["document_id"] == "refund-policy-v2"
    assert meta["department"] == "support"
    assert meta["version"] == "2.0"
    assert meta["topic"] == "refunds"
    assert content.startswith("# Refund Policy")


def test_parse_metadata_without_header():
    sample_text = "Plain document text with no frontmatter header."
    meta, content = parse_metadata_header(sample_text, default_source="plain.txt")

    assert meta["document_id"] == "plain"
    assert meta["source"] == "plain.txt"
    assert content == sample_text


def test_enrich_chunk_metadata():
    doc_meta = {
        "document_id": "doc-1",
        "department": "engineering",
        "version": "1.0"
    }

    chunk_meta = enrich_chunk_metadata(doc_meta, chunk_index=2, total_chunks=5)

    assert chunk_meta["document_id"] == "doc-1"
    assert chunk_meta["department"] == "engineering"
    assert chunk_meta["chunk_index"] == 2
    assert chunk_meta["total_chunks"] == 5
    assert chunk_meta["chunk_id"] == "doc-1_chunk_2"
