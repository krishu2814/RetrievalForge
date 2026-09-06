"""
Unit tests for document chunking.
"""

from app.ingestion.chunking import chunk_document, chunk_all_documents


def test_chunk_document():
    doc = {
        "content": "Paragraph one.\n\nParagraph two with more details.\n\nParagraph three.",
        "metadata": {
            "document_id": "test-doc",
            "department": "support"
        }
    }

    chunks = chunk_document(doc, chunk_size=30, chunk_overlap=10)

    assert len(chunks) > 1
    # Check metadata preservation and chunk keys
    for i, chunk in enumerate(chunks):
        assert "content" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["document_id"] == "test-doc"
        assert chunk["metadata"]["department"] == "support"
        assert chunk["metadata"]["chunk_index"] == i
        assert chunk["metadata"]["total_chunks"] == len(chunks)
        assert chunk["metadata"]["chunk_id"] == f"test-doc_chunk_{i}"


def test_chunk_all_documents():
    docs = [
        {
            "content": "Doc 1 content here.",
            "metadata": {"document_id": "doc-1"}
        },
        {
            "content": "Doc 2 content here.",
            "metadata": {"document_id": "doc-2"}
        }
    ]

    all_chunks = chunk_all_documents(docs, chunk_size=50, chunk_overlap=10)
    assert len(all_chunks) >= 2

    doc_ids = {c["metadata"]["document_id"] for c in all_chunks}
    assert "doc-1" in doc_ids
    assert "doc-2" in doc_ids
