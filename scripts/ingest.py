"""
Document ingestion script.
Loads all documents from data/documents, chunks them, and builds persistent
FAISS (dense) and BM25 (sparse) indexes.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --chunk-size 400 --chunk-overlap 80
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.ingestion.loaders import load_all_documents
from app.ingestion.chunking import chunk_all_documents
from app.ingestion.indexing import build_dense_index, build_sparse_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents and build vector/sparse indexes.")
    parser.add_argument("--docs-dir", type=str, default=settings.DOCUMENTS_DIR, help="Path to raw documents directory")
    parser.add_argument("--chunk-size", type=int, default=settings.CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=settings.CHUNK_OVERLAP, help="Chunk overlap in characters")
    args = parser.parse_args()

    print("=" * 60)
    print("RetrievalForge: Document Ingestion & Indexing Pipeline")
    print("=" * 60)

    # 1. Load documents
    print(f"\n[1/4] Loading documents from: {args.docs_dir}")
    documents = load_all_documents(args.docs_dir)
    if not documents:
        print(f"Error: No documents found in {args.docs_dir}. Aborting.")
        sys.exit(1)
    print(f"      Loaded {len(documents)} document(s) successfully.")

    # 2. Chunk documents
    print(f"\n[2/4] Chunking documents (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunks = chunk_all_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"      Created {len(chunks)} text chunk(s) across {len(documents)} document(s).")

    # Sample chunk inspection
    if chunks:
        sample = chunks[0]
        print(f"      Sample chunk ID: {sample['metadata'].get('chunk_id')}")
        print(f"      Sample source: {sample['metadata'].get('source')}")

    # 3. Build dense vector index (FAISS)
    print(f"\n[3/4] Building FAISS dense index with {settings.EMBEDDING_MODEL}...")
    build_dense_index(chunks, save_path=settings.FAISS_INDEX_PATH)
    print(f"      FAISS index saved to: {settings.FAISS_INDEX_PATH}")

    # 4. Build sparse index (BM25)
    print(f"\n[4/4] Building BM25 sparse index...")
    build_sparse_index(chunks, save_path=settings.BM25_INDEX_PATH)
    print(f"      BM25 index saved to: {settings.BM25_INDEX_PATH}")

    print("\n" + "=" * 60)
    print("Ingestion completed successfully!")
    print(f"Total Documents: {len(documents)}")
    print(f"Total Chunks:    {len(chunks)}")
    print(f"Dense Index:     {settings.FAISS_INDEX_PATH}")
    print(f"Sparse Index:    {settings.BM25_INDEX_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
