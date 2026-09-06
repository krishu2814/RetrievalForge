"""
Document loader module.
Loads raw text documents from disk and parses their metadata.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from app.ingestion.metadata import parse_metadata_header

logger = logging.getLogger(__name__)


def load_document_file(file_path: Path | str) -> Dict[str, Any]:
    """
    Reads a single text file, parses frontmatter metadata, and returns
    a dictionary with 'content' and 'metadata'.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    metadata, content = parse_metadata_header(text, default_source=path.name)

    return {
        "content": content,
        "metadata": metadata
    }


def load_all_documents(directory_path: Path | str) -> List[Dict[str, Any]]:
    """
    Loads all .txt documents in the given directory.
    Returns a list of document dicts.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        logger.warning(f"Documents directory does not exist: {dir_path}")
        return []

    txt_files = sorted(list(dir_path.glob("*.txt")))
    if not txt_files:
        logger.warning(f"No .txt documents found in {dir_path}")
        return []

    documents = []
    for file_path in txt_files:
        try:
            doc = load_document_file(file_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")

    logger.info(f"Loaded {len(documents)} documents from {dir_path}")
    return documents
