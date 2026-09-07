"""
Unit tests to validate the golden evaluation dataset structure and content.
"""

import json
from pathlib import Path
import pytest

from app.config import settings

VALID_CATEGORIES = {
    "exact_keyword",
    "conceptual",
    "version_conflict",
    "multi_hop",
    "negative",
}

REQUIRED_KEYS = {
    "id",
    "query",
    "category",
    "expected_sources",
    "expected_doc_ids",
    "ground_truth_answer",
    "relevant_keywords",
}


def test_eval_dataset_file_exists():
    path = Path(settings.EVAL_DATASET_PATH)
    assert path.exists(), f"Eval dataset not found at {path}"


def test_eval_dataset_schema_and_contents():
    path = Path(settings.EVAL_DATASET_PATH)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert isinstance(data, list)
    assert len(data) >= 20, "Evaluation dataset should contain at least 20 benchmark queries"

    seen_ids = set()

    for item in data:
        # Check required keys
        for key in REQUIRED_KEYS:
            assert key in item, f"Missing key '{key}' in item {item.get('id')}"

        # Check unique IDs
        item_id = item["id"]
        assert item_id not in seen_ids, f"Duplicate query ID: {item_id}"
        seen_ids.add(item_id)

        # Check category
        assert item["category"] in VALID_CATEGORIES, f"Invalid category '{item['category']}' in {item_id}"

        # Category-specific assertions
        if item["category"] == "negative":
            assert len(item["expected_sources"]) == 0
            assert "I don't have enough information" in item["ground_truth_answer"]
        else:
            assert len(item["expected_sources"]) > 0
            assert len(item["expected_doc_ids"]) > 0
