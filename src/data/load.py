"""Utilities to load the raw project JSON files."""

import json
from pathlib import Path


def load_json(path: Path) -> object:
    """
    Load a JSON file and return the parsed Python object.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    object
        Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_all(raw_dir: Path) -> tuple[list[dict], list[dict], list[dict], object]:
    """
    Load the raw datasets used throughout the project.

    Parameters
    ----------
    raw_dir : Path
        Directory containing `docs.json`, `queries_train.json`,
        `queries_test.json`, and `qgts_train.json`.

    Returns
    -------
    tuple[list[dict], list[dict], list[dict], object]
        Documents, training queries, test queries, and raw ground truth.
    """
    raw_dir = Path(raw_dir)

    documents = load_json(raw_dir / "docs.json")
    train_queries = load_json(raw_dir / "queries_train.json")
    test_queries = load_json(raw_dir / "queries_test.json")
    gts = load_json(raw_dir / "qgts_train.json")

    return documents, train_queries, test_queries, gts
