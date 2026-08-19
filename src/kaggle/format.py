"""Helpers to build and save Kaggle submission files."""

from __future__ import annotations

import json
from pathlib import Path


SUBMISSION_COLUMNS = ["query_id", "relevant_doc_ids", "category"]
DEFAULT_CATEGORY = "?"
DEFAULT_TOP_K = 100


def _normalize_docids(docids: list[str], top_k: int) -> list[str]:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    seen: set[str] = set()
    normalized: list[str] = []
    for doc_id in docids:
        doc_id_str = str(doc_id)
        if doc_id_str not in seen:
            seen.add(doc_id_str)
            normalized.append(doc_id_str)

    if len(normalized) < top_k:
        raise ValueError(
            f"Each query must have at least {top_k} predicted doc IDs, got {len(normalized)}."
        )

    return normalized[:top_k]


def make_submission(
    query_ids: list[str],
    pred_docids: list[list[str]],
    top_k: int = DEFAULT_TOP_K,
    category: str = DEFAULT_CATEGORY,
    categories: dict[str, str] | None = None,
):
    """
    Build a Kaggle submission DataFrame.

    Parameters
    ----------
    query_ids : list[str]
        Ordered query identifiers.
    pred_docids : list[list[str]]
        Predicted document IDs aligned with `query_ids`.
    top_k : int, default=100
        Number of document IDs kept per query.
    category : str, default="?"
        Fallback category value used when `categories` is not provided.
    categories : dict[str, str] | None, default=None
        Optional per-query predicted categories for Phase 2.
    """
    if len(query_ids) != len(pred_docids):
        raise ValueError("query_ids and pred_docids must have the same length.")

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required to build submissions. Install dependencies first."
        ) from exc

    rows: list[dict[str, str]] = []
    for i, (qid, docs_for_query) in enumerate(zip(query_ids, pred_docids)):
        if docs_for_query is None:
            raise ValueError(f"pred_docids[{i}] is None.")

        normalized_docids = _normalize_docids(list(docs_for_query), top_k=top_k)

        row_category = categories.get(str(qid), category) if categories else category

        rows.append(
            {
                "query_id": str(qid),
                "relevant_doc_ids": json.dumps(normalized_docids, ensure_ascii=False),
                "category": str(row_category),
            }
        )

    submission_df = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    return submission_df


def save_submission(
    query_ids: list[str],
    pred_docids: list[list[str]],
    output_path: Path = Path("outputs/submissions/submission.csv"),
    top_k: int = DEFAULT_TOP_K,
    category: str = DEFAULT_CATEGORY,
    categories: dict[str, str] | None = None,
):
    """
    Create and save a Kaggle submission CSV.
    """
    submission_df = make_submission(
        query_ids=query_ids,
        pred_docids=pred_docids,
        top_k=top_k,
        category=category,
        categories=categories,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return submission_df
