"""Evaluation helpers for retrieval and classification experiments."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mrr_at_k,
    # Re-export Phase 2 metrics so notebooks can import a single module.
    accuracy,
    macro_f1,
    balanced_accuracy,
    classification_report_phase2,
    confusion_matrix_phase2,
    query_category_accuracy,
)


def indices_to_docids(
    topk_indices: ArrayLike,
    docs: Sequence[dict],
    doc_id_field: str = "id",
) -> list[list[str]]:
    """
    Convert retrieved document indices into document IDs.
    """
    if not docs:
        raise ValueError("docs must not be empty.")

    topk_indices_arr = np.asarray(topk_indices)
    if topk_indices_arr.ndim != 2:
        raise ValueError(
            "topk_indices must be a 2D array-like structure with shape [n_queries, k]."
        )

    doc_ids = []
    for i, doc in enumerate(docs):
        if doc_id_field not in doc:
            raise KeyError(f"docs[{i}] does not contain '{doc_id_field}'.")
        doc_ids.append(str(doc[doc_id_field]))

    pred_docids: list[list[str]] = []
    n_docs = len(doc_ids)
    for q_idx, row in enumerate(topk_indices_arr):
        row_docids: list[str] = []
        for rank_idx, doc_idx in enumerate(row):
            idx = int(doc_idx)
            if idx < 0 or idx >= n_docs:
                raise IndexError(
                    f"topk_indices[{q_idx}][{rank_idx}]={idx} is out of bounds for {n_docs} documents."
                )
            row_docids.append(doc_ids[idx])
        pred_docids.append(row_docids)

    return pred_docids


def _extract_doc_ids(relevant_entries: Any) -> list[str]:
    """
    Normalize raw ground-truth entries into unique document IDs.
    """
    if relevant_entries is None:
        return []

    if not isinstance(relevant_entries, list):
        relevant_entries = [relevant_entries]

    doc_ids: list[str] = []
    seen: set[str] = set()

    for entry in relevant_entries:
        if isinstance(entry, dict):
            doc_id = entry.get("doc_id", entry.get("id", entry.get("docid")))
        else:
            doc_id = entry

        if doc_id is None:
            continue

        doc_id_str = str(doc_id)
        if doc_id_str not in seen:
            seen.add(doc_id_str)
            doc_ids.append(doc_id_str)

    return doc_ids


def adapt_ground_truth(gts_raw: Any) -> dict[str, list[str]]:
    """
    Normalize raw ground truth into `dict[query_id, list[doc_id]]`.
    """
    gt: dict[str, list[str]] = {}

    if isinstance(gts_raw, dict):
        for qid, value in gts_raw.items():
            if isinstance(value, dict):
                relevant_entries = value.get(
                    "relevant_doc_ids",
                    value.get("doc_ids", value.get("relevant", [])),
                )
            else:
                relevant_entries = value

            gt[str(qid)] = _extract_doc_ids(relevant_entries)
        return gt

    if isinstance(gts_raw, list):
        for row in gts_raw:
            if not isinstance(row, dict):
                continue

            qid = row.get("query_id", row.get("qid", row.get("id")))
            if qid is None:
                continue

            relevant_entries = row.get(
                "relevant_doc_ids",
                row.get("doc_ids", row.get("relevant", [])),
            )
            gt[str(qid)] = _extract_doc_ids(relevant_entries)
        return gt

    raise ValueError(f"Unsupported ground-truth format: {type(gts_raw)}")


def evaluate_run(pred_docids: list[list[str]], gt: dict[str, list[str]], query_ids: list[str], k: int) -> dict[str, float | int]:
    """
    Evaluate one retrieval run at a single cutoff `k`.
    """
    precision = precision_at_k(pred_docids, gt, query_ids, k)
    recall = recall_at_k(pred_docids, gt, query_ids, k)
    mrr = mrr_at_k(pred_docids, gt, query_ids, k)

    return {
        "k": k,
        "precision@k": precision,
        "recall@k": recall,
        "mrr@k": mrr,
    }

def evaluate_multi_k(pred_docids: list[list[str]], gt: dict[str, list[str]], query_ids: list[str], k_values: list[int]):
    """
    Evaluate the same predictions for several cutoff values.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for evaluate_multi_k. Install dependencies from requirements.txt."
        ) from exc

    rows = []
    for k in k_values:
        rows.append(evaluate_run(pred_docids, gt, query_ids, k))

    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

def adapt_ground_truth_categories(gts_raw: dict) -> dict[str, str]:
    """
    Extract one ground-truth category per query.
    """

    result = {}
    for qid, entry in gts_raw.items():
        result[qid] = entry["category"]
    return result


def evaluate_with_classification(
    pred_docids: list[list[str]],
    gt: dict[str, list[str]],
    query_ids: list[str],
    k: int,
    predicted_categories: dict[str, str],
    gt_categories: dict[str, str],
) -> dict:
    """
    Evaluate retrieval and category prediction jointly for Phase 2.
    """

    retrieval_metrics = evaluate_run(pred_docids, gt, query_ids, k)
    category_accuracy = query_category_accuracy(query_ids, predicted_categories, gt_categories)
    return {**retrieval_metrics, "query_category_accuracy": category_accuracy}


def evaluate_docs_vs_queries(
    clf,
    X_docs,
    y_docs: list[str],
    X_queries,
    y_queries: list[str],
) -> dict:
    """
    Compare classifier performance on documents and queries separately.
    """
    y_docs_pred = clf.predict(X_docs)
    y_queries_pred = clf.predict(X_queries)

    acc_docs = accuracy(y_true= y_docs, y_pred= y_docs_pred)
    acc_queries = accuracy(y_true= y_queries, y_pred= y_queries_pred)

    macro_f1_docs = macro_f1(y_true= y_docs, y_pred= y_docs_pred)
    macro_f1_queries = macro_f1(y_true= y_queries, y_pred= y_queries_pred)

    balanced_accuracy_docs = balanced_accuracy(y_true= y_docs, y_pred= y_docs_pred)
    balanced_accuracy_queries = balanced_accuracy(y_true= y_queries, y_pred= y_queries_pred)

    classification_report_phase2_docs = classification_report_phase2(y_true= y_docs, y_pred= y_docs_pred)
    classification_report_phase2_queries = classification_report_phase2(y_true= y_queries, y_pred= y_queries_pred)

    return {
        "docs": {
            "accuracy": acc_docs,
            "macro_f1": macro_f1_docs,
            "balanced_accuracy": balanced_accuracy_docs,
            "classification_report": classification_report_phase2_docs,
        },
        "queries": {
            "accuracy": acc_queries,
            "macro_f1": macro_f1_queries,
            "balanced_accuracy": balanced_accuracy_queries,
            "classification_report": classification_report_phase2_queries,
        }
    }
