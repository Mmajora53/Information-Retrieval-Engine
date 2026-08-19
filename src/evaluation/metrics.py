"""Metrics used to evaluate retrieval and classification runs."""

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix)


def _evaluate_at_k(pred_docids: list[list[str]], gt: dict[str, list[str]], query_ids: list[str], k: int, metric_type: str) -> float:
    """
    Compute a ranking metric at cutoff `k` across all queries.
    """
    if metric_type not in {"precision", "recall", "mrr"}:
        raise ValueError("metric_type must be 'precision', 'recall', or 'mrr'.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if len(pred_docids) != len(query_ids):
        raise ValueError("pred_docids and query_ids must have the same length.")
    missing_qids = [str(qid) for qid in query_ids if str(qid) not in gt]
    if missing_qids:
        raise ValueError(
            "Missing query IDs in gt: " + ", ".join(sorted(set(missing_qids)))
        )

    scores = []

    for i in range(len(pred_docids)):
        qid = str(query_ids[i])
        pred_topk = [str(d) for d in pred_docids[i][:k]]
        relevant_set = set(str(d) for d in gt.get(qid, []))

        if len(relevant_set) == 0:
            scores.append(0.0)
            continue

        if metric_type == "precision":
            # Remove duplicates for set-based metrics to avoid counting a doc twice.
            seen = set()
            pred_topk_unique = []
            for d in pred_topk:
                if d not in seen:
                    pred_topk_unique.append(d)
                    seen.add(d)
            hits = sum(1 for d in pred_topk_unique if d in relevant_set)
            scores.append(hits / k)
        elif metric_type == "recall":
            # Remove duplicates for set-based metrics to avoid counting a doc twice.
            seen = set()
            pred_topk_unique = []
            for d in pred_topk:
                if d not in seen:
                    pred_topk_unique.append(d)
                    seen.add(d)
            hits = sum(1 for d in pred_topk_unique if d in relevant_set)
            scores.append(hits / len(relevant_set))
        else:  # mrr
            reciprocal_rank = 0.0
            for rank, doc_id in enumerate(pred_topk, start=1):
                if doc_id in relevant_set:
                    reciprocal_rank = 1.0 / rank
                    break
            scores.append(reciprocal_rank)

    return sum(scores) / len(scores) if scores else 0.0

def precision_at_k(pred_docids: list[list[str]], gt: dict[str, list[str]], query_ids: list[str], k: int) -> float:
    """Compute mean Precision@k across all queries."""
    return _evaluate_at_k(pred_docids, gt, query_ids, k, "precision")

def recall_at_k(pred_docids: list[list[str]], gt: dict[str, list[str]], query_ids: list[str], k: int) -> float:
    """Compute mean Recall@k across all queries."""
    return _evaluate_at_k(pred_docids, gt, query_ids, k, "recall")

def mrr_at_k(pred_docids: list[list[str]], gt: dict[str, list[str]], query_ids: list[str], k: int) -> float:
    """Compute mean MRR@k across all queries."""
    return _evaluate_at_k(pred_docids, gt, query_ids, k, "mrr")


# ---------------------------------------------------------------------------
# Phase 2 metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """
    Compute overall classification accuracy.
    """

    acc = accuracy_score(y_true, y_pred)

    return acc


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """
    Compute macro-averaged F1 across all classes.
    """

    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return macro


def balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """
    Compute balanced accuracy across classes.
    """

    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return balanced_acc



def classification_report_phase2(y_true: list[str], y_pred: list[str]) -> str:
    """
    Return a formatted classification report.
    """

    class_rep = classification_report(y_true, y_pred, zero_division=0)

    return class_rep


def confusion_matrix_phase2(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> "np.ndarray":
    """Return the confusion matrix as a numpy array."""

    if labels is None:
        labels = sorted(list(set(y_true)))  # Use a set to keep unique values.
    
    return confusion_matrix(y_true, y_pred, labels=labels)


def query_category_accuracy(
    query_ids: list[str],
    predicted_categories: dict[str, str],
    gt_categories: dict[str, str],
) -> float:
    """Compute the fraction of correctly predicted query categories."""

    matches = 0
    total = 0
    missing_qids = []

    for qid in query_ids:

        if qid not in gt_categories:
            missing_qids.append(qid)
            continue

        if qid not in predicted_categories:
            raise ValueError(f"The predicted category for query_id {qid} is missing.")

        if predicted_categories[qid] == gt_categories[qid]:
            matches += 1

        total += 1

    if missing_qids:
        print(f"Warning: ignored {len(missing_qids)} query_ids that are not in gt_categories.")

    if total > 0:
        return matches / total
    else:
        return 0.0
