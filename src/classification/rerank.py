"""Reranking utilities driven by predicted query categories."""

from __future__ import annotations

import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

def hard_filter(
    topk_doc_ids: list[str],
    topk_scores: np.ndarray,
    docs_by_id: dict[str, dict],
    predicted_category: str,
    fallback_to_original: bool = True,
) -> tuple[list[str], np.ndarray]:
    """
    Move category-matching documents to the front of the ranking.
    """

    matched_ids = []
    matched_scores = []
    excluded_ids = []
    excluded_scores = []
    
    for i in range(len(topk_doc_ids)):
        doc_id = topk_doc_ids[i]
        score = topk_scores[i]
        doc = docs_by_id[doc_id]

        if doc["category"] == predicted_category:
            matched_ids.append(doc_id)
            matched_scores.append(score)
        else:
            excluded_ids.append(doc_id)
            excluded_scores.append(score)

    if fallback_to_original and len(matched_ids) < len(topk_doc_ids):
        matched_ids += excluded_ids
        matched_scores += excluded_scores

    return matched_ids, np.array(matched_scores)

def soft_boost(
    topk_doc_ids: list[str],
    topk_scores: np.ndarray,
    docs_by_id: dict[str, dict],
    category_proba: np.ndarray | None,
    classes: list[str],
    boost_factor: float = 1.5,
) -> tuple[list[str], np.ndarray]:
    """
    Re-score documents with the predicted category probabilities.
    """

    if category_proba is None:
        return topk_doc_ids, topk_scores
    
    prob_by_cat = {}
    for i in range(len(classes)):
        prob_by_cat[classes[i]] = category_proba[i]
    
    boosted_scores = []
    for i in range(len(topk_doc_ids)):
        doc_id = topk_doc_ids[i]
        score = topk_scores[i]
        doc_category = docs_by_id[doc_id]["category"]
        proba = prob_by_cat.get(doc_category, 0.0)
        new_score = score * (1 + boost_factor * proba)
        boosted_scores.append(new_score)

    boosted_scores = np.array(boosted_scores)
    sorted_indices = np.argsort(boosted_scores)[::-1]

    reranked_doc_ids = [topk_doc_ids[i] for i in sorted_indices]
    reranked_scores = boosted_scores[sorted_indices]

    return reranked_doc_ids, reranked_scores


def build_docs_index(docs: list[dict]) -> dict[str, dict]:
    """
    Build a dictionary keyed by document ID for fast lookup.
    """
    return {str(doc["id"]): doc for doc in docs}


def cross_encoder_rerank(
    query_texts: list[str],
    topk_docids: list[list[str]],
    docs_by_id: dict[str, dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    model=None,
    k_in: int | None = None,
    k_out: int | None = None,
    batch_size: int = 32,
    max_doc_chars: int = 512,
    show_progress_bar: bool = True,
) -> tuple[list[list[str]], list[np.ndarray]]:
    """
    Rerank retrieved candidates with a cross-encoder model.
    """
    
    if CrossEncoder is None:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install it with `pip install sentence-transformers`."
        )
    
    if model is None:
        model = CrossEncoder(model_name)

    if not query_texts:
        return [], []

    K = len(topk_docids[0])
    if k_in is None:
        k_in = K
    if k_out is None:
        k_out = K

    reranked_docids_all: list[list[str]] = []
    reranked_scores_all: list[np.ndarray] = []

    from tqdm.auto import tqdm
    query_iter = tqdm(
        zip(query_texts, topk_docids),
        total=len(query_texts),
        desc="Cross-encoder reranking",
        disable=not show_progress_bar,
    )

    for q_text, cand_ids in query_iter:
        # 1) Limit to top k_in candidates from original retrieval
        cand_ids_in = cand_ids[:k_in]

        # 2) Build (query, doc) pairs — truncate doc content to keep memory low
        pairs = []
        for doc_id in cand_ids_in:
            doc = docs_by_id[doc_id]
            doc_text = doc["content"][:max_doc_chars]
            pairs.append((q_text, doc_text))

        # 3) Score with cross-encoder (batched)
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scores = np.asarray(scores)

        # 4) Sort by score desc and select top k_out
        sorted_idx = np.argsort(scores)[::-1]
        selected_idx = sorted_idx[:k_out]
        reranked_ids = [cand_ids_in[i] for i in selected_idx]
        reranked_scores = scores[selected_idx]

        # 5) (Optional) pad back to original K with remaining original candidates
        if k_out < K:
            remaining = [d for d in cand_ids if d not in reranked_ids]
            reranked_ids = reranked_ids + remaining[: K - len(reranked_ids)]
            # Use 0.0 for candidates that were not scored by the cross-encoder.
            pad_scores = np.zeros(len(reranked_ids) - len(reranked_scores))
            reranked_scores = np.concatenate([reranked_scores, pad_scores])

        reranked_docids_all.append(reranked_ids)
        reranked_scores_all.append(reranked_scores)

    return reranked_docids_all, reranked_scores_all
