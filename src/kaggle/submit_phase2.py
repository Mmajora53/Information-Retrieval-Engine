"""Phase 2 Kaggle submission pipeline.

Pipeline:
    1. Load data
    2. Train TF-IDF + SVC classifier
    3. Predict query categories
    4. Query expansion (category-specific vocabulary)
    5. PRF — BM25 top-k docs → extract tags → append to query
    6. Hybrid BM25 + Embeddings retrieval (RRF)
    7. Hard-filter reranking (move predicted-category docs to front)
    8. Save submission CSV
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    PHASE2_OUTPUT_DIR,
    RANDOM_SEED,
    Phase2Config,
    resolve_runtime_paths,
)
from src.data.load import load_json
from src.data.preprocess import add_content_field
from src.classification.features import (
    build_count_features,
    build_embedding_features,
    build_tfidf_features,
    extract_texts_and_labels,
)
from src.classification.model import Classifier
from src.classification.rerank import build_docs_index, hard_filter, soft_boost
from src.evaluation.evaluate import indices_to_docids
from src.kaggle.format import save_submission
from src.retrieval.bm25 import fit_bm25, retrieve_bm25
from src.retrieval.hybrid import retrieve_hybrid_bm25_embeddings


# ---------------------------------------------------------------------------
# Category vocabulary — appended to each query before encoding
# ---------------------------------------------------------------------------

_CATEGORY_EXPANSIONS: dict[str, str] = {
    "android":     "android mobile app development java kotlin apk",
    "tex":         "tex latex document typesetting mathematics formula",
    "unix":        "unix linux shell bash terminal command line",
    "gaming":      "gaming video game console steam multiplayer",
    "programmers": "programming software development code algorithm design",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _numpy_encoder(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_phase2_pipeline(config: Phase2Config | None = None) -> pd.DataFrame:
    """Run the full Phase 2 submission pipeline and return the submission DataFrame."""
    if config is None:
        config = Phase2Config()

    set_global_seeds(config.random_seed)

    if config.top_k != 100:
        raise ValueError(f"Kaggle submission requires TOP_K=100, got {config.top_k}.")
    if config.rerank_strategy == "soft_boost" and config.classifier_method == "svc":
        print("[warn] soft_boost requires predict_proba — switch to classifier_method='logreg'.")

    paths = resolve_runtime_paths()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    # Always load from raw files to guarantee identical content field construction
    # (same as Kaggle notebook — processed files may have been built with different params)
    docs_raw = load_json(paths.raw_dir / "docs.json")
    queries_train_raw = load_json(paths.raw_dir / "queries_train.json")
    queries_test_raw = load_json(paths.raw_dir / "queries_test.json")

    docs, queries_test = add_content_field(docs_raw, queries_test_raw, clean=False)
    _, queries_train = add_content_field([], queries_train_raw, clean=False)
    print("Loaded from raw files and built content field.")

    print(f"Docs: {len(docs)} | Train queries: {len(queries_train)} | Test queries: {len(queries_test)}")

    # ── 2. Train classifier ───────────────────────────────────────────────────
    all_train_texts, all_train_labels = extract_texts_and_labels(
        docs, queries_train, text_field=config.text_field
    )
    test_texts, _ = extract_texts_and_labels(queries_test, text_field=config.text_field)

    if config.feature_method == "tfidf":
        _, X_train, X_test = build_tfidf_features(
            X_train=all_train_texts, X_val=None, X_test=test_texts
        )
    elif config.feature_method == "count":
        _, X_train, X_test = build_count_features(
            X_train=all_train_texts, X_val=None, X_test=test_texts
        )
    elif config.feature_method == "embeddings":
        _, X_train, X_test = build_embedding_features(
            X_train=all_train_texts, X_val=None, X_test=test_texts
        )
    else:
        raise ValueError(f"Unknown feature_method: {config.feature_method!r}")

    clf = Classifier(method=config.classifier_method)
    clf.fit(X_train, all_train_labels)

    # ── 3. Predict categories ─────────────────────────────────────────────────
    predicted_categories = clf.predict(X_test)
    print(f"Classifier: {config.classifier_method} | classes: {clf._classes}")

    # ── 4. Query expansion ────────────────────────────────────────────────────
    queries_expanded = []
    for q, cat in zip(queries_test, predicted_categories):
        q_exp = q.copy()
        q_exp["content"] = q["content"] + " " + _CATEGORY_EXPANSIONS.get(cat, cat)
        queries_expanded.append(q_exp)

    # ── 5. PRF: BM25 top-k → extract tags → append to query ──────────────────
    print("PRF: fitting BM25 for pseudo-relevance feedback...")
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    bm25_prf = fit_bm25(docs, text_field=config.text_field)
    prf_indices, _ = retrieve_bm25(
        bm25_prf, docs, queries_expanded, k=config.prf_top_k, text_field=config.text_field
    )

    queries_final = []
    for i, q_exp in enumerate(queries_expanded):
        tags: list[str] = []
        for doc_idx in prf_indices[i]:
            doc_tags = docs[doc_idx].get("tags", [])
            if isinstance(doc_tags, list):
                tags.extend(t for t in doc_tags if isinstance(t, str) and t.strip())
        unique_tags = list(dict.fromkeys(tags))[: config.prf_max_tags]
        q_final = q_exp.copy()
        if unique_tags:
            q_final["content"] = q_exp["content"] + " " + " ".join(unique_tags)
        queries_final.append(q_final)

    # ── 6. Hybrid BM25 + Embeddings retrieval (RRF) ───────────────────────────
    retrieval_k = config.top_k * config.retrieval_k_multiplier
    print(f"Retrieval: {retrieval_k} candidates per query (rrf_k={config.hybrid_rrf_k})...")

    topk_indices, topk_scores = retrieve_hybrid_bm25_embeddings(
        docs=docs,
        queries=queries_final,
        top_k=retrieval_k,
        text_field=config.text_field,
        cache_dir=paths.cache_dir,
        embedding_model_name=config.embedding_model_name,
        hybrid_weight_embeddings=config.hybrid_weight_embeddings,
        hybrid_weight_bm25=config.hybrid_weight_bm25,
        hybrid_rrf_k=config.hybrid_rrf_k,
    )
    pred_docids = indices_to_docids(topk_indices, docs)

    # ── 7. Reranking ──────────────────────────────────────────────────────────
    category_proba = (
        clf.predict_proba(X_test) if config.rerank_strategy == "soft_boost" else None
    )
    docs_by_id = build_docs_index(docs)
    reranked_docids: list[list[str]] = []

    for i in tqdm(range(len(queries_test)), desc="Reranking"):
        if config.rerank_strategy == "hard_filter":
            reranked_ids, _ = hard_filter(
                topk_doc_ids=pred_docids[i],
                topk_scores=topk_scores[i],
                docs_by_id=docs_by_id,
                predicted_category=predicted_categories[i],
                fallback_to_original=True,
            )
        elif config.rerank_strategy == "soft_boost":
            reranked_ids, _ = soft_boost(
                topk_doc_ids=pred_docids[i],
                topk_scores=topk_scores[i],
                docs_by_id=docs_by_id,
                category_proba=category_proba[i],
                classes=clf._classes,
                boost_factor=config.rerank_boost_factor,
            )
        else:
            reranked_ids = pred_docids[i]
        reranked_docids.append(reranked_ids)

    # ── 8. Save submission ────────────────────────────────────────────────────
    categories_dict = {
        str(q["id"]): cat for q, cat in zip(queries_test, predicted_categories)
    }
    paths.output_submission.parent.mkdir(parents=True, exist_ok=True)
    df = save_submission(
        query_ids=[str(q["id"]) for q in queries_test],
        pred_docids=reranked_docids,
        output_path=paths.output_submission,
        top_k=config.top_k,
        categories=categories_dict,
    )

    save_phase2_results(
        results={
            "config": config.as_dict(),
            "metrics": {},
            "submission_file": str(paths.output_submission),
        },
        output_dir=config.output_dir,
    )

    print(f"\nSubmission saved: {paths.output_submission}")
    print(f"Rows: {len(df)} | Top-K: {config.top_k}")
    return df


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_phase2_results(
    results: dict,
    output_dir: str | Path | None = None,
) -> Path:
    """Save Phase 2 run metadata to a timestamped JSON file."""
    out_path = Path(output_dir) if output_dir is not None else Path(PHASE2_OUTPUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = out_path / f"run_{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, default=_numpy_encoder)
    return file_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the Phase 2 submission pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2 Kaggle Submission")
    _default = Phase2Config()
    parser.add_argument("--classifier",              type=str,   default=_default.classifier_method,         choices=["nb", "svc", "logreg", "mlp"])
    parser.add_argument("--feature-method",          type=str,   default=_default.feature_method,            choices=["tfidf", "count", "embeddings"])
    parser.add_argument("--rerank",                  type=str,   default=_default.rerank_strategy,           choices=["hard_filter", "soft_boost", "none"])
    parser.add_argument("--boost-factor",            type=float, default=_default.rerank_boost_factor)
    parser.add_argument("--embedding-model",         type=str,   default=_default.embedding_model_name)
    parser.add_argument("--embedding-batch-size",    type=int,   default=_default.embedding_batch_size)
    parser.add_argument("--embedding-device",        type=str,   default=_default.embedding_device)
    parser.add_argument("--hybrid-rrf-k",            type=int,   default=_default.hybrid_rrf_k)
    parser.add_argument("--hybrid-weight-embeddings",type=float, default=_default.hybrid_weight_embeddings)
    parser.add_argument("--hybrid-weight-bm25",      type=float, default=_default.hybrid_weight_bm25)
    parser.add_argument("--retrieval-k-multiplier",  type=int,   default=_default.retrieval_k_multiplier)
    parser.add_argument("--prf-top-k",               type=int,   default=_default.prf_top_k)

    args = parser.parse_args()
    config = Phase2Config(
        classifier_method=args.classifier,
        feature_method=args.feature_method,
        rerank_strategy=args.rerank,
        rerank_boost_factor=args.boost_factor,
        embedding_model_name=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        hybrid_rrf_k=args.hybrid_rrf_k,
        hybrid_weight_embeddings=args.hybrid_weight_embeddings,
        hybrid_weight_bm25=args.hybrid_weight_bm25,
        retrieval_k_multiplier=args.retrieval_k_multiplier,
        prf_top_k=args.prf_top_k,
    )

    submission_df = run_phase2_pipeline(config)
    print(submission_df.head())


if __name__ == "__main__":
    main()
