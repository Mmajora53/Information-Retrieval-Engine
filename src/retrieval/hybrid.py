"""Hybrid retrieval utilities combining BM25 and dense embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from src.retrieval.bm25 import fit_bm25, retrieve_bm25
from src.retrieval.embeddings import build_embeddings, retrieve_embeddings


def truncate_text_field(
    items: list[dict],
    text_field: str,
    max_chars: int | None,
) -> list[dict]:
    """
    Return copies of items with a truncated text field when requested.
    """
    if max_chars is None:
        return items
    if max_chars <= 0:
        raise ValueError("max_chars must be positive when provided.")

    truncated_items: list[dict] = []
    for item in items:
        updated = item.copy()
        updated[text_field] = str(updated.get(text_field) or "")[:max_chars]
        truncated_items.append(updated)
    return truncated_items


def fuse_rankings_rrf(
    emb_indices: np.ndarray,
    bm25_indices: np.ndarray,
    k_out: int,
    rrf_k: int = 60,
    weight_embeddings: float = 3.5,
    weight_bm25: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse embedding and BM25 rankings with weighted Reciprocal Rank Fusion.
    """
    if k_out <= 0:
        raise ValueError("k_out must be > 0")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")
    if weight_embeddings <= 0 or weight_bm25 <= 0:
        raise ValueError("RRF weights must be > 0")
    if emb_indices.ndim != 2 or bm25_indices.ndim != 2:
        raise ValueError("emb_indices and bm25_indices must be 2D arrays")
    if emb_indices.shape[0] != bm25_indices.shape[0]:
        raise ValueError("emb_indices and bm25_indices must have the same number of queries")

    n_queries = emb_indices.shape[0]
    fused_indices = np.zeros((n_queries, k_out), dtype=np.int64)
    fused_scores = np.zeros((n_queries, k_out), dtype=np.float32)

    for qi in range(n_queries):
        scores: dict[int, float] = {}

        for rank, doc_idx in enumerate(emb_indices[qi], start=1):
            did = int(doc_idx)
            scores[did] = scores.get(did, 0.0) + (weight_embeddings / (rrf_k + rank))

        for rank, doc_idx in enumerate(bm25_indices[qi], start=1):
            did = int(doc_idx)
            scores[did] = scores.get(did, 0.0) + (weight_bm25 / (rrf_k + rank))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k_out]
        fused_indices[qi, : len(ranked)] = [doc_id for doc_id, _ in ranked]
        fused_scores[qi, : len(ranked)] = [float(score) for _, score in ranked]

    return fused_indices, fused_scores


def retrieve_hybrid_bm25_embeddings(
    docs: list[dict],
    queries: list[dict],
    *,
    top_k: int,
    text_field: str = "content",
    embedding_model_name: str = "all-MiniLM-L12-v2",
    embedding_batch_size: int = 64,
    show_progress_bar: bool = True,
    cache_dir: Path = Path("data/cache"),
    embedding_device: str | None = None,
    embedding_precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
    embedding_max_seq_length: int | None = None,
    embedding_truncate_dim: int | None = None,
    embedding_chunk_size: int | None = None,
    embedding_normalize: bool = True,
    embedding_backend: Literal["torch", "onnx", "openvino"] = "torch",
    embedding_local_files_only: bool = False,
    hybrid_bm25_method: str = "plus",
    hybrid_candidate_multiplier: int = 1,
    hybrid_rrf_k: int = 60,
    hybrid_weight_embeddings: float = 3.5,
    hybrid_weight_bm25: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve documents with BM25 and embeddings, then fuse both rankings.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if top_k > len(docs):
        raise ValueError(f"top_k={top_k} cannot be greater than number of docs={len(docs)}")
    if hybrid_candidate_multiplier <= 0:
        raise ValueError("hybrid_candidate_multiplier must be > 0")

    candidate_k = min(len(docs), max(top_k, top_k * int(hybrid_candidate_multiplier)))

    bm25_model = fit_bm25(docs, text_field=text_field, method=hybrid_bm25_method)
    bm25_indices, _ = retrieve_bm25(
        bm25_model,
        docs,
        queries,
        k=candidate_k,
        text_field=text_field,
    )

    doc_emb, query_emb = build_embeddings(
        docs,
        queries,
        text_field=text_field,
        model_name=embedding_model_name,
        batch_size=embedding_batch_size,
        show_progress_bar=show_progress_bar,
        cache_dir=cache_dir,
        device=embedding_device,
        precision=embedding_precision,
        model_max_seq_length=embedding_max_seq_length,
        truncate_dim=embedding_truncate_dim,
        chunk_size=embedding_chunk_size,
        normalize_embeddings=embedding_normalize,
        backend=embedding_backend,
        local_files_only=embedding_local_files_only,
    )

    emb_indices, _ = retrieve_embeddings(
        doc_emb,
        query_emb,
        k=candidate_k,
        assume_normalized=embedding_normalize,
    )

    return fuse_rankings_rrf(
        emb_indices=emb_indices,
        bm25_indices=bm25_indices,
        k_out=top_k,
        rrf_k=hybrid_rrf_k,
        weight_embeddings=hybrid_weight_embeddings,
        weight_bm25=hybrid_weight_bm25,
    )
