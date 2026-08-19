"""Dense embedding helpers with optional caching."""

import hashlib
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_MODEL_CACHE: dict[str, Any] = {}


def _normalize_texts(texts: list[str] | None) -> list[str]:
    """Normalize a text list and replace missing values with empty strings."""
    if texts is None:
        raise ValueError("texts must not be None.")
    return ["" if t is None else str(t) for t in texts]


def _texts_fingerprint(texts: list[str]) -> str:
    """Build a short fingerprint for a list of texts."""
    hasher = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, byteorder="little"))
        hasher.update(encoded)
    return hasher.hexdigest()[:12]

def _looks_like_offline_or_cache_error(exc: Exception) -> bool:
    """
    Detect likely offline or missing-cache model loading errors.
    """
    message = str(exc).lower()
    exc_name = exc.__class__.__name__.lower()

    message_markers = (
        "nodename nor servname provided",
        "temporary failure in name resolution",
        "name resolution",
        "cannot send a request",
        "connection error",
        "max retries exceeded",
        "failed to establish a new connection",
        "httpsconnectionpool",
        "connecttimeout",
        "readtimeout",
        "offline",
        "could not connect",
        "not found in local cache",
        "is not a local folder",
        "repository not found",
    )
    name_markers = (
        "localentrynotfound",
        "repositorynotfound",
        "connectionerror",
        "connecttimeout",
        "readtimeout",
    )

    return any(marker in message for marker in message_markers) or any(
        marker in exc_name for marker in name_markers
    )


def _model_load_error_message(model_name: str, original_error: Exception) -> str:
    return (
        f"Failed to load SentenceTransformer model '{model_name}'. "
        "Likely cause: no network access and/or model not cached locally.\n"
        "How to fix:\n"
        "1) Preload the model once in an online environment:\n"
        "   from sentence_transformers import SentenceTransformer; "
        f"SentenceTransformer('{model_name}')\n"
        "2) Reuse the same local Hugging Face cache on this machine.\n"
        "3) Or reuse precomputed embeddings from data/cache for this dataset/model.\n"
        f"Original error: {original_error}"
    )


def _model_cache_key(
    model_name: str,
    device: str | None,
    backend: Literal["torch", "onnx", "openvino"],
    truncate_dim: int | None,
    model_max_seq_length: int | None,
    local_files_only: bool,
) -> str:
    return (
        f"{model_name}|device={device or 'auto'}|backend={backend}|"
        f"truncate_dim={truncate_dim}|max_seq={model_max_seq_length}|local={int(local_files_only)}"
    )


def _get_model(
    model_name: str,
    device: str | None = None,
    backend: Literal["torch", "onnx", "openvino"] = "torch",
    truncate_dim: int | None = None,
    model_max_seq_length: int | None = None,
    local_files_only: bool = False,
) -> "SentenceTransformer":
    """Load and cache a SentenceTransformer model instance."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sentence-transformers is required for encode_texts/build_embeddings. "
            "Install dependencies from requirements.txt."
        ) from exc

    cache_key = _model_cache_key(
        model_name=model_name,
        device=device,
        backend=backend,
        truncate_dim=truncate_dim,
        model_max_seq_length=model_max_seq_length,
        local_files_only=local_files_only,
    )
    if cache_key not in _MODEL_CACHE:
        try:
            model = SentenceTransformer(
                model_name,
                device=device,
                truncate_dim=truncate_dim,
                backend=backend,
                local_files_only=local_files_only,
            )
            if model_max_seq_length is not None:
                model.max_seq_length = int(model_max_seq_length)
            _MODEL_CACHE[cache_key] = model
        except Exception as exc:
            if _looks_like_offline_or_cache_error(exc):
                raise RuntimeError(_model_load_error_message(model_name, exc)) from exc
            raise
    return _MODEL_CACHE[cache_key]

def encode_texts(
    model_name: str,
    texts: list[str],
    batch_size: int = 64,
    show_progress_bar: bool = True,
    device: str | None = None,
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
    model_max_seq_length: int | None = None,
    truncate_dim: int | None = None,
    chunk_size: int | None = None,
    normalize_embeddings: bool = True,
    backend: Literal["torch", "onnx", "openvino"] = "torch",
    local_files_only: bool = False,
) -> np.ndarray:
    """Encode texts into dense embeddings."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    clean_texts = _normalize_texts(texts)
    model = _get_model(
        model_name=model_name,
        device=device,
        backend=backend,
        truncate_dim=truncate_dim,
        model_max_seq_length=model_max_seq_length,
        local_files_only=local_files_only,
    )

    emb = model.encode(
        clean_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        precision=precision,
        convert_to_numpy=True,
        device=device,
        normalize_embeddings=normalize_embeddings,
        truncate_dim=truncate_dim,
        chunk_size=chunk_size,
    )

    emb = np.asarray(emb)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)

    return emb

def cache_save(path: Path, array: np.ndarray) -> None:
    """Save a numpy array to disk as `.npy`."""
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)

def cache_load(path: Path) -> np.ndarray:
    """Load a numpy array from disk."""
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")

    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    return np.load(path, allow_pickle=False)

def build_embeddings(
    docs: list[dict],
    queries: list[dict],
    text_field: str = "content",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress_bar: bool = True,
    cache_dir: Path = Path("data/cache"),
    device: str | None = None,
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
    truncate_dim: int | None = None,
    chunk_size: int | None = None,
    normalize_embeddings: bool = True,
    backend: Literal["torch", "onnx", "openvino"] = "torch",
    model_max_seq_length: int | None = None,
    local_files_only: bool = False,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build document and query embeddings, optionally using the cache."""

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    doc_texts = [str(d.get(text_field) or "") for d in docs]
    query_texts = [str(q.get(text_field) or "") for q in queries]

    safe_model_name = model_name.replace("/", "_")
    safe_text_field = text_field.replace("/", "_")
    safe_precision = precision.replace("/", "_")
    norm_tag = "norm1" if normalize_embeddings else "norm0"
    trunc_tag = f"trunc{truncate_dim}" if truncate_dim is not None else "truncNone"
    maxseq_tag = f"maxseq{model_max_seq_length}" if model_max_seq_length is not None else "maxseqNone"
    device_tag = (device or "auto").replace("/", "_")
    backend_tag = backend.replace("/", "_")
    local_tag = f"local{int(local_files_only)}"
    doc_fingerprint = _texts_fingerprint(doc_texts)
    query_fingerprint = _texts_fingerprint(query_texts)
    doc_cache_path = cache_dir / (
        f"docs_{safe_model_name}_{safe_text_field}_{safe_precision}_{norm_tag}_"
        f"{trunc_tag}_{maxseq_tag}_{backend_tag}_{device_tag}_{local_tag}_{doc_fingerprint}.npy"
    )
    query_cache_path = cache_dir / (
        f"queries_{safe_model_name}_{safe_text_field}_{safe_precision}_{norm_tag}_"
        f"{trunc_tag}_{maxseq_tag}_{backend_tag}_{device_tag}_{local_tag}_{query_fingerprint}.npy"
    )

    if use_cache and doc_cache_path.exists():
        doc_embeddings = cache_load(doc_cache_path)
    else:
        doc_embeddings = encode_texts(
            model_name,
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            device=device,
            precision=precision,
            model_max_seq_length=model_max_seq_length,
            truncate_dim=truncate_dim,
            chunk_size=chunk_size,
            normalize_embeddings=normalize_embeddings,
            backend=backend,
            local_files_only=local_files_only,
        )
        if use_cache:
            cache_save(doc_cache_path, doc_embeddings)

    if use_cache and query_cache_path.exists():
        query_embeddings = cache_load(query_cache_path)
    else:
        query_embeddings = encode_texts(
            model_name,
            query_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            device=device,
            precision=precision,
            model_max_seq_length=model_max_seq_length,
            truncate_dim=truncate_dim,
            chunk_size=chunk_size,
            normalize_embeddings=normalize_embeddings,
            backend=backend,
            local_files_only=local_files_only,
        )
        if use_cache:
            cache_save(query_cache_path, query_embeddings)

    return doc_embeddings, query_embeddings


def _are_rows_unit_norm(
    x: np.ndarray,
    *,
    atol: float = 2e-2,
    max_rows: int = 256,
) -> bool:
    """Return whether the sampled rows look unit-normalized."""
    if x.ndim != 2 or x.shape[0] == 0:
        return False

    sample = x if x.shape[0] <= max_rows else x[:max_rows]
    norms = np.linalg.norm(sample, axis=1)
    return np.all(np.isfinite(norms)) and np.allclose(norms, 1.0, atol=atol)


def retrieve_embeddings(
    doc_emb: np.ndarray,
    query_emb: np.ndarray,
    k: int,
    assume_normalized: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the top-k documents per query with cosine similarity.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    doc_emb = np.asarray(doc_emb, dtype=np.float32)
    query_emb = np.asarray(query_emb, dtype=np.float32)

    if doc_emb.ndim != 2 or query_emb.ndim != 2:
        raise ValueError("doc_emb and query_emb must be 2D arrays.")
    if doc_emb.shape[1] != query_emb.shape[1]:
        raise ValueError("doc_emb and query_emb must have the same embedding dimension.")
    if k > doc_emb.shape[0]:
        raise ValueError("k cannot be greater than the number of documents.")

    n_queries = query_emb.shape[0]
    if n_queries == 0:
        return (
            np.empty((0, k), dtype=np.int64),
            np.empty((0, k), dtype=np.float32),
        )

    if assume_normalized and _are_rows_unit_norm(doc_emb) and _are_rows_unit_norm(query_emb):
        similarities = query_emb @ doc_emb.T
    else:
        doc_norm = np.linalg.norm(doc_emb, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        doc_unit = doc_emb / np.clip(doc_norm, 1e-12, None)
        query_unit = query_emb / np.clip(query_norm, 1e-12, None)
        similarities = query_unit @ doc_unit.T

    topk_unsorted_idx = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
    topk_unsorted_scores = np.take_along_axis(similarities, topk_unsorted_idx, axis=1)
    rerank = np.argsort(-topk_unsorted_scores, axis=1)

    topk_indices = np.take_along_axis(topk_unsorted_idx, rerank, axis=1)
    topk_scores = np.take_along_axis(topk_unsorted_scores, rerank, axis=1)

    return topk_indices, topk_scores
