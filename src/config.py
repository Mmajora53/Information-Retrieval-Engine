"""Shared configuration objects and runtime path helpers."""

from __future__ import annotations 

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


PHASE1_SUBMISSION_TOP_K = 100
PHASE1_SUBMISSION_CATEGORY = "?"
PHASE1_RETRIEVAL_METHOD = "hybrid_bm25_embeddings"

RANDOM_SEED: int = 42
"""Global seed used across the project."""

PHASE2_OUTPUT_DIR: str = "outputs/runs/phase2"
"""Directory where Phase 2 results (metrics, predictions) are saved."""


@dataclass(slots=True)
class RuntimePaths:
    """Resolved filesystem paths used by the pipelines."""

    mode: str
    cwd: Path
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    cache_dir: Path
    output_submission: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "cwd": str(self.cwd),
            "project_root": str(self.project_root),
            "raw_dir": str(self.raw_dir),
            "processed_dir": str(self.processed_dir),
            "cache_dir": str(self.cache_dir),
            "output_submission": str(self.output_submission),
        }


@dataclass(slots=True)
class Phase1HybridSubmissionConfig:
    """Configuration for the Phase 1 hybrid submission pipeline."""

    retrieval_method: str = PHASE1_RETRIEVAL_METHOD
    top_k: int = PHASE1_SUBMISSION_TOP_K
    text_field: str = "content"
    category_value: str = PHASE1_SUBMISSION_CATEGORY

    use_processed_if_available: bool = True
    clean_content: bool = False
    max_docs: int | None = None

    embedding_model_name: str = "all-MiniLM-L12-v2"
    embedding_batch_size: int = 64
    show_progress_bar: bool = True
    quiet_third_party_logs: bool = True

    embedding_device: str | None = "auto"
    embedding_precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32"
    embedding_max_seq_length: int | None = None
    embedding_truncate_dim: int | None = None
    embedding_chunk_size: int | None = None
    embedding_normalize: bool = True
    embedding_backend: Literal["torch", "onnx", "openvino"] = "torch"
    embedding_local_files_only: bool = False

    doc_text_truncate_chars: int | None = None
    query_text_truncate_chars: int | None = None

    hybrid_bm25_method: str = "plus"
    hybrid_candidate_multiplier: int = 1
    hybrid_rrf_k: int = 20
    hybrid_weight_embeddings: float = 3.5
    hybrid_weight_bm25: float = 0.5

    def validate_phase1(self) -> None:
        if self.retrieval_method != PHASE1_RETRIEVAL_METHOD:
            raise ValueError(
                f"Only '{PHASE1_RETRIEVAL_METHOD}' is supported for Phase 1 submission."
            )
        if self.top_k != PHASE1_SUBMISSION_TOP_K:
            raise ValueError(
                f"Phase 1 Kaggle submission expects TOP_K={PHASE1_SUBMISSION_TOP_K}, got {self.top_k}."
            )
        if self.category_value != PHASE1_SUBMISSION_CATEGORY:
            raise ValueError(
                f"Phase 1 submission expects CATEGORY_VALUE='{PHASE1_SUBMISSION_CATEGORY}', "
                f"got {self.category_value!r}."
            )
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be > 0.")
        if self.hybrid_candidate_multiplier <= 0:
            raise ValueError("hybrid_candidate_multiplier must be > 0.")
        if self.hybrid_rrf_k <= 0:
            raise ValueError("hybrid_rrf_k must be > 0.")
        if self.hybrid_weight_embeddings <= 0 or self.hybrid_weight_bm25 <= 0:
            raise ValueError("Hybrid RRF weights must be > 0.")
        if self.max_docs is not None and self.max_docs <= 0:
            raise ValueError("max_docs must be > 0 when provided.")
        if (
            self.doc_text_truncate_chars is not None
            and self.doc_text_truncate_chars <= 0
        ):
            raise ValueError("doc_text_truncate_chars must be > 0 when provided.")
        if (
            self.query_text_truncate_chars is not None
            and self.query_text_truncate_chars <= 0
        ):
            raise ValueError("query_text_truncate_chars must be > 0 when provided.")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _has_files(base: Path, filenames: list[str]) -> bool:
    base = Path(base)
    return all((base / name).exists() for name in filenames)


def _looks_like_project_root(path: Path) -> bool:
    """
    Return whether a path looks like the repository root.
    """
    path = Path(path)
    return (path / "src").is_dir() and (path / "data").is_dir()


def _find_project_root(start: Path) -> Path | None:
    """Walk upwards from `start` and return the first matching project root."""
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if _looks_like_project_root(candidate):
            return candidate
    return None


def resolve_runtime_paths(cwd: Path | None = None) -> RuntimePaths:
    """
    Resolve runtime paths with a standalone-first strategy.
    """
    cwd = Path.cwd().resolve() if cwd is None else Path(cwd).resolve()
    raw_needed = ["docs.json", "queries_test.json"]
    processed_needed = ["docs_with_content.json", "queries_test_with_content.json"]

    if _has_files(cwd, processed_needed) or _has_files(cwd, raw_needed):
        return RuntimePaths(
            mode="standalone",
            cwd=cwd,
            project_root=cwd,
            raw_dir=cwd,
            processed_dir=cwd,
            cache_dir=cwd / "cache",
            output_submission=cwd / "submission.csv",
        )

    if _has_files(cwd / "processed", processed_needed) or _has_files(cwd / "raw", raw_needed):
        return RuntimePaths(
            mode="standalone_subdirs",
            cwd=cwd,
            project_root=cwd,
            raw_dir=cwd / "raw",
            processed_dir=cwd / "processed",
            cache_dir=cwd / "cache",
            output_submission=cwd / "submission.csv",
        )

    project_root = _find_project_root(cwd)
    if project_root is None:
        raise FileNotFoundError(
            "Could not resolve project root from cwd. Expected a parent directory "
            "containing at least 'src/' and 'data/'.\n"
            f"cwd={cwd}"
        )
    return RuntimePaths(
        mode="project",
        cwd=cwd,
        project_root=project_root,
        raw_dir=project_root / "data" / "raw",
        processed_dir=project_root / "data" / "processed",
        cache_dir=project_root / "data" / "cache",
        output_submission=project_root / "outputs" / "submissions" / "submission.csv",
    )


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_mps_available() -> bool:
    try:
        import torch

        return bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def detect_best_embedding_device() -> str:
    """Pick the best available embedding device in priority order."""
    if _torch_cuda_available():
        return "cuda"
    if _torch_mps_available():
        return "mps"
    return "cpu"


@dataclass(slots=True)
class Phase2Config:
    """
    Configuration for the Phase 2 classification and reranking pipeline.
    """
    retrieval_method: str = PHASE1_RETRIEVAL_METHOD
    top_k: int = PHASE1_SUBMISSION_TOP_K
    text_field: str = "content"

    embedding_model_name: str = "all-MiniLM-L12-v2"
    embedding_batch_size: int = 64
    show_progress_bar: bool = True
    embedding_device: str | None = "auto"
    classification_embedding_model_name: str = "all-MiniLM-L12-v2"
    classification_embedding_batch_size: int = 64

    classifier_method: str = "svc"
    """Classifier family used for category prediction."""

    feature_method: str = "tfidf"
    """Feature family used for classification."""

    classifier_val_size: float = 0.15
    classifier_test_size: float = 0.15

    rerank_strategy: str = "hard_filter"
    """Reranking strategy applied after retrieval."""

    rerank_boost_factor: float = 0.5
    """Boost multiplier used by `soft_boost`."""

    hybrid_rrf_k: int = 30
    hybrid_weight_embeddings: float = 3.5
    hybrid_weight_bm25: float = 0.5

    # RETRIEVAL_K = top_k * retrieval_k_multiplier = 100 * 10 = 1000 candidates
    retrieval_k_multiplier: int = 10

    # PRF: top-k BM25 docs → extract tags → append to query
    prf_top_k: int = 3
    prf_max_tags: int = 10

    random_seed: int = RANDOM_SEED

    output_dir: str = PHASE2_OUTPUT_DIR

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_embedding_device(requested_device: str | None) -> tuple[str, str | None]:
    """
    Resolve the embedding device with graceful fallback.
    """
    requested = None if requested_device is None else str(requested_device).lower().strip()

    if requested in (None, "", "auto"):
        resolved = detect_best_embedding_device()
        return resolved, f"Auto-detected embedding_device='{resolved}'."

    if requested == "cpu":
        return "cpu", None

    if requested == "cuda":
        if _torch_cuda_available():
            return "cuda", None
        return "cpu", "Requested embedding_device='cuda' but CUDA is unavailable. Falling back to 'cpu'."

    if requested == "mps":
        if _torch_mps_available():
            return "mps", None
        return "cpu", "Requested embedding_device='mps' but MPS is unavailable. Falling back to 'cpu'."

    raise ValueError(
        "embedding_device must be one of: None, 'auto', 'cpu', 'mps', 'cuda'. "
        f"Got {requested_device!r}."
    )
