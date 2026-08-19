"""CLI pipeline used to generate the Phase 1 Kaggle submission."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from src.config import (
    PHASE1_RETRIEVAL_METHOD,
    PHASE1_SUBMISSION_CATEGORY,
    PHASE1_SUBMISSION_TOP_K,
    Phase1HybridSubmissionConfig,
    RuntimePaths,
    resolve_embedding_device,
    resolve_runtime_paths,
)
from src.data.load import load_json
from src.data.preprocess import add_content_field
from src.evaluation.evaluate import indices_to_docids
from src.kaggle.format import save_submission
from src.retrieval.hybrid import retrieve_hybrid_bm25_embeddings, truncate_text_field


def _print_section(title: str) -> None:
    line = "=" * max(12, len(title) + 4)
    print(f"\n{line}\n{title}\n{line}")


def _print_kv(key: str, value) -> None:
    print(f"{key:<28} {value}")


def load_submission_inputs(
    raw_dir: Path,
    processed_dir: Path,
    *,
    use_processed_if_available: bool = True,
    clean_content: bool = True,
    max_docs: int | None = None,
    text_field: str = "content",
    doc_text_truncate_chars: int | None = None,
    query_text_truncate_chars: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Load Phase 1 submission inputs from processed or raw files.
    """
    docs_processed_path = Path(processed_dir) / "docs_with_content.json"
    queries_test_processed_path = Path(processed_dir) / "queries_test_with_content.json"

    if (
        use_processed_if_available
        and docs_processed_path.exists()
        and queries_test_processed_path.exists()
    ):
        docs = load_json(docs_processed_path)
        queries_test = load_json(queries_test_processed_path)
        print("Loaded processed docs/test queries with content.")
    else:
        docs_raw_path = Path(raw_dir) / "docs.json"
        queries_test_raw_path = Path(raw_dir) / "queries_test.json"

        if not docs_raw_path.exists() or not queries_test_raw_path.exists():
            raise FileNotFoundError(
                "Could not find submission input files. Expected either processed files "
                "(docs_with_content.json + queries_test_with_content.json) or raw files "
                "(docs.json + queries_test.json) in the configured folder(s).\n"
                f"raw_dir={raw_dir}\nprocessed_dir={processed_dir}"
            )

        docs_raw = load_json(docs_raw_path)
        queries_test_raw = load_json(queries_test_raw_path)
        docs, queries_test = add_content_field(docs_raw, queries_test_raw, clean=clean_content)
        print("Processed files not found. Rebuilt content from raw docs.json + queries_test.json.")

    if max_docs is not None:
        docs = docs[:max_docs]

    docs = truncate_text_field(docs, text_field=text_field, max_chars=doc_text_truncate_chars)
    queries_test = truncate_text_field(
        queries_test,
        text_field=text_field,
        max_chars=query_text_truncate_chars,
    )
    return docs, queries_test


def run_retrieval_submission(
    docs: list[dict],
    queries_test: list[dict],
    config: Phase1HybridSubmissionConfig,
    runtime_paths: RuntimePaths,
) -> tuple[list[list[str]], float]:
    """
    Run the Phase 1 hybrid retrieval pipeline.
    """
    t0 = time.perf_counter()
    topk_indices, topk_scores = retrieve_hybrid_bm25_embeddings(
        docs,
        queries_test,
        top_k=config.top_k,
        text_field=config.text_field,
        embedding_model_name=config.embedding_model_name,
        embedding_batch_size=config.embedding_batch_size,
        show_progress_bar=config.show_progress_bar,
        cache_dir=runtime_paths.cache_dir,
        embedding_device=config.embedding_device,
        embedding_precision=config.embedding_precision,
        embedding_max_seq_length=config.embedding_max_seq_length,
        embedding_truncate_dim=config.embedding_truncate_dim,
        embedding_chunk_size=config.embedding_chunk_size,
        embedding_normalize=config.embedding_normalize,
        embedding_backend=config.embedding_backend,
        embedding_local_files_only=config.embedding_local_files_only,
        hybrid_bm25_method=config.hybrid_bm25_method,
        hybrid_candidate_multiplier=config.hybrid_candidate_multiplier,
        hybrid_rrf_k=config.hybrid_rrf_k,
        hybrid_weight_embeddings=config.hybrid_weight_embeddings,
        hybrid_weight_bm25=config.hybrid_weight_bm25,
    )

    pred_docids = indices_to_docids(topk_indices, docs, doc_id_field="id")
    elapsed = time.perf_counter() - t0

    _print_section("Retrieval Result")
    _print_kv("retrieval_method", PHASE1_RETRIEVAL_METHOD)
    _print_kv("topk_indices shape", topk_indices.shape)
    _print_kv("topk_scores shape", topk_scores.shape)
    _print_kv("retrieval_elapsed_s", f"{elapsed:.2f}")
    return pred_docids, elapsed


def verify_phase1_submission(
    submission_df,
    *,
    top_k: int,
    category_value: str,
    output_submission: Path,
) -> None:
    """
    Validate that a Phase 1 submission matches the expected Kaggle format.
    """
    if top_k != PHASE1_SUBMISSION_TOP_K:
        raise ValueError(
            f"Phase 1 Kaggle submission expects TOP_K={PHASE1_SUBMISSION_TOP_K}, got {top_k}"
        )
    if category_value != PHASE1_SUBMISSION_CATEGORY:
        raise ValueError(
            f"Phase 1 submission expects CATEGORY_VALUE='{PHASE1_SUBMISSION_CATEGORY}', "
            f"got {category_value!r}"
        )

    output_submission = Path(output_submission)
    if not output_submission.exists():
        raise FileNotFoundError("submission.csv was not created")
    if output_submission.stat().st_size <= 0:
        raise ValueError("submission.csv is empty")

    if list(submission_df.columns) != ["query_id", "relevant_doc_ids", "category"]:
        raise ValueError("Submission columns are invalid")

    parsed_docids = submission_df["relevant_doc_ids"].map(json.loads)

    if not submission_df["category"].eq(category_value).all():
        raise ValueError(
            "All rows in the submission must use the configured CATEGORY_VALUE."
        )
    if not parsed_docids.map(lambda x: isinstance(x, list)).all():
        raise ValueError("relevant_doc_ids must be JSON lists")
    if not parsed_docids.map(len).eq(top_k).all():
        raise ValueError(f"Each row must contain exactly TOP_K={top_k} doc_ids")

    _print_section("Submission Checks")
    print("All submission checks passed.")
    _print_kv("submission_path", output_submission)


def run_phase1_submission_pipeline(
    config: Phase1HybridSubmissionConfig,
    runtime_paths: RuntimePaths,
):
    """
    Run the full Phase 1 submission pipeline.
    """
    config.validate_phase1()

    docs, queries_test = load_submission_inputs(
        raw_dir=runtime_paths.raw_dir,
        processed_dir=runtime_paths.processed_dir,
        use_processed_if_available=config.use_processed_if_available,
        clean_content=config.clean_content,
        max_docs=config.max_docs,
        text_field=config.text_field,
        doc_text_truncate_chars=config.doc_text_truncate_chars,
        query_text_truncate_chars=config.query_text_truncate_chars,
    )

    query_ids = [str(q["id"]) for q in queries_test]
    pred_docids, retrieval_elapsed = run_retrieval_submission(
        docs=docs,
        queries_test=queries_test,
        config=config,
        runtime_paths=runtime_paths,
    )

    submission_df = save_submission(
        query_ids=query_ids,
        pred_docids=pred_docids,
        output_path=runtime_paths.output_submission,
        top_k=config.top_k,
        category=config.category_value,
    )

    _print_section("Submission Saved")
    _print_kv("path", runtime_paths.output_submission)
    _print_kv("rows", len(submission_df))
    return submission_df, retrieval_elapsed


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer CLI value."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    return int(value)


def _parse_optional_str(value: str | None) -> str | None:
    """Parse an optional string CLI value."""
    if value is None:
        return None
    if value.lower() == "none":
        return None
    return value


def _configure_third_party_logging(quiet: bool) -> None:
    """
    Reduce noisy third-party logs during CLI runs.
    """
    if not quiet:
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for logger_name in (
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    try:
        from sentence_transformers import LoggingHandler  # type: ignore

        logging.getLogger(LoggingHandler.__module__.split(".")[0]).setLevel(logging.ERROR)
    except Exception:
        pass


def _build_parser() -> argparse.ArgumentParser:
    cfg = Phase1HybridSubmissionConfig()
    parser = argparse.ArgumentParser(
        description="Phase 1 Kaggle submission pipeline (hybrid BM25 + embeddings + RRF) from src/"
    )

    parser.add_argument("--cwd", type=Path, default=None, help="Base cwd for auto path resolution (default: current cwd).")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Override raw input directory.")
    parser.add_argument("--processed-dir", type=Path, default=None, help="Override processed input directory.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override embeddings cache directory.")
    parser.add_argument("--output-submission", type=Path, default=None, help="Override output CSV path.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip strict Phase 1 submission checks after saving.")
    parser.add_argument("--top-k", type=int, default=cfg.top_k)
    parser.add_argument("--text-field", type=str, default=cfg.text_field)
    parser.add_argument("--category-value", type=str, default=cfg.category_value)

    parser.add_argument(
        "--use-processed-if-available",
        action=argparse.BooleanOptionalAction,
        default=cfg.use_processed_if_available,
    )
    parser.add_argument(
        "--clean-content",
        action=argparse.BooleanOptionalAction,
        default=cfg.clean_content,
    )
    parser.add_argument("--max-docs", type=_parse_optional_int, default=cfg.max_docs)
    parser.add_argument("--doc-text-truncate-chars", type=_parse_optional_int, default=cfg.doc_text_truncate_chars)
    parser.add_argument("--query-text-truncate-chars", type=_parse_optional_int, default=cfg.query_text_truncate_chars)

    parser.add_argument("--embedding-model-name", type=str, default=cfg.embedding_model_name)
    parser.add_argument("--embedding-batch-size", type=int, default=cfg.embedding_batch_size)
    parser.add_argument(
        "--show-progress-bar",
        action=argparse.BooleanOptionalAction,
        default=cfg.show_progress_bar,
    )
    parser.add_argument("--embedding-device", type=_parse_optional_str, default=cfg.embedding_device)
    parser.add_argument("--embedding-precision", type=str, default=cfg.embedding_precision)
    parser.add_argument("--embedding-max-seq-length", type=_parse_optional_int, default=cfg.embedding_max_seq_length)
    parser.add_argument("--embedding-truncate-dim", type=_parse_optional_int, default=cfg.embedding_truncate_dim)
    parser.add_argument("--embedding-chunk-size", type=_parse_optional_int, default=cfg.embedding_chunk_size)
    parser.add_argument(
        "--embedding-normalize",
        action=argparse.BooleanOptionalAction,
        default=cfg.embedding_normalize,
    )
    parser.add_argument("--embedding-backend", type=str, default=cfg.embedding_backend)
    parser.add_argument(
        "--embedding-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=cfg.embedding_local_files_only,
        help="Require local HF cache only (no network).",
    )

    parser.add_argument("--hybrid-bm25-method", type=str, default=cfg.hybrid_bm25_method)
    parser.add_argument("--hybrid-candidate-multiplier", type=int, default=cfg.hybrid_candidate_multiplier)
    parser.add_argument("--hybrid-rrf-k", type=int, default=cfg.hybrid_rrf_k)
    parser.add_argument("--hybrid-weight-embeddings", type=float, default=cfg.hybrid_weight_embeddings)
    parser.add_argument("--hybrid-weight-bm25", type=float, default=cfg.hybrid_weight_bm25)

    return parser


def _runtime_paths_from_args(args: argparse.Namespace) -> RuntimePaths:
    paths = resolve_runtime_paths(args.cwd)
    return RuntimePaths(
        mode=paths.mode,
        cwd=paths.cwd,
        project_root=paths.project_root,
        raw_dir=Path(args.raw_dir).resolve() if args.raw_dir else paths.raw_dir,
        processed_dir=Path(args.processed_dir).resolve() if args.processed_dir else paths.processed_dir,
        cache_dir=Path(args.cache_dir).resolve() if args.cache_dir else paths.cache_dir,
        output_submission=Path(args.output_submission).resolve() if args.output_submission else paths.output_submission,
    )


def _config_from_args(args: argparse.Namespace) -> Phase1HybridSubmissionConfig:
    return Phase1HybridSubmissionConfig(
        top_k=args.top_k,
        text_field=args.text_field,
        category_value=args.category_value,
        use_processed_if_available=args.use_processed_if_available,
        clean_content=args.clean_content,
        max_docs=args.max_docs,
        embedding_model_name=args.embedding_model_name,
        embedding_batch_size=args.embedding_batch_size,
        show_progress_bar=args.show_progress_bar,
        embedding_device=args.embedding_device,
        embedding_precision=args.embedding_precision,
        embedding_max_seq_length=args.embedding_max_seq_length,
        embedding_truncate_dim=args.embedding_truncate_dim,
        embedding_chunk_size=args.embedding_chunk_size,
        embedding_normalize=args.embedding_normalize,
        embedding_backend=args.embedding_backend,
        embedding_local_files_only=args.embedding_local_files_only,
        doc_text_truncate_chars=args.doc_text_truncate_chars,
        query_text_truncate_chars=args.query_text_truncate_chars,
        hybrid_bm25_method=args.hybrid_bm25_method,
        hybrid_candidate_multiplier=args.hybrid_candidate_multiplier,
        hybrid_rrf_k=args.hybrid_rrf_k,
        hybrid_weight_embeddings=args.hybrid_weight_embeddings,
        hybrid_weight_bm25=args.hybrid_weight_bm25,
    )


def _print_runtime_summary(config: Phase1HybridSubmissionConfig, runtime_paths: RuntimePaths) -> None:
    _print_section("Runtime Paths")
    _print_kv("runtime_mode", runtime_paths.mode)
    _print_kv("cwd", runtime_paths.cwd)
    _print_kv("project_root", runtime_paths.project_root)
    _print_kv("RAW_DIR", runtime_paths.raw_dir)
    _print_kv("PROCESSED_DIR", runtime_paths.processed_dir)
    _print_kv("CACHE_DIR", runtime_paths.cache_dir)
    _print_kv("OUTPUT_SUBMISSION", runtime_paths.output_submission)

    _print_section("Phase 1 Config")
    summary_keys = [
        "retrieval_method",
        "top_k",
        "text_field",
        "category_value",
        "use_processed_if_available",
        "clean_content",
        "max_docs",
        "embedding_model_name",
        "embedding_batch_size",
        "show_progress_bar",
        "quiet_third_party_logs",
        "embedding_device",
        "embedding_precision",
        "embedding_max_seq_length",
        "embedding_truncate_dim",
        "embedding_chunk_size",
        "embedding_normalize",
        "embedding_backend",
        "embedding_local_files_only",
        "doc_text_truncate_chars",
        "query_text_truncate_chars",
        "hybrid_bm25_method",
        "hybrid_candidate_multiplier",
        "hybrid_rrf_k",
        "hybrid_weight_embeddings",
        "hybrid_weight_bm25",
    ]
    cfg_dict = config.as_dict()
    for key in summary_keys:
        _print_kv(key, cfg_dict[key])


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = _config_from_args(args)
    runtime_paths = _runtime_paths_from_args(args)
    _configure_third_party_logging(config.quiet_third_party_logs)
    config.embedding_device, device_resolution_message = resolve_embedding_device(
        config.embedding_device
    )

    t0 = time.perf_counter()
    _print_section("Phase 1 Submission Pipeline")
    _print_runtime_summary(config, runtime_paths)
    if device_resolution_message:
        _print_section("Device Resolution")
        print(device_resolution_message)

    submission_df, retrieval_elapsed = run_phase1_submission_pipeline(config, runtime_paths)
    if not args.skip_checks:
        verify_phase1_submission(
            submission_df,
            top_k=config.top_k,
            category_value=config.category_value,
            output_submission=runtime_paths.output_submission,
        )

    total_elapsed = time.perf_counter() - t0
    minutes, seconds = divmod(int(total_elapsed), 60)
    _print_section("Timing")
    _print_kv("total_pipeline_time", f"{minutes}min {seconds}s")
    _print_kv("retrieval_elapsed_s", f"{retrieval_elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
