"""BM25 indexing and retrieval helpers."""

import re
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus

_TECH_TOKEN_RE = re.compile(
    r"(?:\.[a-z0-9]+|[a-z0-9]+(?:[._-][a-z0-9]+)*(?:\+\+|#)?)"
)


def tokenize(text: str) -> List[str]:
    """
    Tokenize text while preserving common technical tokens.
    """
    return _TECH_TOKEN_RE.findall(text.lower())


def fit_bm25(
    docs: List[Dict],
    text_field: str = "content",
    method: str = "plus",
) -> object:
    """
    Fit a BM25 model on the document corpus.

    Parameters
    ----------
    docs : list[dict]
        Documents to index.
    text_field : str, default="content"
        Field used as input text.
    method : str, default="plus"
        BM25 variant: `"plus"` or `"okapi"`.

    Returns
    -------
    object
        Fitted BM25 model.
    """
    texts = [doc[text_field] for doc in docs]
    tokenized_texts = [tokenize(text) for text in texts]

    if method == "plus":
        bm25_model = BM25Plus(tokenized_texts)
    elif method == "okapi":
        bm25_model = BM25Okapi(tokenized_texts)
    else:
        raise ValueError(f"method must be 'plus' or 'okapi', not {method!r}")

    return bm25_model


def retrieve_bm25(
    bm25_model: object,
    docs: List[Dict],
    queries: List[Dict],
    k: int,
    text_field: str = "content",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the top-k documents for each query with BM25.

    Parameters
    ----------
    bm25_model : object
        Fitted BM25 model.
    docs : list[dict]
        Documents aligned with the BM25 index.
    queries : list[dict]
        Query records.
    k : int
        Number of documents to return per query.
    text_field : str, default="content"
        Field used as query text.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Retrieved indices and BM25 scores.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    n_docs = len(docs)
    if k > n_docs:
        raise ValueError("k cannot be greater than the number of documents.")

    n_queries = len(queries)
    if n_queries == 0:
        return (
            np.empty((0, k), dtype=np.int64),
            np.empty((0, k), dtype=np.float64),
        )

    topk_indices = np.zeros((n_queries, k), dtype=int)
    topk_scores = np.zeros((n_queries, k), dtype=float)

    for i, query in enumerate(tqdm(queries, desc="BM25 retrieval")):
        tokenized_query = tokenize(query[text_field])
        scores = np.array(bm25_model.get_scores(tokenized_query))
        topk = np.argsort(scores)[::-1][:k]

        topk_indices[i, : len(topk)] = topk
        topk_scores[i, : len(topk)] = scores[topk]

    return topk_indices, topk_scores


def map_indices_to_docids(topk_indices: np.ndarray, docs: List[dict]) -> List[List[str]]:
    """
    Convert top-k document indices into document IDs.
    """
    from src.evaluation.evaluate import indices_to_docids

    return indices_to_docids(topk_indices, docs, doc_id_field="id")
