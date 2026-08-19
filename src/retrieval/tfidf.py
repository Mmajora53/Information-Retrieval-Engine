"""TF-IDF indexing and retrieval helpers."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


def fit_tfidf(
    docs: List[dict],
    text_field: str = "content",
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """
    Fit a TF-IDF vectorizer on the document corpus.

    Parameters
    ----------
    docs : list[dict]
        Documents to index.
    text_field : str, default="content"
        Field used as input text.

    Returns
    -------
    tuple[TfidfVectorizer, csr_matrix]
        Fitted vectorizer and document-term matrix.
    """
    texts = [d[text_field] for d in docs]
    # Keep the defaults simple and retrieval-friendly.
    vectorizer = TfidfVectorizer(stop_words=None, min_df=1, sublinear_tf=True)
    doc_matrix = vectorizer.fit_transform(texts)

    return vectorizer, doc_matrix


def retrieve_tfidf(
    vectorizer: TfidfVectorizer,
    doc_matrix: spmatrix,
    queries: List[dict],
    k: int,
    text_field: str = "content",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the top-k documents for each query with cosine similarity.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer.
    doc_matrix : spmatrix
        TF-IDF matrix for documents.
    queries : list[dict]
        Query records.
    k : int
        Number of documents to return per query.
    text_field : str, default="content"
        Field used as query text.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Retrieved indices and cosine scores.
    """

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    n_docs = int(doc_matrix.shape[0])
    if k > n_docs:
        raise ValueError("k cannot be greater than the number of documents.")

    if len(queries) == 0:
        return (
            np.empty((0, k), dtype=np.int64),
            np.empty((0, k), dtype=np.float64),
        )

    query_texts = [q[text_field] for q in queries]
    query_mat = vectorizer.transform(query_texts)
    cos_sim = cosine_similarity(query_mat, doc_matrix)
    topk_indices = np.argsort(-cos_sim, axis=1)[:, :k]
    topk_scores = np.take_along_axis(cos_sim, topk_indices, axis=1)

    return topk_indices, topk_scores


def map_indices_to_docids(topk_indices: ArrayLike, docs: List[dict]) -> List[List[str]]:
    """
    Convert retrieved document indices into document IDs.
    """
    from src.evaluation.evaluate import indices_to_docids

    return indices_to_docids(topk_indices, docs, doc_id_field="id")
