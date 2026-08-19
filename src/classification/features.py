"""Feature extraction helpers for Phase 2 classification."""

from __future__ import annotations

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.config import Phase2Config
from src.retrieval.embeddings import encode_texts

_cfg = Phase2Config()


def extract_texts_and_labels(
    docs: list[dict],
    queries: list[dict] | None = None,
    text_field: str = "content",
) -> tuple[list[str], list[str]]:
    """
    Extract texts and labels from documents and optional queries.

    Parameters
    ----------
    docs : list[dict]
        Documents containing `text_field` and `category`.
    queries : list[dict] | None, default=None
        Optional query records to append.
    text_field : str, default="content"
        Field used as input text.

    Returns
    -------
    tuple[list[str], list[str]]
        Texts and aligned category labels.
    """
    texts = []
    labels = []
    for doc in docs:
        text = doc.get(text_field)
        label = doc.get("category")
        texts.append(text)
        labels.append(label)
    if queries:
        for query in queries:
            text = query.get(text_field)
            label = query.get("category")
            texts.append(text)
            labels.append(label)

    return texts, labels


def stratified_split(
    texts: list[str],
    labels: list[str],
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[
    list[str], list[str],
    list[str], list[str],
    list[str], list[str],
]:
    """
    Split texts and labels into train, validation, and test sets.

    Parameters
    ----------
    texts : list[str]
        Raw text samples.
    labels : list[str]
        Labels aligned with `texts`.
    val_size : float, default=0.15
        Validation split ratio.
    test_size : float, default=0.15
        Test split ratio.
    random_state : int, default=42
        Random seed used by `train_test_split`.

    Returns
    -------
    tuple
        `(X_train, y_train, X_val, y_val, X_test, y_test)`.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels,
        test_size = test_size,
        stratify = labels,
        random_state = random_state
    )

    val_fraction = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size = val_fraction,
        stratify = y_temp,
        random_state = random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test



def build_tfidf_features(
    X_train: list[str],
    X_val: list[str] | None = None,
    X_test: list[str] | None = None,
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
) -> tuple:
    """
    Build TF-IDF feature matrices from text inputs.

    Parameters
    ----------
    X_train : list[str]
        Training texts.
    X_val : list[str] | None, default=None
        Optional validation texts.
    X_test : list[str] | None, default=None
        Optional test texts.
    max_features : int, default=50_000
        Maximum vocabulary size.
    ngram_range : tuple[int, int], default=(1, 2)
        N-gram range passed to `TfidfVectorizer`.
    sublinear_tf : bool, default=True
        Whether to use sublinear term-frequency scaling.

    Returns
    -------
    tuple
        Tuple starting with `(vectorizer, X_train_mat)` and extended with
        transformed validation and test matrices when provided.
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf
    )

    X_train_mat = vec.fit_transform(X_train)
    result = (vec, X_train_mat)

    if X_val is not None:
        X_val_mat = vec.transform(X_val)
        result += (X_val_mat,)
    
    if X_test is not None:
        X_test_mat = vec.transform(X_test)
        result += (X_test_mat,)
    
    return result


def build_count_features(
    X_train: list[str],
    X_val: list[str] | None = None,
    X_test: list[str] | None = None,
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple:
    """
    Build count-based feature matrices from text inputs.
    """
    vec = CountVectorizer(
        max_features = max_features,
        ngram_range = ngram_range
    )

    X_train_mat = vec.fit_transform(X_train)
    result = (vec, X_train_mat)

    if X_val is not None:
        X_val_mat = vec.transform(X_val)
        result += (X_val_mat,)
    
    if X_test is not None:
        X_test_mat = vec.transform(X_test)
        result += (X_test_mat,)
    
    return result


def build_embedding_features(
    X_train: list[str],
    X_val: list[str] | None = None,
    X_test: list[str] | None = None,
    model_name: str = _cfg.classification_embedding_model_name,
    batch_size: int = _cfg.classification_embedding_batch_size,
) -> tuple:
    """
    Build dense embedding matrices from text inputs.
    """
    X_train_mat = encode_texts(
        model_name = model_name,
        texts = X_train,
        batch_size = batch_size,
        device = None
        )
    result = (None, X_train_mat)

    if X_val is not None:
        X_val_mat = encode_texts(
        model_name = model_name,
        texts = X_val,
        batch_size = batch_size,
        device = None
        )
        result += (X_val_mat,)
    
    if X_test is not None:
        X_test_mat = encode_texts(
        model_name = model_name,
        texts = X_test,
        batch_size = batch_size,
        device = None
        )
        result += (X_test_mat,)
    
    return result
