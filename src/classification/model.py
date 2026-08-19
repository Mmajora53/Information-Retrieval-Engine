"""Classifier wrapper used by the Phase 2 pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.config import RANDOM_SEED


class Classifier:
    """
    Unified wrapper around the supported sklearn classifiers.

    Parameters
    ----------
    method : str, default="logreg"
        One of `"nb"`, `"svc"`, `"logreg"`, or `"mlp"`.
    **kwargs
        Extra keyword arguments forwarded to the underlying estimator.
    """

    SUPPORTED_METHODS = ("nb", "svc", "logreg", "mlp")

    def __init__(self, method: str = "logreg", **kwargs) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {self.SUPPORTED_METHODS}, got {method!r}."
            )
        self.method = method
        self._kwargs = kwargs
        self._model = None
        self._classes: list[str] = []

    @property
    def classes_(self) -> list[str]:
        """Ordered category labels, aligned with predict_proba columns."""
        if not self._classes:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")
        return self._classes

    def fit(self, X, y: list[str]) -> None:
        """
        Fit the classifier on a feature matrix and label vector.
        """
        kwargs = self._kwargs.copy()
        if self.method in ("svc", "logreg", "mlp"):
            kwargs.setdefault("random_state", RANDOM_SEED)

        if self.method == "nb":
            self._model = MultinomialNB(**kwargs)
        elif self.method == "svc":
            self._model = LinearSVC(**kwargs)
        elif self.method == "logreg":
            self._model = LogisticRegression(**kwargs)
        elif self.method == "mlp":
            self._model = MLPClassifier(**kwargs)
        
        self._model.fit(X, y)
        self._classes = self._model.classes_.tolist()

    def predict(self, X) -> list[str]:
        """
        Predict category labels for the input samples.
        """
        if self._model is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")
        return self._model.predict(X).tolist()

    def predict_proba(self, X) -> np.ndarray | None:
        """
        Predict class probabilities when the underlying model supports them.
        """
        if self._model is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")
        if self.method != "svc":
            return self._model.predict_proba(X)
        return None

    def __repr__(self) -> str:
        fitted = self._model is not None
        return f"Classifier(method={self.method!r}, fitted={fitted})"
