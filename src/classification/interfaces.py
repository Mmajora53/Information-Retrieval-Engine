"""Shared interfaces and constants used across the Phase 2 pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

CATEGORIES: list[str] = ["android", "gaming", "programmers", "tex", "unix"]
"""All possible category labels in the dataset (documents and queries)."""


@runtime_checkable
class ClassifierProtocol(Protocol):
    """
    Minimal interface implemented by any classifier used in Phase 2.
    """

    @property
    def classes_(self) -> list[str]:
        """
        Ordered category labels aligned with `predict_proba` columns.
        """
        ...

    def fit(self, X, y: list[str]) -> None:
        """
        Fit the classifier on a feature matrix and label vector.
        """
        ...

    def predict(self, X) -> list[str]:
        """
        Predict one category label per sample.
        """
        ...

    def predict_proba(self, X) -> np.ndarray | None:
        """
        Predict class probabilities when supported by the model.
        """
        ...
