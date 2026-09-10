#!/usr/bin/env python3
"""A fixed, importable mean-ensemble wrapper for multi-seed artifacts.

Joblib pickles classes by reference, so an ensemble stored inside a model
artifact must live in an importable module. This is that module: a thin
container around fitted estimators whose ``predict`` is the mean of their
predictions — the five-seed average every multi-seed artifact in this program
is built on."""
from __future__ import annotations

from typing import Any, Sequence


class MeanEnsemble:
    """``predict(X)`` is the unweighted mean of the member predictions."""

    def __init__(self, models: Sequence[Any]):
        self.models = list(models)

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        first = preds[0]
        out = first.copy() if hasattr(first, "copy") else first
        for p in preds[1:]:
            out = out + p
        return out / len(preds)

    def __len__(self) -> int:
        return len(self.models)
