"""Data failures carry a contract ``Problem``, never a bare string.

Phase-2 guide §11: every refusal this package raises is one of the registered
``DATA_FAILURE_CODES`` (``engine.v2.contracts.data``), so a caller branches on
a stable code and category rather than parsing a message. Mirrors
``engine.v2.ops.errors`` exactly, one layer down.

Messages are written here, by this package, and are redacted on the way in:
no legacy filesystem path and no row value ever reaches one (phase-2 guide
§7.2's "the message is redacted"). A column or table *name* is schema
metadata, not a row value, and may appear.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from engine.v2.contracts import DATA_FAILURE_CODES, Problem

__all__ = ["DataError", "fail", "make_problem"]


class DataError(Exception):
    """A refused or failed data operation, with its envelope."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(f"{problem.code}: {problem.message}")
        self.problem = problem

    @property
    def code(self) -> str:
        return self.problem.code


def make_problem(code: str, message: str, *, stage: str | None = None,
                 details: Mapping[str, Any] | None = None) -> Problem:
    """A ``Problem`` whose category and retryability come from the registry."""
    if code not in DATA_FAILURE_CODES:
        raise ValueError(f"{code!r} is not a registered data failure code")
    category, retryable = DATA_FAILURE_CODES[code]
    return Problem(code=code, category=category, retryable=retryable, message=message,
                   stage=stage, details=dict(details or {}))


def fail(code: str, message: str, **kwargs: Any) -> DataError:
    """``raise fail(...)`` — the one way this package refuses."""
    return DataError(make_problem(code, message, **kwargs))
