"""Ops failures carry a contract ``Problem``, never a bare string.

Every refusal the supervisor produces is one of the registered
``FAILURE_CODES`` (phase-1 guide §5.3), so a caller branches on a stable code
and a category rather than parsing a message. Messages and details are written
here, by ops, and never interpolate a submitted value: a parameter, URL or
exception text may carry a credential (§5.2).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from engine.v2.contracts import FAILURE_CODES, Problem

__all__ = ["OpsError", "fail", "make_problem"]


class OpsError(Exception):
    """A refused or failed operation, with its envelope."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(f"{problem.code}: {problem.message}")
        self.problem = problem

    @property
    def code(self) -> str:
        return self.problem.code


def make_problem(code: str, message: str, *, stage: str | None = None,
                 details: Mapping[str, Any] | None = None,
                 retry_after_seconds: int | None = None,
                 dependency_refs: Iterable[str] = ()) -> Problem:
    """A ``Problem`` whose category and retryability come from the registry."""
    if code not in FAILURE_CODES:
        raise ValueError(f"{code!r} is not a registered failure code")
    category, retryable = FAILURE_CODES[code]
    return Problem(code=code, category=category, retryable=retryable, message=message,
                   stage=stage, details=dict(details or {}),
                   retry_after_seconds=retry_after_seconds,
                   dependency_refs=tuple(dependency_refs))


def fail(code: str, message: str, **kwargs: Any) -> OpsError:
    """``raise fail(...)`` — the one way ops refuses."""
    return OpsError(make_problem(code, message, **kwargs))
