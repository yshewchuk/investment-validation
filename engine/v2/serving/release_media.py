"""Release-file content types for the operations release route (P6-4).

One wrapper over :func:`mimetypes.guess_type` so ``operations._release_route``
serves exactly the same ``Content-Type`` it always has: ``operations.py``
imports this module and the derivation projection together through one
``from . import ...`` statement, the sibling-merge pattern
``guides/rearchitecture_tech_debt.md`` TD-4 records, keeping the transport
inside the eight-module fan-out budget.
"""
from __future__ import annotations

import mimetypes

__all__ = ["content_type_for"]


def content_type_for(filename: str) -> str:
    """``mimetypes.guess_type``'s answer with the generic fallback."""
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"
