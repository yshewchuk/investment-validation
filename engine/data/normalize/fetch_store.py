"""Reading Tier-1 fetch-store payloads back out.

The grandfathered pulls and the new fetch wrapper store the *same* ORATS rows in
two different envelopes:

* legacy chain files wrap them —
  ``{"entry_date": ..., "tickers": [...], "rows": [...]}`` — because the old
  puller unwrapped the response and added its own context;
* the fetch wrapper stores the response **verbatim**, which for ORATS is
  ``{"data": [...]}``, because Tier 1's contract is "every byte as received".

Both are right for their purpose, and the difference must not reach the
normalizers: a rebuild has to see one stream of rows regardless of which pull
generation produced them. This module is that adapter.

Without it, Tier 1 is write-only for everything the new wrapper fetches — the
Sep-1 pull would land 16,000 calls' worth of chains that no rebuild could read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from engine.data.fetch import CachedEntry, iter_cached

__all__ = [
    "orats_rows",
    "iter_orats_rows",
    "SourceRows",
]

#: ORATS wraps every historical payload in a ``data`` list. A response that is
#: already a bare list (some endpoints, and hand-made fixtures) is accepted too.
ORATS_ENVELOPE_KEY = "data"


def orats_rows(payload: Any) -> list[dict]:
    """Extract the row list from an ORATS response body.

    Tolerant by design: a shape this does not recognize yields no rows rather
    than raising, because a single odd payload must not be able to abort a
    rebuild over the whole store. Anything skipped shows up as a coverage gap,
    which is visible; an exception mid-rebuild is not more informative.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get(ORATS_ENVELOPE_KEY)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        # The legacy chain envelope, in case a caller hands one over.
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


class SourceRows:
    """Rows from one Tier-1 payload, plus where they came from.

    ``source_id`` lands in ``src_file`` so a Tier-2 row can always be traced
    back to the exact payload that produced it, whichever pull generation that
    was.
    """

    def __init__(self, source_id: str, rows: list[dict], params: dict | None = None):
        self.source_id = source_id
        self.rows = rows
        self.params = params or {}

    def __len__(self) -> int:
        return len(self.rows)


def iter_orats_rows(
    endpoint: str,
    *,
    root: Path | None = None,
) -> Iterator[SourceRows]:
    """Stream ``(source_id, rows)`` for every cached response on one endpoint."""
    for entry in iter_cached("orats", endpoint, root=root):
        try:
            rows = orats_rows(entry.json())
        except (OSError, ValueError):
            continue
        if rows:
            yield SourceRows(entry.source_id, rows, entry.params)
