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
    "ORATS_ENVELOPE_KEY",
    "payload_shape",
    "orats_rows",
    "iter_orats_rows",
    "SourceRows",
]

#: ORATS wraps every historical payload in a ``data`` list. A response that is
#: already a bare list (some endpoints, and hand-made fixtures) is accepted too.
ORATS_ENVELOPE_KEY = "data"


def payload_shape(payload: Any) -> tuple[str, list[dict]]:
    """Classify an ORATS body and extract its rows.

    Returns ``(shape, rows)`` where shape is one of:

    * ``"data"`` / ``"rows"`` / ``"list"`` — a recognized envelope; ``rows``
      holds its dict members (may be empty: a recognized ``data: []`` is a
      legitimate zero-row answer, not an error);
    * ``"empty"`` — ``None``;
    * ``"unrecognized"`` — anything else.

    The distinction the caller cannot make from the row list alone: zero rows
    can mean "the API answered and there was nothing" or "a byte that cost
    quota parsed to nothing we understand". Only the second is a silent
    failure, so only the second gets counted and flagged.
    """
    if payload is None:
        return "empty", []
    if isinstance(payload, list):
        return "list", [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get(ORATS_ENVELOPE_KEY)
        if isinstance(rows, list):
            return "data", [row for row in rows if isinstance(row, dict)]
        # The legacy chain envelope, in case a caller hands one over.
        rows = payload.get("rows")
        if isinstance(rows, list):
            return "rows", [row for row in rows if isinstance(row, dict)]
    return "unrecognized", []


def orats_rows(payload: Any) -> list[dict]:
    """Extract the row list from an ORATS response body.

    Tolerant by design: a shape this does not recognize yields no rows rather
    than raising, because a single odd payload must not be able to abort a
    rebuild over the whole store. The counting and flagging of unrecognized
    shapes happens one level up, in :func:`iter_orats_rows` — tolerance here
    must not mean silence there.
    """
    return payload_shape(payload)[1]


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


def _flag_bad_payload(source_id: str, reason: str, detail: str) -> None:
    """Quarantine-flag a payload that cost quota but produced no rows.

    Mirrors ``validate.quarantine``: the raw bytes stay in Tier 1 (every byte
    ever fetched is kept), and a pointer plus the reason lands in the
    quarantine dir so the silent-failure mode — a rebuild that sees zero rows
    and raises nothing — becomes a count and a named offender instead.
    """
    from engine.data import validate

    validate.quarantine(source_id, reason, {"detail": detail})


def iter_orats_rows(
    endpoint: str,
    *,
    root: Path | None = None,
    stats: dict | None = None,
) -> Iterator[SourceRows]:
    """Stream ``(source_id, rows)`` for every cached response on one endpoint.

    ``stats`` is an optional mutable counter the caller passes to make the
    tolerated failures VISIBLE rather than silent. A body that parses to a
    shape this does not recognize, or does not parse at all, cost quota and
    produced nothing: it is counted (``stats["unrecognized"]`` /
    ``stats["unreadable"]``) and quarantine-flagged, the way ``validate.py``
    treats a bad file. A recognized envelope that simply holds zero rows is a
    legitimate empty answer and is only counted (``stats["empty"]``).
    """
    if stats is not None:
        stats.setdefault("scanned", 0)
        stats.setdefault("payloads", 0)
        stats.setdefault("empty", 0)
        stats.setdefault("unrecognized", 0)
        stats.setdefault("unreadable", 0)

    for entry in iter_cached("orats", endpoint, root=root):
        outcome, source = classify_entry(entry, stats)
        if outcome == "payload":
            yield source


def count_outcome(stats: dict | None, outcome: str) -> None:
    """Record one scanned entry's outcome in ``stats`` (see :func:`iter_orats_rows`)."""
    if stats is None:
        return
    stats["scanned"] = stats.get("scanned", 0) + 1
    key = {"payload": "payloads"}.get(outcome, outcome)
    stats[key] = stats.get(key, 0) + 1


def classify_entry(entry, stats: dict | None = None) -> tuple[str, SourceRows | None]:
    """Parse one cached entry: ``(outcome, rows)``, counting and flagging as it goes.

    ``outcome`` is ``payload`` (rows returned), ``empty``, ``unreadable`` or
    ``unrecognized``; the last two are quarantine-flagged.
    """
    try:
        payload = entry.json()
    except (OSError, EOFError, ValueError) as exc:
        count_outcome(stats, "unreadable")
        _flag_bad_payload(entry.source_id, "unreadable fetch-store body",
                          f"{type(exc).__name__}: {exc}")
        return "unreadable", None
    shape, rows = payload_shape(payload)
    if shape == "unrecognized":
        count_outcome(stats, "unrecognized")
        _flag_bad_payload(entry.source_id, "unrecognized ORATS envelope",
                          f"payload type {type(payload).__name__}")
        return "unrecognized", None
    if not rows:
        count_outcome(stats, "empty")
        return "empty", None
    count_outcome(stats, "payload")
    return "payload", SourceRows(entry.source_id, rows, entry.params)
