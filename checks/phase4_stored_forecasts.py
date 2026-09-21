"""Resolve a captured stored-forecast REFERENCE into a value, natively.

Legacy ``Scorer._crush_forecast`` answers ``pred_iv_crush_30`` two ways. For
a forward event it serves a Tier-4 fold. For an event that has already
printed it reads the cell straight out of the stored Tier-4 forecasts table,
serving no model at all. Every HISTORICAL row of the seven planned-exit
DYN-SV strategies (TWIN-P, TWIN-P5, CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5)
takes the second path, because all seven require the ``iv_crush`` role.

Phase 4 compares native scoring against legacy scoring, so the value legacy
read is an ANSWER: supplying it back as a native input would make the
comparison circular, and the check would pass with native's retrieval path
completely broken. ``tools/phase4_release_assembler.py::_reject_answers``
refuses it for exactly that reason.

So the capture emits an address instead, at
``native_inputs.forecast.stored_refs[output]``::

    {"row": {"table", "table_sha256", "column", "ticker", "event_date",
             "model_id", "fold_start"},
     "row_hash": "<content hash of {row, value}>"}

and this module performs the lookup: it opens the named table, refuses
unless the file's sha256 is the vintage the capture named, reads the one
cell at (ticker, event_date, column), and only then checks what it read
against the declared producer model and against ``row_hash``. The hash is a
verification aid and cannot replace the lookup -- a content hash does not
invert, so the value has to exist before anything can be checked against it.

Failure is a refusal, never a fallback. Every refusal below raises
:class:`StoredForecastError` carrying a ``reason`` code; nothing in this
module can produce a value from the trace alone, so there is no path by
which a broken lookup quietly becomes a passing comparison. A caller that
skips this module entirely does not get a silent fold forecast either:
``engine/v2/scoring/stages.py::_execute_local_forecast`` flags an
unresolved ``stored_refs`` entry ``UNRESOLVED_STORED_FORECAST:<output>``
and marks the output invalid.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping

from engine.v2.scoring.source_inputs import (
    STORED_REF_FIELDS,
    STORED_ROW_FIELDS,
    stored_forecast_row_hash,
)
from engine.v2.scoring.stages import NativeScoreInputs

__all__ = [
    "STORED_TABLES",
    "StoredForecastError",
    "resolve_stored_forecasts",
    "with_stored_forecasts",
]


class StoredForecastError(RuntimeError):
    """A declared stored-forecast reference could not be resolved.

    Always a refusal. There is deliberately no degraded outcome: the only
    other thing this module could do is use a value the capture supplied,
    which is the circularity the reference exists to remove.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def _tier4_path() -> Path:
    from engine import paths

    return Path(paths.TIER4)


#: Table name -> the file that holds it. A reference naming anything else is
#: refused rather than guessed at.
STORED_TABLES: Mapping[str, Callable[[], Path]] = {"tier4_forecasts": _tier4_path}

_DIGESTS: dict[tuple[str, int, int], str] = {}


def _file_digest(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _DIGESTS.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def _text(value: Any) -> str | None:
    """A provenance cell as canonical text, or ``None`` when it is null.

    Mirrors ``engine/score.py::_stored_cell_text``: the ``*_model_id`` column
    is pandas' nullable ``string`` dtype, so an unnamed producer arrives as
    ``pd.NA``/``None``/``NaN`` depending on the reader, and all three mean the
    same thing.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value)
    return None if text in ("<NA>", "NaT", "nan", "None") else text


def _date_text(value: Any) -> str | None:
    """A ``*_fold_start`` cell as an ISO date, or ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    date = getattr(value, "date", None)
    if callable(date):
        return str(date())
    text = str(value)
    if text in ("<NA>", "NaT", "nan", "None"):
        return None
    return text[:10]


def _read_cell(path: Path, ticker: str, event_date: str, column: str) -> dict[str, Any]:
    """The one stored row at ``(ticker, event_date)``, as plain Python.

    Reads five columns of one table through pyarrow rather than
    ``engine.data.features.tier4.load_forecasts``: this is native's own
    retrieval path, and it must not borrow legacy's. Only the queried
    ticker's rows are materialized, so the read costs kilobytes.
    """
    import pyarrow.parquet as pq

    model_column = f"{column}_model_id"
    fold_column = f"{column}_fold_start"
    try:
        table = pq.read_table(
            path,
            columns=["ticker", "event_date", column, model_column, fold_column],
            filters=[("ticker", "==", ticker)],
        )
    except (OSError, ValueError, KeyError) as exc:
        raise StoredForecastError(
            "STORED_FORECAST_TABLE_UNREADABLE", f"{path.name}: {exc}",
        ) from exc
    matched = [
        row for row in table.to_pylist()
        if _date_text(row.get("event_date")) == event_date
    ]
    if not matched:
        raise StoredForecastError(
            "STORED_FORECAST_ROW_MISSING",
            f"{path.name} has no row for the declared key",
        )
    if len(matched) > 1:
        raise StoredForecastError(
            "STORED_FORECAST_ROW_AMBIGUOUS",
            f"{path.name} has {len(matched)} rows for the declared key",
        )
    row = matched[0]
    return {
        "value": row.get(column),
        "model_id": _text(row.get(model_column)),
        "fold_start": _date_text(row.get(fold_column)),
    }


def _resolve_one(output: str, entry: Any, reader) -> dict[str, Any]:
    location = f"native_inputs.forecast.stored_refs.{output}"
    if not isinstance(entry, Mapping) or set(map(str, entry)) != STORED_REF_FIELDS:
        raise StoredForecastError(
            "STORED_FORECAST_REF_MALFORMED",
            f"{location} must name {sorted(STORED_REF_FIELDS)}",
        )
    row = entry["row"]
    if not isinstance(row, Mapping) or set(map(str, row)) != STORED_ROW_FIELDS:
        raise StoredForecastError(
            "STORED_FORECAST_REF_MALFORMED",
            f"{location}.row must name {sorted(STORED_ROW_FIELDS)}",
        )
    row = {str(key): row[key] for key in sorted(row)}
    if row["column"] != output:
        raise StoredForecastError(
            "STORED_FORECAST_REF_MALFORMED",
            f"{location}.row.column does not name {output}",
        )
    locate = STORED_TABLES.get(str(row["table"]))
    if locate is None:
        raise StoredForecastError(
            "STORED_FORECAST_TABLE_UNKNOWN", f"{location}.row.table",
        )
    path = locate()
    if not path.is_file():
        raise StoredForecastError(
            "STORED_FORECAST_TABLE_MISSING", f"{row['table']} is not built here",
        )
    digest = _file_digest(path)
    if digest != str(row["table_sha256"]):
        # A different vintage of the same table would answer, and answer
        # differently. Comparing against a value legacy never read is not a
        # weaker comparison, it is a wrong one, so this refuses and waits
        # for a capture taken against the table on disk.
        raise StoredForecastError(
            "STORED_FORECAST_TABLE_MISMATCH",
            f"{row['table']} on disk is not the vintage the capture read",
        )
    read = reader(path, str(row["ticker"]), str(row["event_date"]), output)
    raw = read["value"]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float("nan")
    if not isfinite(value):
        # Legacy takes the stored branch only for a non-NaN cell; a NaN one
        # means the capture and this table disagree about what is stored.
        raise StoredForecastError(
            "STORED_FORECAST_VALUE_MISSING",
            f"{row['table']}.{output} holds no finite value for the declared key",
        )
    for key in ("model_id", "fold_start"):
        if read[key] != row[key]:
            raise StoredForecastError(
                "STORED_FORECAST_PROVENANCE_MISMATCH",
                f"{location}.row.{key} is not what the resolved row carries",
            )
    row_hash = stored_forecast_row_hash(row, value)
    if row_hash != str(entry["row_hash"]):
        raise StoredForecastError(
            "STORED_FORECAST_ROW_HASH_MISMATCH",
            f"{location}: the resolved cell is not the one the capture hashed",
        )
    return {"value": value, "row": row, "row_hash": row_hash}


def resolve_stored_forecasts(
    inputs: NativeScoreInputs, *, reader=_read_cell,
) -> dict[str, dict[str, Any]] | None:
    """Every stored-forecast reference in ``inputs``, resolved to a value.

    ``None`` when the trace declares none. Raises
    :class:`StoredForecastError` on the first reference that cannot be
    resolved; it never returns a partial result, because a half-resolved
    forecast block would score.

    ``reader`` is injectable for tests and for a consumer serving the same
    bytes from a staged root; the default reads the real table.
    """
    refs = inputs.forecast.get("stored_refs")
    if not refs:
        return None
    if not isinstance(refs, Mapping):
        raise StoredForecastError(
            "STORED_FORECAST_REF_MALFORMED",
            "native_inputs.forecast.stored_refs must be an object",
        )
    if inputs.forecast.get("stored"):
        raise StoredForecastError(
            "STORED_FORECAST_ALREADY_RESOLVED",
            "native_inputs.forecast already carries a resolved stored block",
        )
    return {
        str(output): _resolve_one(str(output), entry, reader)
        for output, entry in sorted(refs.items())
    }


def with_stored_forecasts(
    inputs: NativeScoreInputs, resolved: Mapping[str, Mapping[str, Any]] | None,
) -> NativeScoreInputs:
    """``inputs`` with the resolved stored cells in ``forecast["stored"]``.

    The references stay, so the scored inputs still record which address the
    value came from. ``_execute_local_forecast`` prefers ``stored`` when it
    is present, exactly as legacy prefers the stored cell over a fold.
    """
    if not resolved:
        return inputs
    forecast = dict(inputs.forecast)
    forecast["stored"] = {key: dict(value) for key, value in resolved.items()}
    return replace(inputs, forecast=forecast)
