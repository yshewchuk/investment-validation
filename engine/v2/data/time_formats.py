"""The one naive-timestamp wire format, shared by its producer and its checker.

Legacy pandas ``datetime64[ns]``/``datetime64[us]`` columns carry no UTC
offset (``objects.py``'s ``legacy_nan_is_null.v1`` neighbor judgement call:
a legacy value is trusted for its instant, never its zone). Seven of the
eight legacy-mapped tables declare exactly this kind of column as their
``observation_time_column``, so ``objects._format_naive_timestamp`` writes
it, and ``documents.py`` must accept it back as a genuine ``TimeInterval``/
``FragmentRecord`` time bound — but only in this one exact shape, never
silently as an alias for ``foundation.clock``'s offset-required RFC 3339
wire form.

One module rather than one module defining it and the other guessing at its
shape: ``objects.py`` calls :func:`format_naive_timestamp` to produce it,
``documents.py`` calls :func:`is_naive_timestamp` to recognize it, and
neither carries its own copy of the pattern.

Layer 1: stdlib only (``re``, ``datetime``), so this module can sit below
both ``objects`` and ``documents`` without creating a dependency between them.
"""
from __future__ import annotations

import re
from datetime import datetime

__all__ = ["NAIVE_TIMESTAMP_FORMAT", "NAIVE_TIMESTAMP_RE", "format_naive_timestamp", "is_naive_timestamp"]

#: The strftime pattern the wire form is written with: no offset, and always
#: exactly six microsecond digits (``%f`` zero-pads on write; the matching
#: regex below requires all six back on read — never a truncated fraction).
NAIVE_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

#: A cheap pre-check before ``strptime`` — also what tells this format apart
#: from a bare date (no ``T``) and an RFC 3339 aware timestamp (always ends
#: ``Z``, which this pattern's anchored end refuses).
NAIVE_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}$")


def format_naive_timestamp(value: datetime) -> str:
    """``value`` (any tzinfo — or none) written in the shared naive wire form."""
    return value.strftime(NAIVE_TIMESTAMP_FORMAT)


def is_naive_timestamp(value: str) -> bool:
    """True iff ``value`` is exactly the wire form above, and a real calendar value.

    The regex alone would accept ``"2024-13-40T00:00:00.000000"``; the
    ``strptime`` re-parse is what refuses an impossible month or day, the
    same division of labor ``documents._is_real_date`` uses for dates.
    """
    if not isinstance(value, str) or not NAIVE_TIMESTAMP_RE.match(value):
        return False
    try:
        datetime.strptime(value, NAIVE_TIMESTAMP_FORMAT)
    except ValueError:
        return False
    return True
