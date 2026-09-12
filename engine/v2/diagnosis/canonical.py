"""Canonical JSON and content hashing, per component contracts §2.2.

JCS ordering and serialization rules: object keys sorted by their UTF-16 code
units, no insignificant whitespace, and **no display rounding anywhere**. Two
defects this program has already paid for — ``json_safe`` rounding a replay
input to six places (`b33036c`) and ``_write_pair`` re-rounding it after the
exemption (`6b9d5cf`) — are both "the identity was taken over a rounded value".
So there is no ``round_to`` parameter here and there must never be one: a
display path that wants six places rounds on its own way to a screen, after the
hash is taken.

This belongs in ``engine/v2/foundation`` once that package is written
(§4.4 maps ``jsonio.py`` there). It lives here for now because phase 0 writes no
production logic into any v2 package except this one, and a comparator that
cannot hash cannot produce a receipt. Moving it is a mechanical import change
that the tier-0 corpus will prove inert.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

__all__ = ["canonical_json", "content_hash", "CONTENT_HASH_PREFIX"]

#: contracts §2.1: ContentHash is ``sha256:`` plus the full 64-hex digest.
CONTENT_HASH_PREFIX = "sha256:"


def _scalar(value: Any) -> Any:
    """Normalize one leaf to a JSON-representable value, losing no precision."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            # contracts §2.1: a missing value is never NaN, Infinity or zero.
            # Naming it here rather than writing `NaN` keeps the hash over
            # strict JSON, and keeps "absent" distinguishable from "0.0".
            return {"__nonfinite__": repr(value)}
        return value
    # Dates, Decimals, numpy scalars and anything else with a faithful string
    # form. `repr` is deliberately not used: it is a Python display convention,
    # and contracts §2.2 forbids identity over one.
    return str(value)


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        # Array order is meaningful — feature order, menu order, leg order —
        # so it is preserved rather than sorted (contracts §2.2).
        return [_normalize(v) for v in value]
    return _scalar(value)


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to canonical JSON. Keys sorted, no whitespace."""
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """``sha256:<64 hex>`` over the canonical JSON of ``value``."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{CONTENT_HASH_PREFIX}{digest}"
