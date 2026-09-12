"""Canonical JSON and content hashing, per RFC 8785 (JCS) and contracts §2.2.

This is a real JCS implementation, not Python's ``json.dumps(sort_keys=True)``
wearing its name. The two differ wherever an identity matters across languages:

- **Numbers** serialize per ECMAScript ``Number::toString`` (RFC 8785
  §3.2.2.2): ``1.0`` is ``1``, ``-0.0`` is ``0``, ``1e-5`` is ``0.00001``,
  ``1e-7`` is ``1e-7``, ``1e21`` is ``1e+21``. Python's repr emits ``1.0``,
  ``-0.0``, ``1e-05`` — a different hash over the same value.
- **Object keys** sort by UTF-16 code units, not Unicode code points. The two
  orders disagree for every key mixing astral characters (U+10000 and above)
  with U+E000..U+FFFF: the UTF-16 form of an astral character starts with a
  surrogate (0xD800..0xDFFF), which sorts BELOW 0xE000. Code-point sorting
  puts it above. Sorting is over the encoded units, per the RFC.
- **No display rounding anywhere.** Two defects this program has already paid
  for — ``json_safe`` rounding a replay input to six places (`b33036c`) and
  ``_write_pair`` re-rounding it after the exemption (`6b9d5cf`) — are both
  "the identity was taken over a rounded value". So there is no ``round_to``
  parameter here and there must never be one: a display path that wants six
  places rounds on its own way to a screen, after the hash is taken.

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
from decimal import Decimal
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


def _number(value: float) -> str:
    """ECMAScript ``Number::toString`` for a finite double (RFC 8785 §3.2.2.2).

    ``repr`` already gives the shortest round-tripping decimal; this only
    re-lays it out the way a JS consumer would, so both languages hash the
    same bytes. ``m`` is the significand's digits with trailing zeros
    stripped, ``k`` its length, and the value is ``0.m * 10**n``.
    """
    if value == 0.0:
        return "0"  # and -0.0: ES6 renders negative zero as "0"
    sign = "-" if value < 0 else ""
    tup = Decimal(repr(abs(value))).as_tuple()
    digits = "".join(str(d) for d in tup.digits)
    e = int(tup.exponent)
    while len(digits) > 1 and digits.endswith("0"):
        digits = digits[:-1]
        e += 1
    k = len(digits)
    n = k + e
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    exp = n - 1
    mantissa = digits[0] + ("." + digits[1:] if k > 1 else "")
    return f"{sign}{mantissa}e{'+' if exp >= 0 else '-'}{abs(exp)}"


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_serialize(v) for v in value) + "]"
    if isinstance(value, dict):
        # UTF-16-BE, so byte order IS code-unit order. Little-endian would
        # swap the bytes of every unit and sort "\ue000" (00 E0) below "1"
        # (31 00) — the reverse of the RFC's ordering. ``surrogatepass``
        # because Python strings may hold lone surrogates that JS would have
        # merged; encoding their code units is the JCS-correct treatment.
        items = sorted(value.items(),
                       key=lambda kv: kv[0].encode("utf-16-be", "surrogatepass"))
        body = ",".join(f"{json.dumps(k, ensure_ascii=False)}:{_serialize(v)}"
                        for k, v in items)
        return "{" + body + "}"
    raise TypeError(f"not canonicalizable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to RFC 8785 canonical JSON."""
    return _serialize(_normalize(value))


def content_hash(value: Any) -> str:
    """``sha256:<64 hex>`` over the canonical JSON of ``value``."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{CONTENT_HASH_PREFIX}{digest}"
