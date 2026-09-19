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

Phase 0 wrote this into ``engine/v2/diagnosis`` because that was the only v2
package allowed code. Rearchitecture phase 1 §3.2 moved it here, unchanged, so
production ops can hash without importing diagnosis (a sink nothing may
import). ``engine.v2.diagnosis.canonical`` re-exports these same objects, and
``tests/test_v2_ops_foundation.py`` pins hashes computed by the phase-0 copy
before the move, so a byte of drift fails rather than silently re-keying every
receipt and corpus hash.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from typing import Any, Iterator

__all__ = ["canonical_json", "content_hash", "CONTENT_HASH_PREFIX",
           "NONFINITE_KEY", "tag_nonfinite", "untag_nonfinite",
           "iter_canonical_json", "stream_content_hash"]

#: contracts §2.1: ContentHash is ``sha256:`` plus the full 64-hex digest.
CONTENT_HASH_PREFIX = "sha256:"

#: The tag a non-finite float (NaN, +/-Infinity) normalizes to below, and the
#: same key ``tools/capture_tier0_corpus.py``/``checks/tier0_corpus.py`` use
#: for the tier-0 corpus's own frozen NaN markers -- one convention, not two.
NONFINITE_KEY = "__nonfinite__"


def _scalar(value: Any) -> Any:
    """Normalize one leaf to a JSON-representable value, losing no precision."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            # contracts §2.1: a missing value is never NaN, Infinity or zero.
            # Naming it here rather than writing `NaN` keeps the hash over
            # strict JSON, and keeps "absent" distinguishable from "0.0".
            return {NONFINITE_KEY: repr(value)}
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
    if type(value) is float:
        # The digits and the decimal point straight from repr's text
        # ("d.ddde±x", "ddd.ddd" or "0.000ddd"): the same m and n the Decimal
        # path below derives, without building a Decimal per float (a
        # capture hashes ~10^5-10^6 floats per document).
        mantissa, _, exponent = repr(abs(value)).partition("e")
        whole, _, fraction = mantissa.partition(".")
        raw = whole + fraction
        digits = raw.lstrip("0")
        n = len(whole) + (int(exponent) if exponent else 0) - (len(raw) - len(digits))
        digits = digits.rstrip("0")
        k = len(digits)
    else:  # a float subclass: the original, general path
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
        # swap the bytes of every unit and sort "" (00 E0) below "1"
        # (31 00) — the reverse of the RFC's ordering. ``surrogatepass``
        # because Python strings may hold lone surrogates that JS would have
        # merged; encoding their code units is the JCS-correct treatment.
        items = sorted(value.items(),
                       key=lambda kv: kv[0].encode("utf-16-be", "surrogatepass"))
        body = ",".join(f"{json.dumps(k, ensure_ascii=False)}:{_serialize(v)}"
                        for k, v in items)
        return "{" + body + "}"
    raise TypeError(f"not canonicalizable: {type(value).__name__}")


def canonical_json(value: Any, *, fragments: Any = None) -> str:
    """Serialize ``value`` to RFC 8785 canonical JSON.

    ``fragments``: optional memo for sub-values shared BY IDENTITY across
    many hashed documents (one served model pool embedded in every capture
    candidate it served). It is asked ``fragments.canonical(node, render)``
    for each container node and returns that node's canonical text (calling
    ``render(node)`` once per shared node) or ``None`` for a node it does
    not hold. The text is byte-identical to the plain path: JCS serializes
    each member and element independently of where it sits.
    """
    if fragments is None:
        return _serialize(_normalize(value))
    return _serialize_shared(value, fragments)


def _plain(value: Any) -> str:
    return _serialize(_normalize(value))


_CONTAINERS = (dict, list, tuple)


def _serialize_shared(value: Any, fragments: Any) -> str:
    if not isinstance(value, _CONTAINERS):
        return _serialize(_scalar(value))  # a leaf: exactly _plain(value)
    text = fragments.canonical(value, _plain)
    if text is not None:
        return text
    if isinstance(value, dict):
        members = {str(k): v for k, v in value.items()}
        items = sorted(members.items(),
                       key=lambda kv: kv[0].encode("utf-16-be", "surrogatepass"))
        body = ",".join(
            f"{json.dumps(k, ensure_ascii=False)}:"
            + (_serialize_shared(v, fragments) if isinstance(v, _CONTAINERS)
               else _serialize(_scalar(v)))
            for k, v in items)
        return "{" + body + "}"
    return "[" + ",".join(
        _serialize_shared(v, fragments) if isinstance(v, _CONTAINERS)
        else _serialize(_scalar(v)) for v in value) + "]"


def content_hash(value: Any, *, fragments: Any = None) -> str:
    """``sha256:<64 hex>`` over the canonical JSON of ``value``. ``fragments``
    (see :func:`canonical_json`) only saves work; the hash is the same."""
    text = canonical_json(value, fragments=fragments)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{CONTENT_HASH_PREFIX}{digest}"


def iter_canonical_json(value: Any, *, fragments: Any = None) -> Iterator[str]:
    """Stream :func:`canonical_json`'s text as chunks, never joined.

    Byte-for-byte identical to ``canonical_json(value, fragments=fragments)``
    when the yielded pieces are concatenated -- proven in
    ``tests/test_v2_ops_foundation.py`` by comparing this against that
    function (the oracle) over the same fixtures ``canonical_json`` is
    already pinned against, plus large/shared/nonfinite ones the batch path
    would rather not build as one string.

    This does the SAME per-node work ``_serialize_shared`` already does
    (``_scalar`` at every leaf, ``fragments.canonical`` at every container,
    UTF-16-code-unit key order) but recurses directly over ``value`` instead
    of building the result with ``str.join``, so no single Python object ever
    holds more than one node's rendered text. A cache hit under ``fragments``
    still yields that fragment's whole cached text as one chunk -- bounded by
    ``_SharedTraceDocuments``'s own text budget, not by document size.
    """
    if not isinstance(value, _CONTAINERS):
        yield _serialize(_scalar(value))
        return
    cached = fragments.canonical(value, _plain) if fragments is not None else None
    if cached is not None:
        yield cached
        return
    if isinstance(value, dict):
        members = {str(k): v for k, v in value.items()}
        items = sorted(members.items(),
                       key=lambda kv: kv[0].encode("utf-16-be", "surrogatepass"))
        yield "{"
        first = True
        for k, v in items:
            yield ("" if first else ",") + json.dumps(k, ensure_ascii=False) + ":"
            first = False
            if isinstance(v, _CONTAINERS):
                yield from iter_canonical_json(v, fragments=fragments)
            else:
                yield _serialize(_scalar(v))
        yield "}"
        return
    # a list or tuple: order preserved, never sorted (contracts §2.2)
    yield "["
    first = True
    for v in value:
        if not first:
            yield ","
        first = False
        if isinstance(v, _CONTAINERS):
            yield from iter_canonical_json(v, fragments=fragments)
        else:
            yield _serialize(_scalar(v))
    yield "]"


def stream_content_hash(value: Any, *, fragments: Any = None) -> str:
    """``content_hash``, but the sha256 is fed chunk by chunk from
    :func:`iter_canonical_json` -- no whole-document string is ever built.
    Byte-identical digest to ``content_hash(value, fragments=fragments)``
    because it hashes the exact same UTF-8 bytes, just incrementally: every
    yielded chunk is a complete Python ``str`` (never a partial character),
    so encoding each chunk and concatenating the results is the same as
    encoding the one joined string would have been.
    """
    digest = hashlib.sha256()
    for chunk in iter_canonical_json(value, fragments=fragments):
        digest.update(chunk.encode("utf-8"))
    return f"{CONTENT_HASH_PREFIX}{digest.hexdigest()}"


def tag_nonfinite(value: Any) -> Any:
    """``value`` normalized into strict-JSON-safe form (contracts §2.1): a
    NaN or +/-Infinity float becomes ``{"__nonfinite__": repr(value)}``
    instead of raising, everything else round-trips unchanged. This is the
    same normalization :func:`canonical_json`/:func:`content_hash` already
    apply internally, exposed so a writer that must put the SAME value on
    disk as a real (``allow_nan=False``) JSON document can do so and still
    have its file agree byte-for-semantics with the hash taken over the raw
    value -- see :func:`untag_nonfinite` for the read-side inverse.
    """
    return _normalize(value)


def untag_nonfinite(value: Any) -> Any:
    """Inverse of :func:`tag_nonfinite`: decode a ``{"__nonfinite__": ...}``
    tag back into a real ``float('nan')``/``inf``/``-inf``, recursively over
    dicts and lists. The read-boundary companion to ``tag_nonfinite`` --
    apply this right after ``json.loads`` on anything written that way, so a
    legacy NaN round-trips as a real float for every downstream reader
    (arithmetic, ``or 0.0`` truthiness, comparisons) instead of landing as an
    opaque one-key dict.
    """
    if isinstance(value, dict):
        if set(value) == {NONFINITE_KEY} and isinstance(value[NONFINITE_KEY], str):
            try:
                return float(value[NONFINITE_KEY])
            except ValueError:
                pass
        return {k: untag_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [untag_nonfinite(v) for v in value]
    return value
