"""Foundation primitives for rearchitecture phase 1 (slice P1-1).

The canonical hash moved from ``engine/v2/diagnosis`` to ``engine/v2/foundation``
(§3.2). A move that changed one byte would silently re-key every corpus hash,
receipt and baseline manifest, so the golden values below were computed by the
phase-0 diagnosis copy BEFORE the move and are pinned here, not recomputed.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2 import foundation  # noqa: E402
from engine.v2.contracts import (  # noqa: E402
    AttemptReceipt,
    JobSpec,
    ProcessIdentity,
    ResolvedResources,
)
from engine.v2.foundation import (  # noqa: E402
    DocumentError,
    canonical_json,
    content_hash,
    format_timestamp,
    from_document,
    parse_timestamp,
    tag_nonfinite,
    to_document,
    untag_nonfinite,
)

# --------------------------------------------------------------------------
# canonical hash continuity
# --------------------------------------------------------------------------

#: Computed 2026-09-12 by engine.v2.diagnosis.canonical at e9a1f43, pre-move.
GOLDEN = {
    "floats": (
        [1.0, -0.0, 1e-5, 1e-7, 1e21, 0.026114337940089646, 123456789012345680000.0, 5e-324],
        "[1,0,0.00001,1e-7,1e+21,0.026114337940089646,123456789012345680000,5e-324]",
        "sha256:125126538568dbabfc07a8b3ea78e4b6bb80b67e3077ec799d7dc29d157c4e90",
    ),
    "keys": (
        {"\U0001F600": 1, "": 2, "a": 3, "B": 4},
        '{"B":4,"a":3,"\U0001F600":1,"":2}',
        "sha256:dfecf36a93b2b287ba9dd472d390e9d82361e034e15ce9303eb89d6ecc37e03c",
    ),
    "nested": (
        {"b": [3, 1, 2], "a": {"z": None, "y": True, "x": False}},
        '{"a":{"x":false,"y":true,"z":null},"b":[3,1,2]}',
        "sha256:df87c5d096ffe3f49b64b01135b01a979ac2ba96cc9ffcbd3ba207cdf157a07d",
    ),
    "nonfinite": (
        {"n": float("nan"), "i": float("inf")},
        '{"i":{"__nonfinite__":"inf"},"n":{"__nonfinite__":"nan"}}',
        "sha256:0b25633d3e34e4af6ec7f891e3e29f80d58777184c82e26980bc56f79b744d1d",
    ),
    "strings": (
        ["é", "line\nbreak", 'quote"', " "],
        '["é","line\\nbreak","quote\\""," "]',
        "sha256:f9fd72d9e189419e8ba9683dc211880dc84841a083f80014c71a6265bdab3bdf",
    ),
    "ints": (
        [0, -1, 2**63, 10**30],
        "[0,-1,9223372036854775808,1000000000000000000000000000000]",
        "sha256:fc3ee1826cd2b75f018d45e63b3627801984c4ed6a97fe5f87adb727a8beec6c",
    ),
    "tuple": (
        (1, "two", 3.5),
        '[1,"two",3.5]',
        "sha256:0e4641b18f79b78ebc38cec3afc14cfb23bc9a4e1fc9465b45e0d99ec90ea980",
    ),
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_foundation_reproduces_pre_move_hashes(name):
    value, canonical, digest = GOLDEN[name]
    assert canonical_json(value) == canonical
    assert content_hash(value) == digest


def test_diagnosis_reexports_the_same_objects_not_a_copy():
    from engine.v2 import diagnosis
    from engine.v2.diagnosis import canonical as diag_canonical

    assert diagnosis.canonical_json is foundation.canonical_json
    assert diagnosis.content_hash is foundation.content_hash
    assert diag_canonical.content_hash is foundation.content_hash


def test_exactly_one_canonicalizer_exists_in_v2():
    """§3.2: 'Do not copy a second canonicalizer into ops.'"""
    definers = []
    for path in sorted((ROOT / "engine" / "v2").rglob("*.py")):
        tree = ast.parse(path.read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        if {"canonical_json", "_number"} & names:
            definers.append(path.relative_to(ROOT).as_posix())
    assert definers == ["engine/v2/foundation/canonical.py"]


def test_the_seeded_hash_regression_is_detectable():
    """Negative control: a repr-style float would move the pinned hash."""
    value, _, digest = GOLDEN["floats"]
    assert content_hash([repr(v) for v in value]) != digest


# The GOLDEN floats above hold no negative number, no value with n == 0
# (0.1 <= |x| < 1) and no multi-digit mantissa in exponent form. The expected
# strings are ECMAScript Number::toString (RFC 8785 §3.2.2.2), e.g. in node
# String(-1.5), String(0.5), String(1.5e-7), String(-2.25e+22).
@pytest.mark.parametrize("value, expected", [
    (-1.5, "-1.5"),
    (-3.0, "-3"),
    (0.5, "0.5"),
    (-0.015, "-0.015"),
    (1.5e-7, "1.5e-7"),
    (1.25e-7, "1.25e-7"),
    (-2.25e22, "-2.25e+22"),
])
def test_canonical_number_layout_matches_ecmascript(value, expected):
    assert canonical_json(value) == expected
    assert canonical_json({"x": value}) == '{"x":' + expected + "}"


def test_non_json_leaves_serialize_by_their_string_form():
    """Dates, Decimals and similar leaves hash over ``str(value)``: two
    different dates must never share one content hash."""
    from datetime import date
    from decimal import Decimal

    assert canonical_json({"d": date(2026, 9, 19)}) == '{"d":"2026-09-19"}'
    assert canonical_json([Decimal("1.10")]) == '["1.10"]'
    assert content_hash(date(2026, 9, 19)) != content_hash(date(2026, 9, 20))


def test_keys_with_lone_surrogates_sort_by_utf16_code_units():
    """JCS orders keys by UTF-16 code units. A lone-surrogate key sorts as its
    own unit (D800 < D83D DE00 < E000) instead of raising."""
    doc = {"": 1, "\U0001F600": 2, "\ud800": 3}
    assert canonical_json(doc) == '{"\ud800":3,"\U0001F600":2,"":1}'


# --------------------------------------------------------------------------
# clocks
# --------------------------------------------------------------------------


def test_timestamp_round_trips_with_microseconds_and_z():
    instant = datetime(2026, 9, 12, 21, 5, 3, 120, tzinfo=timezone(timedelta(hours=-4)))
    text = format_timestamp(instant)
    assert text == "2026-09-13T01:05:03.000120Z"
    assert parse_timestamp(text) == instant


def test_naive_datetime_is_refused_rather_than_assumed_utc():
    with pytest.raises(ValueError):
        format_timestamp(datetime(2026, 9, 12, 21, 0))


@pytest.mark.parametrize("text", ["2026-09-12T21:00:00+00:00", "2026-09-12", "", None])
def test_parse_refuses_anything_but_the_wire_form(text):
    with pytest.raises(ValueError):
        parse_timestamp(text)


# --------------------------------------------------------------------------
# strict typed documents
# --------------------------------------------------------------------------

Colour = Literal["red", "green"]


@dataclass(frozen=True, kw_only=True)
class Inner:
    weight: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Outer:
    name: str
    count: int
    colour: Colour
    inner: Inner
    extra: dict[str, Any] = field(default_factory=dict)
    maybe: int | None = None
    schema_version: str = "outer.v1.2"


def _outer_doc(**changes):
    doc = {"name": "n", "count": 2, "colour": "red",
           "inner": {"weight": 1, "tags": ["a", "b"]}, "schema_version": "outer.v1.2"}
    doc.update(changes)
    return doc


def test_round_trip_preserves_nested_values_and_order():
    value = from_document(Outer, _outer_doc(extra={"k": [1, 2.5]}))
    assert value.inner == Inner(weight=1.0, tags=("a", "b"))
    assert from_document(Outer, to_document(value)) == value


@pytest.mark.parametrize("changes,code,path", [
    ({"surprise": 1}, "UNKNOWN_FIELD", "$.surprise"),
    ({"colour": "blue"}, "BAD_ENUM", "$.colour"),
    ({"count": True}, "BAD_TYPE", "$.count"),
    ({"count": 2.0}, "BAD_TYPE", "$.count"),
    ({"inner": {"weight": float("nan")}}, "BAD_TYPE", "$.inner.weight"),
    ({"inner": {"weight": 1, "tags": "ab"}}, "BAD_TYPE", "$.inner.tags"),
    ({"inner": {"weight": 1, "oops": 1}}, "UNKNOWN_FIELD", "$.inner.oops"),
    ({"extra": {"k": float("inf")}}, "BAD_TYPE", "$.extra.k"),
    ({"maybe": "3"}, "BAD_TYPE", "$.maybe"),
    ({"schema_version": "outer.v2.0"}, "UNSUPPORTED_VERSION", "$.schema_version"),
    ({"schema_version": "outer.v1.3"}, "UNSUPPORTED_VERSION", "$.schema_version"),
    ({"schema_version": "other.v1.2"}, "UNSUPPORTED_VERSION", "$.schema_version"),
    ({"schema_version": "garbage"}, "BAD_SCHEMA_VERSION", "$.schema_version"),
])
def test_contract_violations_are_refused_with_code_and_path(changes, code, path):
    with pytest.raises(DocumentError) as err:
        from_document(Outer, _outer_doc(**changes))
    assert (err.value.code, err.value.path) == (code, path)


def test_missing_required_and_missing_version_are_refused():
    doc = _outer_doc()
    del doc["count"]
    with pytest.raises(DocumentError) as err:
        from_document(Outer, doc)
    assert err.value.code == "MISSING_FIELD"
    doc = _outer_doc()
    del doc["schema_version"]
    with pytest.raises(DocumentError) as err:
        from_document(Outer, doc)
    assert err.value.path == "$.schema_version"


def test_an_older_minor_is_readable():
    assert from_document(Outer, _outer_doc(schema_version="outer.v1.0")).name == "n"


def test_refusal_never_echoes_the_offending_value():
    """§5.2: a rejected field may carry a credential; the error names the path only."""
    secret = "tok_live_SECRET_VALUE"
    with pytest.raises(DocumentError) as err:
        from_document(Outer, _outer_doc(colour=secret))
    assert secret not in str(err.value)


def test_real_contracts_round_trip_through_the_decoder():
    resources = ResolvedResources(
        effective_host_budget_bytes=6 << 30, reserved_memory_bytes=3 << 30,
        assigned_cpu_ids=(2, 5, 7), thread_count=3, scratch_limit_bytes=1 << 30,
        executor_mode="watchdog", containment="best_effort", provider_leases=(),
        resource_profile_version="policy.v1")
    attempt = AttemptReceipt(
        job_id="job_1", attempt_id="att_1", attempt_number=1, fence=4,
        supervisor_epoch="sup_1", host_boot_id="boot", state="running",
        process_state="alive", resolved_resources=resources,
        process_identity=ProcessIdentity(boot_id="boot", pid=10, start_ticks=99,
                                         process_group=10))
    assert from_document(AttemptReceipt, to_document(attempt)) == attempt
    spec = JobSpec(kind="synthetic.echo", implementation_ref="impl", spec_hash=None,
                   environment_ref="env", output_namespace="ns", resource_class="io_fetch",
                   retry_policy_ref="retry.none", checkpoint_contract_ref="ckpt.none",
                   parameters={"n": 1})
    assert from_document(JobSpec, to_document(spec)) == spec


# --------------------------------------------------------------------------
# tag_nonfinite / untag_nonfinite: legacy NaN round trip through a real
# artifact write (real shadow nightly attempt 14, model_evidence.json's
# magnitude_spearman)
# --------------------------------------------------------------------------


def test_tag_nonfinite_is_valid_strict_json_and_matches_content_hash():
    import json
    import math

    value = {"models": {"dyn_sv_chooser_v1_1": {"inputs": [
        {"name": "magnitude_spearman", "value": float("nan")},
        {"name": "other", "value": float("inf")},
    ]}}}
    tagged = tag_nonfinite(value)
    # allow_nan=False must not raise: every NaN/Infinity is gone.
    text = json.dumps(tagged, sort_keys=True, allow_nan=False)
    reloaded = json.loads(text)
    # Written bytes and the identity taken over the raw value agree --
    # `_write_action` writes `tag_nonfinite(value)` but hashes `value`.
    assert canonical_json(reloaded) == canonical_json(value)
    assert content_hash(value) == content_hash(reloaded)
    restored = untag_nonfinite(reloaded)
    inputs = restored["models"]["dyn_sv_chooser_v1_1"]["inputs"]
    assert math.isnan(inputs[0]["value"])
    assert math.isinf(inputs[1]["value"]) and inputs[1]["value"] > 0


def test_untag_nonfinite_round_trips_legacy_reader_semantics():
    """The legacy consumer this exists for: ``abs(s.get("magnitude_spearman")
    or 0.0)`` (``engine/dashboard/model_evidence.py:387``) sees a real NaN,
    not the tag, and NaN stays truthy under ``or`` the way legacy's own
    ``json.load`` of a raw ``NaN`` literal already behaves -- ``or 0.0``
    must NOT replace it with ``0.0``.
    """
    import math

    tagged = {"magnitude_spearman": {"__nonfinite__": "nan"}}
    restored = untag_nonfinite(tagged)
    value = restored.get("magnitude_spearman") or 0.0
    assert isinstance(value, float) and math.isnan(value)


def test_tag_nonfinite_leaves_ordinary_values_unchanged():
    value = {"a": [1, 2.5, "x", None, True], "b": {"c": 3}}
    assert tag_nonfinite(value) == value
    assert untag_nonfinite(tag_nonfinite(value)) == value


def test_fast_number_layout_matches_the_decimal_path():
    """``_number`` reads m and n from ``repr``; it must render every double
    exactly as the Decimal-based layout it replaced (random bit patterns,
    ordinary values and the JCS boundary cases)."""
    import random
    import struct
    from decimal import Decimal

    from engine.v2.foundation.canonical import _number

    def reference(value):
        if value == 0.0:
            return "0"
        sign = "-" if value < 0 else ""
        tup = Decimal(repr(abs(value))).as_tuple()
        digits = "".join(str(d) for d in tup.digits)
        e = int(tup.exponent)
        while len(digits) > 1 and digits.endswith("0"):
            digits, e = digits[:-1], e + 1
        k, n = len(digits), len(digits) + e
        if k <= n <= 21:
            return sign + digits + "0" * (n - k)
        if 0 < n <= 21:
            return sign + digits[:n] + "." + digits[n:]
        if -6 < n <= 0:
            return sign + "0." + "0" * (-n) + digits
        mantissa = digits[0] + ("." + digits[1:] if k > 1 else "")
        return f"{sign}{mantissa}e{'+' if n - 1 >= 0 else '-'}{abs(n - 1)}"

    rng = random.Random(7)
    values = [1e21, 1e20, 9.999999999999999e20, 1e-6, 1e-7, 1.5e-6, 5e-324,
              2.2250738585072014e-308, 1.7976931348623157e308, 0.1, 100.0, 1.0,
              1e22, 0.000001234, 12.5, 1e16, 1.2345e16, 123456789012345680000.0]
    values += [-v for v in values]
    for _ in range(20000):
        v = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if v == v and abs(v) != float("inf"):
            values.append(v)
    values += [rng.gauss(0.0, 1.0) * 10 ** rng.randint(-8, 8) for _ in range(20000)]
    values += [float(rng.randint(-10**7, 10**7)) for _ in range(5000)]
    assert [_number(v) for v in values] == [reference(v) for v in values]
