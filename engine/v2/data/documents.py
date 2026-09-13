"""Strict document decoding for data contracts, beyond what typed.py can do.

Phase-2 guide §5, §12 (D01). ``engine.v2.foundation.typed`` decodes a document
from a dataclass's own annotations: it already refuses unknown fields, unknown
enum members, non-finite numbers, and an incompatible major/newer-minor schema
version. It cannot, from an annotation of plain ``str``, tell a hash from a
timestamp from an ordinary free-text field, and it cannot see a rule that
spans two fields (a batch limit that exceeds a result limit, a primary key
naming a column the table never declared). Those checks live here instead,
against one explicit per-``(class, field)`` format table (phase-2 guide §5:
"reject ... hashes that are not sha256: plus 64 lowercase hexadecimal
characters") so a new field cannot silently skip validation by drifting from a
name-guessing convention.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts`` and ``engine.v2.foundation``, per phase-2 guide §3.3
("data owner ... imports only contracts/foundation").
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import re
from typing import Any

from engine.v2.foundation import DocumentError, from_document, parse_timestamp

__all__ = ["decode_document", "loads_document"]

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Explicit (dataclass name, field name) -> "hash" | "timestamp" | "date".
#: Every field across ``engine/v2/contracts/data.py`` ending in ``_hash`` or
#: ``_at``, or named ``deadline``/``observation_ceiling``/``known_from``, must
#: appear here (checked by ``tests/test_v2_data_contracts.py``). Entries for
#: reused handles (``LegacyFileRef.content_hash``) are included too, even
#: though that dataclass lives in ``contracts.jobs`` and is outside the test's
#: required set, so a document nesting one still gets checked.
_FORMATS: dict[tuple[str, str], str] = {
    ("TableContract", "definition_hash"): "hash",
    ("TableContractRef", "definition_hash"): "hash",
    ("ObjectRef", "content_hash"): "hash",
    ("FragmentRef", "manifest_hash"): "hash",
    ("FragmentRecord", "manifest_hash"): "hash",
    ("FragmentRecord", "byte_hash"): "hash",
    ("FragmentRecord", "logical_content_hash"): "hash",
    ("FragmentRecord", "import_request_hash"): "hash",
    ("DatasetVersionRef", "manifest_hash"): "hash",
    ("DatasetManifest", "logical_content_hash"): "hash",
    ("SnapshotRef", "manifest_hash"): "hash",
    ("DataQuery", "deadline"): "timestamp",
    ("EarningsEvent", "scheduled_event_date"): "date",
    ("EarningsEvent", "actual_announcement_at"): "timestamp",
    ("EarningsEvent", "known_from"): "timestamp",
    ("ContractId", "expiry"): "date",
    ("ChainQuery", "observation_ceiling"): "timestamp",
    ("ChainQuery", "session_date"): "date",
    ("ChainSnapshot", "observed_at"): "timestamp",
    ("ChainSnapshot", "available_at"): "timestamp",
    ("ChainSnapshot", "received_at"): "timestamp",
    ("ChainSnapshot", "session_date"): "date",
    ("DependencyPlan", "request_hash"): "hash",
    ("SnapshotImportRequest", "source_manifest_hash"): "hash",
    ("SnapshotImportReceipt", "request_hash"): "hash",
    ("LegacyMaterializationRequest", "request_hash"): "hash",
    ("LegacyFileRef", "content_hash"): "hash",
}


def loads_document(cls: type, text: str) -> Any:
    """Parse ``text`` as JSON, refusing a duplicate key, then decode it."""
    doc = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    return decode_document(cls, doc)


def decode_document(cls: type, doc: Any) -> Any:
    """Strictly decode ``doc`` as ``cls``, then apply the checks typed.py can't."""
    instance = from_document(cls, doc)
    _walk(instance, doc, "$")
    return instance


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise DocumentError("DUPLICATE_KEY", f"$.{key}",
                                f"{key!r} appears more than once in one object")
        seen[key] = value
    return seen


def _walk(value: Any, doc_value: Any, path: str) -> None:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _walk_dataclass(value, doc_value, path)
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            in_range = isinstance(doc_value, list) and index < len(doc_value)
            item_doc = doc_value[index] if in_range else None
            _walk(item, item_doc, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            item_doc = doc_value.get(key) if isinstance(doc_value, dict) else None
            _walk(item, item_doc, f"{path}.{key}")


def _walk_dataclass(value: Any, doc_value: Any, path: str) -> None:
    cls_name = type(value).__name__
    for f in dataclasses.fields(value):
        child = getattr(value, f.name)
        child_doc = doc_value.get(f.name) if isinstance(doc_value, dict) else None
        child_path = f"{path}.{f.name}"
        fmt = _FORMATS.get((cls_name, f.name))
        if fmt is not None and child is not None:
            _check_format(fmt, child, child_path)
        _walk(child, child_doc, child_path)
    if cls_name == "TableContract":
        _check_table_contract(value, path)
    elif cls_name == "DataQuery":
        _check_data_query(value, path)
    elif cls_name == "KeyPredicate":
        _check_key_predicate(value, path)


def _check_format(kind: str, value: Any, path: str) -> None:
    if not isinstance(value, str):
        raise DocumentError("BAD_FORMAT", path, f"expected a string for a {kind}")
    if kind == "hash":
        if not _HASH.match(value):
            raise DocumentError("BAD_HASH_FORMAT", path,
                                "expected sha256: plus 64 lowercase hex characters")
    elif kind == "date":
        if not _DATE.match(value):
            raise DocumentError("BAD_DATE_FORMAT", path, "expected YYYY-MM-DD")
        _check_real_date(value, path)
    elif kind == "timestamp":
        try:
            parse_timestamp(value)
        except ValueError as exc:
            raise DocumentError("BAD_TIMESTAMP_FORMAT", path, str(exc)) from exc


def _check_real_date(value: str, path: str) -> None:
    try:
        datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise DocumentError("BAD_DATE_FORMAT", path, str(exc)) from exc


def _check_table_contract(tc: Any, path: str) -> None:
    names = [c.name for c in tc.columns]
    if len(set(names)) != len(names):
        raise DocumentError("DUPLICATE_COLUMN_NAME", f"{path}.columns",
                            "column names must be unique")
    if not tc.primary_key:
        raise DocumentError("EMPTY_PRIMARY_KEY", f"{path}.primary_key",
                            "a table's primary key must be non-empty")
    declared = set(names)
    key_lists = (("primary_key", tc.primary_key), ("partition_columns", tc.partition_columns),
                 ("filterable_columns", tc.filterable_columns),
                 ("orderable_columns", tc.orderable_columns))
    for field_name, cols in key_lists:
        if len(set(cols)) != len(cols):
            raise DocumentError("DUPLICATE_COLUMN_NAME", f"{path}.{field_name}",
                                f"duplicate entry in {field_name}")
        undeclared = [c for c in cols if c not in declared]
        if undeclared:
            raise DocumentError("UNDECLARED_COLUMN", f"{path}.{field_name}",
                                f"{undeclared[0]!r} is not one of this table's columns")


def _check_data_query(dq: Any, path: str) -> None:
    if dq.max_batch_rows <= 0:
        raise DocumentError("INVALID_LIMIT", f"{path}.max_batch_rows",
                            "must be a positive integer")
    if dq.max_result_rows <= 0:
        raise DocumentError("INVALID_LIMIT", f"{path}.max_result_rows",
                            "must be a positive integer")
    if dq.max_batch_rows > dq.max_result_rows:
        raise DocumentError("BATCH_EXCEEDS_RESULT", f"{path}.max_batch_rows",
                            "batch limit exceeds the result limit")
    if not dq.columns:
        raise DocumentError("EMPTY_COLUMNS", f"{path}.columns",
                            "must project at least one column")
    if len(set(dq.columns)) != len(dq.columns):
        raise DocumentError("DUPLICATE_SELECTED_COLUMN", f"{path}.columns",
                            "duplicate column in the projection")
    predicate_columns = [kp.column for kp in dq.key_filter]
    if len(set(predicate_columns)) != len(predicate_columns):
        raise DocumentError("DUPLICATE_PREDICATE_COLUMN", f"{path}.key_filter",
                            "the same column is filtered twice")
    if not dq.key_filter and dq.time_interval is None:
        raise DocumentError("QUERY_NOT_BOUNDED", path,
                            "needs at least one key predicate or a time bound")


def _check_key_predicate(kp: Any, path: str) -> None:
    if kp.operator == "eq":
        if len(kp.values) != 1:
            raise DocumentError("EQ_REQUIRES_ONE_VALUE", f"{path}.values",
                                "eq takes exactly one value")
        return
    if not kp.values:
        raise DocumentError("EMPTY_IN_VALUES", f"{path}.values",
                            "in requires at least one value")
    types = {type(v) for v in kp.values}
    if len(types) != 1:
        raise DocumentError("MIXED_IN_VALUE_TYPES", f"{path}.values",
                            "in values must share one scalar type")
    if len(set(kp.values)) != len(kp.values):
        raise DocumentError("DUPLICATE_IN_VALUE", f"{path}.values",
                            "in values must be unique")
    if list(kp.values) != sorted(kp.values):
        raise DocumentError("UNSORTED_IN_VALUES", f"{path}.values",
                            "in values must be in canonical sorted order")
