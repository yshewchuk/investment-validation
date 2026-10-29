"""Private bounded payload transfer; not native identity admission or a session."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from engine.v2.contracts.data import KnowledgeMode, TableContract
from engine.v2.data import _bounded_catalog as probes
from engine.v2.data import documents, errors


@dataclass(frozen=True)
class _Evidence:
    coverage_receipt_refs: tuple[str, ...]
    availability_evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class _Keys:
    primary_key_min: tuple[str | int | bool | float, ...]
    primary_key_max: tuple[str | int | bool | float, ...]


@dataclass(frozen=True)
class _Times:
    time_min: str | None
    time_max: str | None


@dataclass(frozen=True)
class _SnapshotJSON:
    finality_receipt_refs_json: tuple[str, ...]
    knowledge_mode_by_table_json: dict[str, KnowledgeMode]


@dataclass(frozen=True)
class _TableJSON:
    evidence_json: _Evidence
    partition_logical_hashes_json: dict[str, str]
    definition_json: TableContract


@dataclass(frozen=True)
class _FragmentJSON:
    key_bounds_json: _Keys
    time_bounds_json: _Times | None
    input_receipt_refs_json: tuple[str, ...]


@dataclass(frozen=True)
class CatalogPayload:
    """Ordered qualified-column maps; JSON columns hold native-decoded values.

    Detached transfer evidence only: counts/bytes do not authenticate identities.
    """

    probe: probes.CatalogProbe
    snapshot: dict
    tables: tuple[dict, ...]
    fragments: tuple[dict, ...]
    utf8_bytes: int


def _transfer(cursor, fields, source, where, params, count, total, cap, order=""):
    tokens = fields.split()
    names = [token.rstrip("?#") for token in tokens]
    rows = []
    for values in cursor.execute(
            f"SELECT {', '.join(names)} FROM {source} WHERE {where} {order} LIMIT ?",
            (*params, count + 1)):
        if len(rows) == count:
            raise errors.fail("MANIFEST_CORRUPT", "catalog transfer membership changed")
        for token, value in zip(tokens, values, strict=True):
            if value is None and token.endswith("?"):
                continue
            if type(value) is not (int if token.endswith("#") else str):
                raise errors.fail("MANIFEST_CORRUPT", "catalog transfer has invalid field")
            if type(value) is str:
                total += len(value.encode("utf-8"))
                probes._check_cap(total, cap)
        rows.append(dict(zip(names, values, strict=True)))
    if len(rows) != count:
        raise errors.fail("MANIFEST_CORRUPT", "catalog transfer has missing membership")
    return tuple(rows), total


def _decode(rows, cls):
    for row in rows:
        parsed = {key.split(".")[1]: (None if value is None else json.loads(
            value, object_pairs_hook=documents.reject_duplicate_keys))
            for key, value in row.items() if key.endswith("_json")}
        if any(value is not None and parsed[key.split(".")[1]] is None
               for key, value in row.items() if key.endswith("_json")):
            raise ValueError("unexpected JSON null")
        decoded = documents.decode_document(cls, parsed)
        for key in row:
            if key.endswith("_json"):
                row[key] = getattr(decoded, key.split(".")[1])


def transfer_catalog(conn: sqlite3.Connection, snapshot_id: str, table_name: str,
                     limits: dict[str, int]) -> CatalogPayload:
    """Probe all tables, transfer incrementally, then decode in one borrowed view.

    The caller owns its active read transaction and cleanup. No resolver, object
    store, factory or public session is invoked; native identity checking is later.
    """
    if (type(conn) is not sqlite3.Connection or conn.row_factory is not sqlite3.Row
            or conn.text_factory is not str):
        raise errors.fail("CONTRACT_MISMATCH", "invalid catalog transfer configuration")
    cursor = None
    try:
        probe = probes.probe_catalog(conn, snapshot_id, table_name, limits)
        cursor = conn.cursor()
        cursor.row_factory = None
        params, cap = (snapshot_id,), limits["max_catalog_bytes"]
        snapshots, total = _transfer(cursor, probes._SNAPSHOT, "data_snapshots s",
                                    "s.snapshot_id=?", params, 1, 0, cap)
        tables, total = _transfer(cursor, probes._TABLE, probes._TABLE_JOINS,
                                 "t.snapshot_id=?", params, probe.tables, total, cap,
                                 "ORDER BY t.table_name")
        fragments, total = _transfer(cursor, probes._FRAGMENT, probes._FRAGMENT_JOINS,
                                    "t.snapshot_id=?", params, probe.catalog_fragments,
                                    total, cap, "ORDER BY t.table_name, m.ordinal")
        # No JSON is parsed until the independent all-table UTF-8 cap has passed.
        for rows, cls in ((snapshots, _SnapshotJSON), (tables, _TableJSON),
                          (fragments, _FragmentJSON)):
            _decode(rows, cls)
        return CatalogPayload(probe, snapshots[0], tables, fragments, total)
    except sqlite3.Error as exc:
        code = getattr(exc, "sqlite_errorcode", None)
        resource = isinstance(code, int) and code & 255 in probes._RESOURCE_CODES
        raise errors.fail("RESOURCE_UNAVAILABLE" if resource else "MANIFEST_CORRUPT",
                          "catalog transfer unavailable" if resource
                          else "catalog transfer malformed") from None
    except (ValueError, RecursionError):
        raise errors.fail("MANIFEST_CORRUPT", "catalog transfer malformed") from None
    finally:
        if cursor is not None:
            cursor.close()
