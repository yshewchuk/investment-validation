"""Factory-owned bounded catalog admission and detached native read lifetime.

No production caller, source coverage or numerical admission.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from copy import deepcopy

from engine.v2.contracts.data import SnapshotRef
from engine.v2.data import _bounded_catalog as probes
from engine.v2.data import _bounded_catalog_identity as identity
from engine.v2.data import _bounded_catalog_transfer as transfer
from engine.v2.data import _bounded_scan as scanning
from engine.v2.data import documents, errors
from engine.v2.foundation import DocumentError, to_document


def _expected_snapshot(snapshot):
    try:
        if type(snapshot) is not SnapshotRef:
            raise TypeError
        expected = documents.decode_document(SnapshotRef, to_document(snapshot))
        if expected != snapshot or not expected.snapshot_id:
            raise ValueError
        expected.snapshot_id.encode("utf-8")
        return expected
    except (DocumentError, ValueError, TypeError, RecursionError):
        raise errors.fail("SNAPSHOT_UNRESOLVED", "invalid expected snapshot") from None


def _sql_failure(exc):
    code = getattr(exc, "sqlite_errorcode", None)
    resource = isinstance(code, int) and code & 255 in probes._RESOURCE_CODES
    return errors.fail("RESOURCE_UNAVAILABLE" if resource else "MANIFEST_CORRUPT",
                       "catalog session unavailable" if resource else "catalog session malformed")


def _admit(repository, expected, table_name, limits):
    if type(repository) is not scanning.Repository:
        raise errors.fail("CONTRACT_MISMATCH", "invalid catalog session holder")
    conn = repository._conn
    if (type(conn) is not sqlite3.Connection or conn.row_factory is not sqlite3.Row
            or conn.text_factory is not str):
        raise errors.fail("CONTRACT_MISMATCH", "invalid catalog session configuration")
    probes._validate_inputs(conn, expected.snapshot_id, table_name, limits)
    began, primary = False, None
    try:
        if conn.in_transaction:
            raise errors.fail("CONTRACT_MISMATCH", "catalog session requires exclusive admission")
        conn.execute("BEGIN")
        began = True
        payload = transfer.transfer_catalog(conn, expected.snapshot_id, table_name, limits)
        return identity.reconstruct_catalog(payload, expected)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if began:
            try:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            except sqlite3.Error:
                if primary is None:
                    raise


class BoundedSnapshot:
    """Factory-created native view; reads are valid only within its context."""

    __slots__ = ("_active", "_resolved", "_contracts", "_records", "_selected",
                 "_scanner", "_iterator")

    def __init__(self):
        raise errors.fail("CONTRACT_MISMATCH", "bounded session requires factory admission")

    def _check_active(self):
        if not self._active:
            raise errors.fail("CONTRACT_MISMATCH", "bounded session is closed")

    def _check_id(self, snapshot_id):
        self._check_active()
        if type(snapshot_id) is not str or snapshot_id != self._resolved.snapshot.snapshot_id:
            raise errors.fail("SNAPSHOT_NOT_FOUND", "snapshot is not admitted")

    def _check_table(self, snapshot_ref, table_name):
        self._check_active()
        if (type(snapshot_ref) is not SnapshotRef or snapshot_ref != self._resolved.snapshot
                or type(table_name) is not str or table_name not in self._records):
            raise errors.fail("CONTRACT_MISMATCH", "table binding is not admitted")

    def resolve(self, snapshot_id):
        """Return the exact admitted reference, detached on every call."""
        self._check_id(snapshot_id)
        return deepcopy(self._resolved.snapshot)

    def resolve_full(self, snapshot_id):
        """Return complete detached all-table native membership."""
        self._check_id(snapshot_id)
        return deepcopy(self._resolved)

    def table_contract(self, snapshot_ref, table_name):
        self._check_table(snapshot_ref, table_name)
        return deepcopy(self._contracts[table_name])

    def fragment_records(self, snapshot_ref, table_name):
        self._check_table(snapshot_ref, table_name)
        return deepcopy(self._records[table_name])

    def _check_selected(self, table_name):
        self._check_active()
        if type(table_name) is not str or table_name != self._selected:
            raise errors.fail("CONTRACT_MISMATCH", "scan table is not admitted")

    def scan_population_bound(self, snapshot_id, *, table_name, table_contract_ref,
                              key_filter=(), time_interval=None):
        self._check_selected(table_name)
        return self._scanner.scan_population_bound(snapshot_id, table_name=table_name,
            table_contract_ref=table_contract_ref, key_filter=key_filter, time_interval=time_interval)

    def scan(self, query, *, table_name, max_object_bytes=None):
        self._check_active()
        return scanning.prepare_scan(self, query, table_name, max_object_bytes)


@contextmanager
def open_bounded_snapshot(repository, snapshot, *, table_name, limits):
    """Admit in one owned read transaction; release it before yielding metadata.

    Borrow exclusive connection use only during admission. Never touch holder
    pins/caches/settings or close it; a caller-owned transaction is refused.
    Native DataErrors retain their full envelopes, including primary failures
    during cleanup. Every reopening transfers and authenticates a fresh view.
    """
    expected = _expected_snapshot(snapshot)
    try:
        resolved = _admit(repository, expected, table_name, limits)
    except sqlite3.Error as exc:
        raise _sql_failure(exc) from None
    session = object.__new__(BoundedSnapshot)
    session._resolved = resolved
    contracts = {contract.contract_id: contract for contract in resolved.contracts}
    records = {record.fragment_id: record for record in resolved.records}
    session._contracts = {
        name: contracts[manifest.dataset_version_ref.table_contract_ref.contract_id]
        for name, manifest in resolved.table_manifests.items()}
    session._records = {name: tuple(records[ref.fragment_id] for ref in manifest.fragment_refs)
                        for name, manifest in resolved.table_manifests.items()}
    session._active = True
    session._selected, session._iterator = table_name, None
    session._scanner = scanning.FrozenRepository(session, repository._store)
    primary = None
    try:
        yield session
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            if session._iterator is not None:
                session._iterator.close()
        except BaseException:
            if primary is None:
                raise
        finally:
            session._active = False
