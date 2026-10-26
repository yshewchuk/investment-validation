"""Private native scanner dispatch over an admitted session, never a catalog."""
from engine.v2.data import errors, objects
from engine.v2.data.repository import Repository


class FrozenRepository(Repository):
    """Reuse native scanning only; the borrowed holder's SQL is never retained."""

    def __init__(self, session, store):
        super().__init__(None, store, pin_snapshots=False)
        self._session = session

    def resolve(self, snapshot_id):
        return self._session.resolve(snapshot_id)

    def _table_records(self, snap, table_name, table_contract_ref, key_filter=()):
        session = self._session
        session._check_selected(table_name)
        session._check_table(snap, table_name)
        if table_contract_ref != snap.table_versions[table_name].table_contract_ref:
            raise errors.fail("CONTRACT_MISMATCH",
                "table_contract_ref does not match the version pinned under this table name",
                details={"table_name": table_name})
        return session.table_contract(snap, table_name), session.fragment_records(snap, table_name)


def prepare_scan(session, query, table_name, max_object_bytes):
    session._check_selected(table_name)
    objects.validate_max_object_bytes(max_object_bytes)
    if session._scanner._store is None:
        raise errors.fail("CONTRACT_MISMATCH", "bounded scan requires an object store")
    return ScanIterator(session, session._scanner.scan(
        query, table_name=table_name, max_object_bytes=max_object_bytes))


class ScanIterator:
    """Lifetime guard wraps even an unstarted or already exhausted generator."""

    def __init__(self, session, stream):
        self._session, self._stream = session, stream
        self._closed = False
        self._running = False

    def __iter__(self):
        self._session._check_active()
        return self

    def __next__(self):
        session = self._session
        session._check_active()
        if self._closed:
            raise StopIteration
        if self._running or (session._iterator is not None and session._iterator is not self):
            raise errors.fail("CONTRACT_MISMATCH", "bounded scan is already active")
        session._iterator = self
        try:
            self._running = True
            try:
                return next(self._stream)
            finally:
                self._running = False
        except BaseException:
            # Closing a failed stream must not replace its native exception.
            try:
                self.close()
            except BaseException:
                pass
            raise

    def close(self):
        self._session._check_active()
        if self._running:
            raise errors.fail("CONTRACT_MISMATCH", "bounded scan is already active")
        self._closed = True
        try:
            self._stream.close()
        finally:
            if self._session._iterator is self:
                self._session._iterator = None
