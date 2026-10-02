"""D10: exact snapshot resolution — phase-2 guide §8.1, §12.

``Repository.resolve`` never touches object bytes on disk (that is scan's job,
P2-4, out of scope here) — it rebuilds identity purely from catalog rows, so
every fixture in this file uses the same synthetic (non-materialized)
``FragmentRecord``s ``tests/test_v2_data_manifests.py`` already builds, reused
per the task brief rather than re-derived. ``resolve_snapshot_head`` is the
one function here that does touch a real store: it publishes the resolved
``SnapshotRef`` document as a Phase 1 artifact.

The "corrupt membership" and "resolve never reads data_snapshot_heads" cases
work on a **consistent copy** of the catalog file (a fresh ``sqlite3``
``backup()``, never a raw file copy under WAL), so the corruption never
touches the connection other tests in this file still use.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import (  # noqa: E402
    DataQuery,
    KeyPredicate,
    TimeInterval,
)
from engine.v2.data import query as query_mod  # noqa: E402
from engine.v2.data.catalog import commit_snapshot  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.catalog import connect as ops_connect  # noqa: E402
from engine.v2.ops.snapshots import resolve_snapshot_head  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    hand_built_record,
    publish_and_inspect,
    table_from_rows,
)
from tests.ops_support import FakeClock  # noqa: E402
from tests.test_v2_data_commit import (  # noqa: E402
    _commit,
    _hash,
    _manifest_and_snapshot,
    _noop_fence,
)
from tests.test_v2_data_manifests import _SEC_CONTRACT, _record_for  # noqa: E402


def _catalog(tmp_path):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    return conn, clock


def _consistent_copy(src_path: Path, dst_path: Path) -> None:
    src = sqlite3.connect(str(src_path))
    dst = sqlite3.connect(str(dst_path))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()


# --------------------------------------------------------------------------
# unknown id, corrupt membership, never reads data_snapshot_heads
# --------------------------------------------------------------------------


def test_resolve_unknown_snapshot_id_not_found(tmp_path):
    conn, _clock = _catalog(tmp_path)
    with pytest.raises(DataError) as err:
        Repository(conn).resolve("snap_" + "0" * 32)
    assert err.value.code == "SNAPSHOT_NOT_FOUND"


def test_resolve_corrupt_membership_is_manifest_corrupt(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1")
    conn.close()

    copy_path = tmp_path / "corrupt.sqlite"
    _consistent_copy(tmp_path / "catalog.sqlite", copy_path)
    corrupt = ops_connect(copy_path)
    corrupt.execute("DROP TRIGGER data_version_fragments_no_delete")
    corrupt.execute("DROP TRIGGER data_version_fragments_no_update")
    dsv_id = corrupt.execute(
        "SELECT dataset_version_id FROM data_snapshot_tables WHERE snapshot_id = ?",
        (snap.snapshot_id,)).fetchone()[0]
    corrupt.execute("DELETE FROM data_version_fragments WHERE dataset_version_id = ?", (dsv_id,))

    with pytest.raises(DataError) as err:
        Repository(corrupt).resolve(snap.snapshot_id)
    assert err.value.code == "MANIFEST_CORRUPT"
    corrupt.close()


def test_resolve_never_reads_data_snapshot_heads(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1")
    conn.close()

    copy_path = tmp_path / "no_heads.sqlite"
    _consistent_copy(tmp_path / "catalog.sqlite", copy_path)
    stripped = ops_connect(copy_path)
    stripped.execute("DROP TABLE data_snapshot_heads")

    resolved = Repository(stripped).resolve(snap.snapshot_id)
    assert resolved.snapshot_id == snap.snapshot_id
    assert resolved.manifest_hash == snap.manifest_hash
    stripped.close()


# --------------------------------------------------------------------------
# a fixed reader while another connection advances the head
# --------------------------------------------------------------------------


def test_resolved_ref_fixed_while_another_connection_advances_the_head(tmp_path):
    conn, clock = _catalog(tmp_path)
    record_a = _record_for("2024")
    _, snap_a = _commit(conn, clock, [record_a], receipt_id="ra", scope="shadow")
    resolved_before = Repository(conn).resolve(snap_a.snapshot_id)

    store = ArtifactStore(tmp_path / "artifacts")
    ref_before = resolve_snapshot_head(conn, store, "shadow", clock=clock)
    content_before = json.loads(store.read_verified(ref_before).decode())
    assert content_before["snapshot_id"] == snap_a.snapshot_id

    conn2 = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    record_b = _record_for("2025")
    manifest_ab, snap_b = _manifest_and_snapshot([record_a, record_b])
    commit_snapshot(
        conn2, scope="shadow", request_hash=_hash("b-request"), contracts=[_SEC_CONTRACT],
        objects=[record_a.object_ref, record_b.object_ref], records=[record_a, record_b],
        manifests=[manifest_ab], snapshot=snap_b, expected_head_snapshot_id=snap_a.snapshot_id,
        expected_head_generation=1, receipt_id="rb", attempt_id="att-1", fence=1,
        fence_check=_noop_fence, clock=clock)

    # A's already-resolved ref is unchanged, and re-resolving A is identical.
    resolved_after = Repository(conn).resolve(snap_a.snapshot_id)
    assert resolved_after == resolved_before

    # resolve_snapshot_head now returns B's artifact.
    ref_after = resolve_snapshot_head(conn, store, "shadow", clock=clock)
    assert ref_after != ref_before
    content_after = json.loads(store.read_verified(ref_after).decode())
    assert content_after["snapshot_id"] == snap_b.snapshot_id
    conn2.close()


def test_resolve_snapshot_head_repeat_calls_same_ref_over_an_unchanged_head(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _commit(conn, clock, [record], receipt_id="r1", scope="shadow")
    store = ArtifactStore(tmp_path / "artifacts")

    ref1 = resolve_snapshot_head(conn, store, "shadow", clock=clock)
    ref2 = resolve_snapshot_head(conn, store, "shadow", clock=clock)
    assert ref1 == ref2


def test_resolve_snapshot_head_no_committed_head_raises(tmp_path):
    conn, _clock = _catalog(tmp_path)
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(DataError) as err:
        resolve_snapshot_head(conn, store, "shadow", clock=_clock)
    assert err.value.code == "SNAPSHOT_NOT_READY"


# --------------------------------------------------------------------------
# Coverage ratchet fix (2026-09-15): resolve_full/latest_dataset_version/
# table_contract/fragment_records and the private per-row corruption checks
# were entirely untested under the Phase 2 fixed suite even though
# Repository is already in scope here (D10).
# --------------------------------------------------------------------------


def test_read_only_refuses_a_nested_transaction(tmp_path):
    conn, _clock = _catalog(tmp_path)
    conn.execute("BEGIN")
    with pytest.raises(RuntimeError):
        Repository(conn).resolve("snap_" + "0" * 32)
    conn.execute("ROLLBACK")


def test_private_row_lookups_refuse_unknown_ids(tmp_path):
    """Direct calls into Repository's own per-row corruption checks --
    ``resolve``/``resolve_full`` reach these only through a fully corrupted
    catalog, so this exercises each refusal in isolation instead."""
    conn, _clock = _catalog(tmp_path)
    repo = Repository(conn)
    with pytest.raises(DataError) as err:
        repo._manifest(conn, "dsv_" + "0" * 32)
    assert err.value.code == "MANIFEST_CORRUPT"

    with pytest.raises(DataError) as err:
        repo._contract_ref(conn, "contract_" + "0" * 32)
    assert err.value.code == "MANIFEST_CORRUPT"

    with pytest.raises(DataError) as err:
        repo._full_contract(conn, "contract_" + "0" * 32)
    assert err.value.code == "MANIFEST_CORRUPT"

    with pytest.raises(DataError) as err:
        repo._record(conn, {"object_id": "obj_" + "0" * 32}, _SEC_CONTRACT)
    assert err.value.code == "MANIFEST_CORRUPT"


def test_resolve_full_happy_path_and_unknown_id(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1")

    resolved = Repository(conn).resolve_full(snap.snapshot_id)
    assert resolved.snapshot == Repository(conn).resolve(snap.snapshot_id)
    assert len(resolved.records) == 1
    assert resolved.records[0].fragment_id == record.fragment_id
    assert len(resolved.contracts) == 1
    assert len(resolved.objects) == 1

    with pytest.raises(DataError) as err:
        Repository(conn).resolve_full("snap_" + "0" * 32)
    assert err.value.code == "SNAPSHOT_NOT_FOUND"


def test_resolve_and_resolve_full_refuse_a_tampered_snapshot_row(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1")
    conn.close()

    copy_path = tmp_path / "tampered.sqlite"
    _consistent_copy(tmp_path / "catalog.sqlite", copy_path)
    tampered = ops_connect(copy_path)
    tampered.execute("DROP TRIGGER data_snapshots_no_delete")
    tampered.execute("DROP TRIGGER data_snapshots_no_update")
    tampered.execute("UPDATE data_snapshots SET manifest_hash = ? WHERE snapshot_id = ?",
                     ("sha256:" + "0" * 64, snap.snapshot_id))

    with pytest.raises(DataError) as err:
        Repository(tampered).resolve(snap.snapshot_id)
    assert err.value.code == "MANIFEST_CORRUPT"

    with pytest.raises(DataError) as err:
        Repository(tampered).resolve_full(snap.snapshot_id)
    assert err.value.code == "MANIFEST_CORRUPT"
    tampered.close()


def test_latest_dataset_version_empty_and_populated(tmp_path):
    conn, clock = _catalog(tmp_path)
    repo = Repository(conn)
    manifest, records = repo.latest_dataset_version(_SEC_CONTRACT.contract_id)
    assert manifest is None
    assert records == ()

    record = _record_for("2024")
    _commit(conn, clock, [record], receipt_id="r1")
    manifest, records = repo.latest_dataset_version(_SEC_CONTRACT.contract_id)
    assert manifest is not None
    assert len(records) == 1
    assert records[0].fragment_id == record.fragment_id


def test_latest_dataset_version_trusts_rowid_not_the_wall_clock(tmp_path):
    """``registered_at`` is wall-clock text; a backward clock step (WSL resume)
    can make a later insert carry an *earlier* timestamp. "Latest" must then
    still mean the most recently registered version, so ordering keys on
    ``rowid`` alone -- never on ``registered_at``."""
    conn, clock = _catalog(tmp_path)
    repo = Repository(conn)
    record_2024 = _record_for("2024")
    _commit(conn, clock, [record_2024], receipt_id="r1", scope="shadow")

    clock.value -= timedelta(days=1)  # the second registration is stamped EARLIER
    record_2025 = _record_for("2025")
    _commit(conn, clock, [record_2025], receipt_id="r2", scope="other")

    manifest, records = repo.latest_dataset_version(_SEC_CONTRACT.contract_id)
    assert manifest is not None
    assert len(records) == 1
    assert records[0].fragment_id == record_2025.fragment_id


# --------------------------------------------------------------------------
# Phase 6 slice 3: resolve_pinned/resolve_full_pinned — snapshot-read
# foundation for research tooling (UD-4)
# --------------------------------------------------------------------------


def test_resolve_pinned_no_committed_head_raises(tmp_path):
    conn, _clock = _catalog(tmp_path)
    with pytest.raises(DataError) as err:
        Repository(conn).resolve_pinned("shadow")
    assert err.value.code == "SNAPSHOT_NOT_READY"

    with pytest.raises(DataError) as err:
        Repository(conn).resolve_full_pinned("shadow")
    assert err.value.code == "SNAPSHOT_NOT_READY"


def test_resolve_pinned_does_not_fall_back_to_a_different_scopes_head(tmp_path):
    """Negative control (plan): a snapshot committed but never promoted to
    THIS scope's head must refuse, not silently fall back to some other
    head that does exist."""
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _commit(conn, clock, [record], receipt_id="r1", scope="candidate")

    with pytest.raises(DataError) as err:
        Repository(conn).resolve_pinned("shadow")
    assert err.value.code == "SNAPSHOT_NOT_READY"


def test_resolve_pinned_tracks_head_moves(tmp_path):
    conn, clock = _catalog(tmp_path)
    record_a = _record_for("2024")
    _, snap_a = _commit(conn, clock, [record_a], receipt_id="ra", scope="shadow")

    resolved = Repository(conn).resolve_pinned("shadow")
    assert resolved == snap_a
    full = Repository(conn).resolve_full_pinned("shadow")
    assert full.snapshot == snap_a

    record_b = _record_for("2025")
    manifest_ab, snap_b = _manifest_and_snapshot([record_a, record_b])
    commit_snapshot(
        conn, scope="shadow", request_hash=_hash("b-request"), contracts=[_SEC_CONTRACT],
        objects=[record_a.object_ref, record_b.object_ref], records=[record_a, record_b],
        manifests=[manifest_ab], snapshot=snap_b, expected_head_snapshot_id=snap_a.snapshot_id,
        expected_head_generation=1, receipt_id="rb", attempt_id="att-1", fence=1,
        fence_check=_noop_fence, clock=clock)

    assert Repository(conn).resolve_pinned("shadow") == snap_b


def test_resolve_pinned_refuses_a_tampered_head_snapshot(tmp_path):
    """Negative control (brief): a tampered manifest hash is refused even
    when reached through resolve_pinned's head lookup, not just resolve."""
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1", scope="shadow")
    conn.close()

    copy_path = tmp_path / "tampered_pinned.sqlite"
    _consistent_copy(tmp_path / "catalog.sqlite", copy_path)
    tampered = ops_connect(copy_path)
    tampered.execute("DROP TRIGGER data_snapshots_no_delete")
    tampered.execute("DROP TRIGGER data_snapshots_no_update")
    tampered.execute("UPDATE data_snapshots SET manifest_hash = ? WHERE snapshot_id = ?",
                     ("sha256:" + "0" * 64, snap.snapshot_id))

    with pytest.raises(DataError) as err:
        Repository(tampered).resolve_pinned("shadow")
    assert err.value.code == "MANIFEST_CORRUPT"

    with pytest.raises(DataError) as err:
        Repository(tampered).resolve_full_pinned("shadow")
    assert err.value.code == "MANIFEST_CORRUPT"
    tampered.close()


def test_resolve_rejects_an_implicit_latest_sentinel_not_an_explicit_id(tmp_path):
    """Negative control (brief): a read without an explicit, real snapshot id
    is refused — no "latest"/"current" sentinel is special-cased anywhere in
    this foundation, on the underlying resolve/resolve_full it is built on."""
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _commit(conn, clock, [record], receipt_id="r1", scope="shadow")

    for sentinel in ("latest", "current", "", None):
        with pytest.raises(DataError) as err:
            Repository(conn).resolve(sentinel)
        assert err.value.code == "SNAPSHOT_NOT_FOUND"
        with pytest.raises(DataError) as err:
            Repository(conn).resolve_full(sentinel)
        assert err.value.code == "SNAPSHOT_NOT_FOUND"


def test_table_contract_and_fragment_records_refuse_unknown_table(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1")
    repo = Repository(conn)

    contract = repo.table_contract(snap, "securities")
    assert contract.contract_id == _SEC_CONTRACT.contract_id
    records = repo.fragment_records(snap, "securities")
    assert len(records) == 1

    with pytest.raises(DataError) as err:
        repo.table_contract(snap, "not_a_real_table")
    assert err.value.code == "CONTRACT_MISMATCH"
    with pytest.raises(DataError) as err:
        repo.fragment_records(snap, "not_a_real_table")
    assert err.value.code == "CONTRACT_MISMATCH"


# --------------------------------------------------------------------------
# task brief #286: the vectorized batch filter. A query the batch matcher
# can express must materialize ZERO row dicts for rows it drops (the whole
# point of the fix), and the per-row fallback it defers to when
# ``compile_batch_matcher`` returns ``None`` must agree with it exactly on
# real data — including rows whose filtered column is null.
# --------------------------------------------------------------------------

_SEC = contract_for("securities")
_SEC_REF = contract_ref_for(_SEC)
_OC = contract_for("option_chains")
_OC_REF = contract_ref_for(_OC)


def _securities_row(ticker: str, year: int) -> dict:
    return dict(ticker=ticker, year=year, first_date=None, last_date=None, mcap_usd=1.5e9,
                mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False,
                n_obs=250, src="orats")


def _chain_row(ticker, year: int, day: int, *, with_obs_date: bool = True) -> dict:
    from datetime import datetime
    return dict(ticker=ticker, obs_date=datetime(year, 1, day) if with_obs_date else None,
                year=year, expiry=datetime(year, 2, 16), dte=45, strike=100.0, right="C",
                bid=1.0, ask=1.2, mid=1.1, iv=30.0, delta=0.5, spot=100.0, src="orats",
                src_file="f.parquet", chain_kind="entry", volume=None, open_interest=None,
                bid_size=None, ask_size=None, quote_repaired=False)


def test_nonmatching_vectorized_filter_materializes_zero_row_dicts(tmp_path, monkeypatch):
    """A ``key_filter`` that matches nothing in a surviving fragment removes
    every row at the Arrow-batch level: ``_decode_rows`` is still reached
    (the fragment WAS opened — the "AAB" predicate value falls inside the
    fragment's real key bounds, so pruning alone cannot explain the empty
    result) but materializes zero Python dicts, and the scan completes with
    zero rows and no exception."""
    conn, clock, store = catalog_and_store(tmp_path)
    record = publish_and_inspect(store, _SEC, _SEC_REF,
                                 [_securities_row(t, 2024) for t in ("AAA", "BBB", "CCC")], "2024")
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": _SEC})
    repo = Repository(conn, store)

    decoded_rows: list[dict] = []
    decode_calls: list[int] = []
    real_decode = Repository._decode_rows

    def counting_decode(self, batch, needed, present, missing):
        decode_calls.append(batch.num_rows)
        for row in real_decode(self, batch, needed, present, missing):
            decoded_rows.append(row)
            yield row

    monkeypatch.setattr(Repository, "_decode_rows", counting_decode)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_SEC_REF, columns=("ticker", "year"),
        key_filter=(KeyPredicate(column="ticker", operator="in", values=("AAB",)),),
        order_by=("ticker", "year"), max_batch_rows=10, max_result_rows=10)
    batches = list(repo.scan(query, table_name="securities"))
    assert sum(batch.num_rows for batch in batches) == 0
    assert decoded_rows == []
    assert decode_calls and all(num_rows == 0 for num_rows in decode_calls)


def test_forced_row_fallback_agrees_with_vectorized_filter(tmp_path, monkeypatch):
    """Same real multi-fragment scan twice: once through the vectorized
    batch filter (a ``ticker`` ``in`` predicate -- string equality, this
    package's one vectorized case), once with ``compile_batch_matcher``
    monkeypatched to always return ``None`` (the per-row path ``repository``
    used before #286, and the one every OTHER query still takes). A null
    ticker row may not appear in either result, and the two results must be
    byte-identical."""
    conn, clock, store = catalog_and_store(tmp_path)
    table_2024 = table_from_rows(_OC, [_chain_row("AAA", 2024, 2), _chain_row("ZZZ", 2024, 3),
                                       _chain_row(None, 2024, 4)])
    frag_2024 = hand_built_record(
        store, _OC, _OC_REF, table_2024, partition_key="2024", row_count=3,
        primary_key_min=("AAA", "2024-01-02T00:00:00.000000", "2024-02-16T00:00:00.000000",
                         100.0, "C"),
        primary_key_max=("ZZZ", "2024-01-04T00:00:00.000000", "2024-02-16T00:00:00.000000",
                         100.0, "C"))
    table_2025 = table_from_rows(_OC, [_chain_row("AAA", 2025, 2), _chain_row("BBB", 2025, 3)])
    frag_2025 = hand_built_record(
        store, _OC, _OC_REF, table_2025, partition_key="2025", row_count=2,
        primary_key_min=("AAA", "2025-01-02T00:00:00.000000", "2025-02-16T00:00:00.000000",
                         100.0, "C"),
        primary_key_max=("BBB", "2025-01-03T00:00:00.000000", "2025-02-16T00:00:00.000000",
                         100.0, "C"))
    snap = commit_tables(conn, clock, {"option_chains": [frag_2024, frag_2025]},
                         {"option_chains": _OC})
    repo = Repository(conn, store)

    def _query() -> DataQuery:
        return DataQuery(
            snapshot_id=snap.snapshot_id, table_contract_ref=_OC_REF,
            columns=("ticker", "obs_date"),
            key_filter=(KeyPredicate(column="ticker", operator="in", values=("AAA", "BBB")),),
            order_by=("ticker", "obs_date", "expiry", "strike", "right"),
            max_batch_rows=10, max_result_rows=10)

    vectorized = [row for batch in repo.scan(_query(), table_name="option_chains")
                  for row in batch.to_pylist()]
    monkeypatch.setattr(query_mod, "compile_batch_matcher",
                        lambda contract, query: None)
    fallback = [row for batch in repo.scan(_query(), table_name="option_chains")
                for row in batch.to_pylist()]
    assert len(vectorized) == 3  # the two AAA rows plus the 2025 BBB row; null ticker excluded
    assert sorted(r["ticker"] for r in vectorized) == ["AAA", "AAA", "BBB"]
    assert fallback == vectorized


def test_time_interval_query_always_takes_row_path(tmp_path):
    """#286 follow-up: a query with a ``time_interval`` always falls back
    to the per-row path now (``compile_batch_matcher`` refuses any
    interval at all) -- a null ``obs_date`` row is excluded the same way
    it always was, via the row path's own ``row.get(column) is None``."""
    conn, clock, store = catalog_and_store(tmp_path)
    table_2025 = table_from_rows(_OC, [_chain_row("AAA", 2025, 2),
                                       _chain_row("BBB", 2025, 3, with_obs_date=False)])
    frag_2025 = hand_built_record(
        store, _OC, _OC_REF, table_2025, partition_key="2025", row_count=2,
        primary_key_min=("AAA", "2025-01-02T00:00:00.000000", "2025-02-16T00:00:00.000000",
                         100.0, "C"),
        primary_key_max=("BBB", "2025-01-03T00:00:00.000000", "2025-02-16T00:00:00.000000",
                         100.0, "C"))
    snap = commit_tables(conn, clock, {"option_chains": [frag_2025]}, {"option_chains": _OC})
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_OC_REF,
        columns=("ticker", "obs_date"), key_filter=(),
        time_interval=TimeInterval(column="obs_date", start_inclusive="2024-01-01",
                                   end_exclusive="2026-01-01"),
        order_by=("ticker", "obs_date", "expiry", "strike", "right"),
        max_batch_rows=10, max_result_rows=10)
    assert query_mod.compile_batch_matcher(_OC, query) is None
    rows = [row for batch in repo.scan(query, table_name="option_chains")
            for row in batch.to_pylist()]
    assert [r["ticker"] for r in rows] == ["AAA"]


def test_batch_matcher_evaluation_failure_falls_back_for_that_fragment(tmp_path, monkeypatch):
    """A batch matcher that compiles successfully but raises when actually
    EVALUATED against a real batch -- a failure mode compile-time
    validation from contract/query alone cannot rule out in general --
    must not crash the scan: repository.py catches it per fragment and
    falls back to the row matcher for the rest of that fragment, giving
    the independently-known-correct result (not merely the same result
    as a second, equally-fallible run of the real matcher -- that
    agreement is already covered by
    test_forced_row_fallback_agrees_with_vectorized_filter above)."""
    conn, clock, store = catalog_and_store(tmp_path)
    record = publish_and_inspect(store, _SEC, _SEC_REF,
                                 [_securities_row(t, 2024) for t in ("AAA", "BBB", "CCC")], "2024")
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": _SEC})
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_SEC_REF, columns=("ticker", "year"),
        key_filter=(KeyPredicate(column="ticker", operator="in", values=("AAA", "BBB")),),
        order_by=("ticker", "year"), max_batch_rows=10, max_result_rows=10)
    # Independently known from _securities_row/the query alone, not derived
    # from any scan: AAA and BBB (year 2024) in ticker order; CCC is
    # excluded by the key_filter.
    expected = [{"ticker": "AAA", "year": 2024}, {"ticker": "BBB", "year": 2024}]

    real_compile = query_mod.compile_batch_matcher

    def failing_compile(contract, q):
        real_mask = real_compile(contract, q)
        assert real_mask is not None
        state = {"calls": 0}

        def _raising_mask(columns, num_rows):
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("simulated evaluation failure")
            return real_mask(columns, num_rows)

        return _raising_mask

    monkeypatch.setattr(query_mod, "compile_batch_matcher", failing_compile)
    result = [row for batch in repo.scan(query, table_name="securities")
             for row in batch.to_pylist()]
    assert result == expected

