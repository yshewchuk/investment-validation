"""Synthetic complete pinned metadata proof; no source or numerical admission."""
import json
import sqlite3
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest

from engine.v2.data import bounded_snapshot, errors, manifests, objects
from engine.v2.data.repository import Repository
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.research import magnitude_confirmed_metadata as index
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    hand_built_record,
    publish_and_inspect,
    table_from_rows,
)

TABLE = "earnings_events"
COLUMNS = ("event_id", "ticker", "event_date", "session", "date_conflict",
           "event_cluster_id", "src_orats")
LIMITS = dict(max_tickers=10, max_catalog_tables=10, max_catalog_fragments=20,
    max_catalog_bytes=1_000_000, max_partitions=10, max_fragments=20,
    max_scan_rows=100, max_object_bytes=1_000_000, max_total_object_bytes=10_000_000,
    max_batch_rows=10, max_retained_rows=100, max_retained_metadata_bytes=100_000,
    max_proof_bytes=1_000_000)


def row(identity, year=2020, ticker="A", day=1, **changes):
    values = dict(event_id=identity, ticker=ticker, event_date=datetime(year, 1, day),
        year=year, session="BMO", session_src="orats", annc_tod=None, src_orats=True,
        src_oquants=False, src_nasdaq=False, src_yfinance=False, date_agree=True,
        date_conflict=False, updated_at=None, event_cluster_id=None, claim_count=None,
        reconciliation=None)
    return dict(values, **changes)


@pytest.fixture
def source(tmp_path):
    """Each construction uses native objects/catalog and an independently resolved oracle."""
    connections = []
    def make(groups=None, *, contract=None, unrelated=True, hand_built=False):
        root = tmp_path / str(len(connections))
        root.mkdir()
        conn, clock, store = catalog_and_store(root)
        connections.append(conn)
        contract = contract or contract_for(TABLE)
        ref = contract_ref_for(contract)
        if groups is None:
            groups = [("1999", [row("a-old", 1999), row("b-old", 1999, day=2)]),
                      ("2000", [row("c-old", 2000), row("d-old", 2000, day=2)]),
                      ("2025", [row("e-new", 2025),
                                row("f-unconfirmed", 2025, "B", session=None),
                                row("g-other", 2025, "C"),
                                row("h-unrequested", 2025, "D")])]
        records = []
        for partition, rows in groups:
            rows = sorted(rows, key=lambda value: value["event_id"])
            if hand_built or not rows:
                records.append(hand_built_record(store, contract, ref,
                    table_from_rows(contract, rows), partition_key=partition,
                    row_count=len(rows), primary_key_min=(rows[0]["event_id"],) if rows else ("a",),
                    primary_key_max=(rows[-1]["event_id"],) if rows else ("z",),
                    logical_label=f"synthetic-{partition}"))
            else:
                records.append(publish_and_inspect(store, contract, ref, rows, partition))
        tables, contracts = {TABLE: records}, {TABLE: contract}
        other_record = None
        if unrelated:
            other = contract_for("securities")
            other_record = publish_and_inspect(store, other, contract_ref_for(other),
                [dict(ticker="POISON", year=2020, first_date=None, last_date=None,
                      mcap_usd=1., mcap_log=0., mcap_raw=1., mcap_unit_era="billions",
                      mcap_quantized=False, n_obs=1, src="orats")], "2020")
            tables["securities"], contracts["securities"] = [other_record], other
        snapshot = commit_tables(conn, clock, tables, contracts)
        holder = Repository(conn, store, pin_snapshots=True)
        oracle = holder.resolve_full(snapshot.snapshot_id)
        return SimpleNamespace(conn=conn, holder=holder, snapshot=snapshot, oracle=oracle,
            store=store, contract=contract, ref=ref, records=records, other=other_record)
    yield make
    for conn in connections:
        conn.set_authorizer(None)
        conn.close()


def build(source, *, tickers=None, limits=None):
    return index.build_index(source.holder, source.snapshot,
        tickers=["A", "B", "C", "Z"] if tickers is None else tickers,
        limits=dict(LIMITS) if limits is None else limits)


def refused(code, call):
    with pytest.raises(errors.DataError) as caught:
        call()
    assert caught.value.code == code
    return caught.value


def forbidden(*_, **__):
    pytest.fail("forbidden source access")


def test_native_complete_proof_exact_shape_source_and_ordinal_history(source):
    fixture = source()
    result = build(fixture)
    assert set(result) == {"schema_version", "policy", "source", "tickers", "sequences",
                           "counts", "index_id"}
    assert result["schema_version"] == "magnitude_confirmed_metadata.v1"
    assert result["policy"] == dict(confirmation="src_orats_and_session.v1", warmup_events=4,
        conflicts="refuse_ticker_scoped_view.v1", coverage="complete_pinned_view_only.v1")
    assert result["source"] == dict(snapshot=to_document(fixture.snapshot), table=TABLE,
        contract_ref=to_document(fixture.ref),
        dataset_manifest=to_document(fixture.oracle.table_manifests[TABLE]),
        fragment_records=[to_document(record) for record in fixture.records],
        columns=list(COLUMNS))
    assert result["tickers"] == ["A", "B", "C", "Z"]
    assert result["counts"] == dict(partitions=3, fragments=3, candidate_rows=8,
                                    selected_rows=7, confirmed_rows=6)
    assert result["index_id"] == content_hash({k: v for k, v in result.items() if k != "index_id"})
    assert json.loads(json.dumps(result)) == result
    a, b, c, z = result["sequences"]
    assert a["status"] == "confirmed_events" and a["first_confirmed_date"] == "1999-01-01"
    assert [event["ordinal"] for event in a["events"]] == list(range(5))
    assert [event["warmup_complete"] for event in a["events"]] == [False] * 4 + [True]
    for event in a["events"] + c["events"]:
        assert set(event) == set(COLUMNS) | {"ordinal", "warmup_complete"}
        assert event["src_orats"] is True and event["date_conflict"] is False
    assert b == dict(ticker="B", status="no_confirmed_events", selected_rows=1,
                    confirmed_rows=0, first_confirmed_date=None, events=[])
    assert z == dict(ticker="Z", status="absent_in_snapshot", selected_rows=0,
                    confirmed_rows=0, first_confirmed_date=None, events=[])


@pytest.mark.parametrize("batch", [1, 2, 100_000])
def test_request_batch_invariance_detachment_single_partition_scan(source, monkeypatch, batch):
    fixture = source()
    expected = build(fixture)
    scans, reads = [], []
    native_scan, native_read = bounded_snapshot.BoundedSnapshot.scan, objects.read_verified_bytes
    def scan(session, query, **kwargs):
        scans.append((query, kwargs))
        assert not fixture.conn.in_transaction
        fixture.conn.set_authorizer(lambda *_: sqlite3.SQLITE_DENY)
        return native_scan(session, query, **kwargs)
    def read(store, obj, **kwargs):
        assert obj.object_id != fixture.other.object_ref.object_id
        reads.append(obj.object_id)
        return native_read(store, obj, **kwargs)
    monkeypatch.setattr(bounded_snapshot.BoundedSnapshot, "scan", scan)
    monkeypatch.setattr(objects, "read_verified_bytes", read)
    for name in ("resolve", "resolve_full", "table_contract", "fragment_records"):
        monkeypatch.setattr(Repository, name, forbidden)
    actual = build(fixture, tickers=["Z", "C", "B", "A"],
                   limits=dict(LIMITS, max_batch_rows=batch))
    assert actual == expected and len(scans) == len(reads) == 3
    for (query, kwargs), partition, bound in zip(scans, (1999, 2000, 2025), (2, 2, 4)):
        assert query.columns == COLUMNS and query.order_by == ("event_id",)
        assert query.time_interval is None and query.max_result_rows == bound
        assert query.max_batch_rows == min(batch, 50_000, bound)
        assert [(p.column, p.operator, p.values) for p in query.key_filter] == [
            ("year", "eq", (partition,)), ("ticker", "in", ("A", "B", "C", "Z"))]
        assert kwargs == dict(table_name=TABLE, max_object_bytes=LIMITS["max_object_bytes"])
    actual["source"]["snapshot"]["table_versions"].clear()
    actual["sequences"][0]["events"].clear()
    assert fixture.snapshot.table_versions and expected["sequences"][0]["events"]


@pytest.mark.parametrize("tickers", [[], (), iter(["A"]), ["A", "A"], [""], [" A"],
                                     ["A\0"], ["\ud800"], [1]])
def test_bad_tickers_before_repository_access(monkeypatch, tickers):
    monkeypatch.setattr(bounded_snapshot, "open_bounded_snapshot", forbidden)
    refused("CONTRACT_MISMATCH", lambda: index.build_index(None, None, tickers=tickers,
                                                            limits=dict(LIMITS)))


@pytest.mark.parametrize("key", LIMITS)
@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "2", sys.maxsize])
def test_cap_types_refuse_before_repository_access(monkeypatch, key, value):
    monkeypatch.setattr(bounded_snapshot, "open_bounded_snapshot", forbidden)
    refused("CONTRACT_MISMATCH", lambda: index.build_index(None, None, tickers=["A"],
                                                            limits=dict(LIMITS, **{key: value})))


@pytest.mark.parametrize("limits", [None, [], {}, dict(LIMITS, extra=1),
                                     {k: v for k, v in LIMITS.items() if k != "max_tickers"}])
def test_closed_limits_shape(monkeypatch, limits):
    monkeypatch.setattr(bounded_snapshot, "open_bounded_snapshot", forbidden)
    refused("CONTRACT_MISMATCH", lambda: index.build_index(None, None, tickers=["A"], limits=limits))


@pytest.mark.parametrize("key,bound", [("max_tickers", 4), ("max_catalog_tables", 2),
    ("max_catalog_fragments", 4), ("max_partitions", 3), ("max_fragments", 3),
    ("max_scan_rows", 8)])
def test_count_cap_equality_and_one_over_before_any_objects(source, monkeypatch, key, bound):
    fixture = source()
    expected = build(fixture)
    assert build(fixture, limits=dict(LIMITS, **{key: bound})) == expected
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    refused("RESULT_LIMIT_EXCEEDED", lambda: build(fixture, limits=dict(LIMITS, **{key: bound - 1})))


def test_complete_preflight_byte_totals_before_first_object(source, monkeypatch):
    fixture = source()
    total = sum(record.object_ref.byte_size for record in fixture.records)
    assert build(fixture, limits=dict(LIMITS, max_total_object_bytes=total)) == build(fixture)
    maximum = max(record.object_ref.byte_size for record in fixture.records)
    assert build(fixture, limits=dict(LIMITS, max_object_bytes=maximum)) == build(fixture)
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    refused("RESULT_LIMIT_EXCEEDED", lambda: build(fixture,
        limits=dict(LIMITS, max_total_object_bytes=total - 1)))
    refused("RESULT_LIMIT_EXCEEDED", lambda: build(fixture,
        limits=dict(LIMITS, max_object_bytes=maximum - 1)))
    refused("RESULT_LIMIT_EXCEEDED", lambda: build(fixture, limits=dict(LIMITS, max_catalog_bytes=1)))


def test_retained_counts_bytes_and_full_proof_caps(source):
    fixture = source()
    expected = build(fixture)
    selected = [dict(row("a-old", 1999)), dict(row("b-old", 1999, day=2)),
                dict(row("c-old", 2000)), dict(row("d-old", 2000, day=2)),
                dict(row("e-new", 2025)), dict(row("f-unconfirmed", 2025, "B", session=None)),
                dict(row("g-other", 2025, "C"))]
    rows = [{key: value[key] for key in COLUMNS} for value in selected]
    for value in rows:
        value["event_date"] = value["event_date"].date().isoformat()
    retained_bytes = sum(len(canonical_json(value).encode("utf-8")) for value in rows)
    proof_bytes = len(canonical_json(expected).encode("utf-8"))
    for key, bound in (("max_retained_rows", 7), ("max_retained_metadata_bytes", retained_bytes),
                       ("max_proof_bytes", proof_bytes)):
        assert build(fixture, limits=dict(LIMITS, **{key: bound})) == expected
        refused("RESULT_LIMIT_EXCEEDED", lambda: build(fixture, limits=dict(LIMITS, **{key: bound - 1})))


@pytest.mark.parametrize("groups", [[], [("2020", [])]])
def test_empty_snapshot_and_zero_population_partition(source, monkeypatch, groups):
    fixture = source(groups, unrelated=False)
    requests = []
    native = bounded_snapshot.BoundedSnapshot.scan
    def scan(session, query, **kwargs):
        requests.append(query)
        return native(session, query, **kwargs)
    monkeypatch.setattr(bounded_snapshot.BoundedSnapshot, "scan", scan)
    result = build(fixture)
    assert result["counts"] == dict(partitions=len(groups), fragments=len(groups),
                                    candidate_rows=0, selected_rows=0, confirmed_rows=0)
    assert all(sequence["status"] == "absent_in_snapshot" for sequence in result["sequences"])
    assert len(requests) == len(groups)
    assert all(query.max_result_rows == 0 and query.max_batch_rows == 1 for query in requests)


def test_same_columns_different_exact_contract_refuses_without_objects(source, monkeypatch):
    definition = replace(contract_for(TABLE), provenance_semantics="synthetic changed meaning")
    definition = replace(definition, definition_hash=manifests.table_contract_hash(definition))
    fixture = source(contract=definition)
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    refused("UNSUPPORTED_CONTRACT", lambda: build(fixture))


@pytest.mark.parametrize("partition", ["20", "020", "02020", "0000", "10000", "abcd", "２０２０"])
def test_noncanonical_partition_refuses_before_objects(source, monkeypatch, partition):
    fixture = source([(partition, [row("a")])], hand_built=True, unrelated=False)
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    refused("CONTRACT_MISMATCH", lambda: build(fixture))


@pytest.mark.parametrize("partition", ["0001", "0999"])
def test_canonical_year_native_candidate_mismatch_refuses_without_partial_proof(source, monkeypatch, partition):
    # Native year pruning uses str(integer), so it cannot exhaust zero-padded
    # membership. Syntax acceptance must not authorize an incomplete inventory.
    fixture = source([(partition, [row("a")])], hand_built=True, unrelated=False)
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    refused("CONTRACT_MISMATCH", lambda: build(fixture))


def test_late_native_failure_preserves_full_envelope_and_closes_session(source, monkeypatch, capsys):
    fixture = source()
    native = bounded_snapshot.BoundedSnapshot.scan
    primary = errors.fail("RESOURCE_UNAVAILABLE", "synthetic complete native diagnostic",
        stage="synthetic-stage", details={"why": ["synthetic-secret-not-for-logs", 3]})
    sessions, closed = [], []
    def scan(session, query, **kwargs):
        sessions.append(session)
        def stream():
            try:
                yield from native(session, query, **kwargs)
                if len(sessions) == 3:
                    raise primary
            finally:
                closed.append(True)
        return stream()
    monkeypatch.setattr(bounded_snapshot.BoundedSnapshot, "scan", scan)
    assert refused("RESOURCE_UNAVAILABLE", lambda: build(fixture)) is primary
    assert len(sessions) == len(closed) == 3
    refused("CONTRACT_MISMATCH", lambda: sessions[0].resolve(fixture.snapshot.snapshot_id))
    assert not fixture.conn.in_transaction
    logged = capsys.readouterr()
    assert "synthetic-secret-not-for-logs" not in logged.err + logged.out


@pytest.mark.parametrize("kind", ["id", "cluster", "date_conflict", "day"])
@pytest.mark.parametrize("unconfirmed_first", [True, False])
def test_unconfirmed_sibling_conflicts_across_batches_and_partitions(source, kind, unconfirmed_first):
    first, last = row("a-first", 1999), row("z-last", 2025)
    if kind == "id":
        last["event_id"] = first["event_id"]
    elif kind == "cluster":
        first["event_cluster_id"] = last["event_cluster_id"] = "opaque-sibling"
    elif kind == "date_conflict":
        last["date_conflict"] = True
    else:
        last["event_date"] = first["event_date"]
        last["year"] = first["year"]
    (first if unconfirmed_first else last)["session"] = None
    groups = [("1999", [first, last])] if kind == "day" else [("1999", [first]), ("2025", [last])]
    fixture = source(groups, unrelated=False)
    refused("IDENTITY_CONFLICT", lambda: build(fixture, limits=dict(LIMITS, max_batch_rows=1)))


def test_equal_opaque_clusters_across_tickers_are_not_siblings(source):
    fixture = source([("2020", [row("a", event_cluster_id="opaque"),
        row("b", ticker="B", event_cluster_id="opaque")])], unrelated=False)
    result = build(fixture)
    assert result["counts"]["confirmed_rows"] == 2


def test_exact_snapshot_ref_required_and_mutation_detachment(source):
    fixture = source()
    expected = build(fixture)
    for invalid in (None, fixture.snapshot.snapshot_id,
                    replace(fixture.snapshot, calendar_version="other")):
        refused("SNAPSHOT_UNRESOLVED", lambda: index.build_index(fixture.holder, invalid,
            tickers=["A"], limits=dict(LIMITS)))
    copied = deepcopy(expected)
    copied["source"]["fragment_records"][0]["input_receipt_refs"].clear()
    assert build(fixture) == expected


def test_catalog_byte_boundary_covers_unrelated_membership(source, monkeypatch):
    from engine.v2.data._bounded_catalog import probe_catalog
    fixture = source()
    fixture.conn.execute("BEGIN")
    try:
        cap = probe_catalog(fixture.conn, fixture.snapshot.snapshot_id, TABLE,
            {key: LIMITS[key] for key in ("max_catalog_tables", "max_catalog_fragments",
                "max_catalog_bytes", "max_fragments", "max_partitions")}).stored_bytes
    finally:
        fixture.conn.rollback()
    assert build(fixture, limits=dict(LIMITS, max_catalog_bytes=cap)) == build(fixture)
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    refused("RESULT_LIMIT_EXCEEDED", lambda: build(fixture,
        limits=dict(LIMITS, max_catalog_bytes=cap - 1)))


@pytest.mark.parametrize("changes", [dict(event_date=datetime(2020, 1, 1, 0, 0, 1)),
    dict(session="PM"), dict(event_cluster_id=""), dict(event_id="")])
def test_selected_unconfirmed_metadata_validated_before_confirmation(source, changes):
    bad = row("a", src_orats=False)
    bad.update(changes)
    fixture = source([("2020", [bad])], unrelated=False)
    refused("CONTRACT_MISMATCH", lambda: build(fixture))


def test_utf8_retention_bytes_not_character_count(source):
    fixture = source([("2020", [row("café/事件", ticker="Δ")])], unrelated=False)
    expected = build(fixture, tickers=["Δ"])
    event = expected["sequences"][0]["events"][0]
    metadata = {key: event[key] for key in COLUMNS}
    encoded = len(canonical_json(metadata).encode("utf-8"))
    assert encoded > len(canonical_json(metadata))
    assert build(fixture, tickers=["Δ"],
        limits=dict(LIMITS, max_retained_metadata_bytes=encoded)) == expected
    refused("RESULT_LIMIT_EXCEEDED", lambda: build(fixture, tickers=["Δ"],
        limits=dict(LIMITS, max_retained_metadata_bytes=encoded - 1)))


def test_native_catalog_integrity_error_not_relabelled(source, monkeypatch):
    fixture = source()
    fixture.conn.execute("DROP TRIGGER data_objects_no_update")
    fixture.conn.execute("UPDATE data_objects SET byte_size=byte_size+1 WHERE object_id=?",
                         (fixture.other.object_ref.object_id,))
    fixture.conn.commit()
    catalog_limits = {key: LIMITS[key] for key in ("max_catalog_tables", "max_catalog_fragments",
        "max_catalog_bytes", "max_fragments", "max_partitions")}
    with pytest.raises(errors.DataError) as native:
        with bounded_snapshot.open_bounded_snapshot(fixture.holder, fixture.snapshot,
                table_name=TABLE, limits=catalog_limits):
            pytest.fail("corrupt unrelated identity admitted")
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    actual = refused(native.value.code, lambda: build(fixture))
    assert actual.problem == native.value.problem


def test_no_caller_admission_substitute(source):
    fixture = source()
    for holder in (None, SimpleNamespace(resolve=lambda _: fixture.snapshot),
                   {"snapshot": fixture.snapshot, "admitted": True}):
        refused("CONTRACT_MISMATCH", lambda: index.build_index(holder, fixture.snapshot,
            tickers=["A"], limits=dict(LIMITS)))
