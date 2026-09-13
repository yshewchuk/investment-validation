"""P2-6: snapshot-to-legacy materialization — phase-2 guide §9.1, §9.2; D13
(synthetic) and the adapter half of D14. Real SQLite catalog + ArtifactStore +
Parquet, no market data, no network.

Five synthetic tables (earnings_events, daily_market, trades, feature_panel,
tier4_forecasts) are committed into one real snapshot — ``daily_market``'s
2020 partition deliberately split across two fragments (tickers AAA/BBB) so
the round trip exercises a genuine multi-fragment partition, not just a
single-object one. ``build_materialization_request``/``materialize`` are then
run against that snapshot exactly as a caller (task 6b's stage, out of this
task's scope) would.
"""
from __future__ import annotations

import importlib
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import ObjectRef  # noqa: E402
from engine.v2.data import legacy_materialization as lm  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.legacy_adapter import materialize  # noqa: E402
from engine.v2.data.objects import partition_logical_hash  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    commit_tables,
    contract_for,
    publish_and_inspect,
    publish_bytes,
)

# --------------------------------------------------------------------------
# synthetic rows — one small, valid fixture per table in the read plan
# --------------------------------------------------------------------------

TABLES = ("earnings_events", "daily_market", "trades", "feature_panel", "tier4_forecasts")

_EE_COMMON = dict(src_orats=True, src_oquants=False, src_nasdaq=False, src_yfinance=False,
                  date_agree=True, date_conflict=False)
EE_ROWS = {
    "2020": [dict(event_id="EE1", ticker="AAA", event_date=datetime(2020, 1, 15), year=2020,
                  session="BMO", **_EE_COMMON)],
    "2021": [dict(event_id="EE2", ticker="BBB", event_date=datetime(2021, 2, 10), year=2021,
                  session="AMC", **_EE_COMMON)],
}

DM_ROWS = {
    ("2020", "AAA"): [dict(ticker="AAA", date=datetime(2020, 1, 2), year=2020, spot=100.0, iv10=0.3)],
    ("2020", "BBB"): [dict(ticker="BBB", date=datetime(2020, 1, 3), year=2020, spot=50.0, iv10=0.4)],
    ("2021", "AAA"): [dict(ticker="AAA", date=datetime(2021, 1, 4), year=2021, spot=110.0, iv10=0.25)],
}

TRADE_ROWS = {
    "2020": [dict(trade_id="T1", kind="option", strategy="S1", ticker="AAA", year=2020,
                  entry_date=datetime(2020, 1, 3), exit_date=datetime(2020, 1, 20),
                  provenance="engine.replay")],
    "2021": [dict(trade_id="T2", kind="option", strategy="S1", ticker="BBB", year=2021,
                  entry_date=datetime(2021, 2, 15), exit_date=datetime(2021, 3, 1),
                  provenance="engine.replay")],
}

PANEL_ROWS = [
    dict(ticker="AAA", k=1, date=datetime(2020, 1, 15), n_prior=5, mean_prior_move=0.01,
        mean_prior_abs_move=0.02, year=2020, signed_streak=1, ema12r_abs=0.03),
    dict(ticker="BBB", k=2, date=datetime(2021, 2, 10), n_prior=3, mean_prior_move=-0.01,
        mean_prior_abs_move=0.015, year=2021, signed_streak=-1, ema12r_abs=0.02),
]

TIER4_ROWS = [
    dict(ticker="AAA", event_date=datetime(2020, 1, 15), tier3_snapshot="snap1", pred_abs_move=0.05),
    dict(ticker="BBB", event_date=datetime(2021, 2, 10), tier3_snapshot="snap1", pred_abs_move=0.04),
]

DIRECT_SCOPE = {"tickers": ["AAA"], "years": [2020]}
EVIDENCE_SCOPE = {"tickers": ["AAA", "BBB"], "years": [2020, 2021]}


def _build_snapshot(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    contracts = {name: contract_for(name) for name in TABLES}

    records = {"earnings_events": [], "daily_market": [], "trades": [], "feature_panel": [],
               "tier4_forecasts": []}
    for year, rows in EE_ROWS.items():
        records["earnings_events"].append(
            publish_and_inspect(store, contracts["earnings_events"], _ref(contracts["earnings_events"]),
                               rows, partition_key=year))
    # daily_market's 2020 partition is deliberately two fragments (AAA, then
    # BBB — non-overlapping since ticker is the leading primary-key column),
    # in ascending key order, so the round trip exercises a genuine
    # multi-fragment partition rather than a single-object one.
    for (year, _ticker), rows in sorted(DM_ROWS.items()):
        records["daily_market"].append(
            publish_and_inspect(store, contracts["daily_market"], _ref(contracts["daily_market"]),
                               rows, partition_key=year))
    for year, rows in TRADE_ROWS.items():
        records["trades"].append(
            publish_and_inspect(store, contracts["trades"], _ref(contracts["trades"]),
                               rows, partition_key=year))
    records["feature_panel"].append(
        publish_and_inspect(store, contracts["feature_panel"], _ref(contracts["feature_panel"]),
                           PANEL_ROWS, partition_key="all"))
    records["tier4_forecasts"].append(
        publish_and_inspect(store, contracts["tier4_forecasts"], _ref(contracts["tier4_forecasts"]),
                           TIER4_ROWS, partition_key="all"))

    dm_2020 = [r for r in records["daily_market"] if r.partition_key == "2020"]
    dm_2020_hash = partition_logical_hash(
        store, [r.object_ref for r in dm_2020], contracts["daily_market"],
        _ref(contracts["daily_market"]), "2020")

    snap = commit_tables(conn, clock, records, contracts, store=store,
                         partition_logical_hashes={"daily_market": {"2020": dm_2020_hash}})
    return conn, store, snap


def _ref(contract):
    from engine.v2.contracts.data import TableContractRef
    return TableContractRef(contract_id=contract.contract_id, definition_hash=contract.definition_hash)


def _snapshot_object_ref(store) -> ObjectRef:
    return publish_bytes(store, b'{"schema_version": "legacy_snapshot.v1", "tickers": ["AAA", "BBB"]}')


def _pinned_refs(store):
    registry_hash = store.publish_bytes(b'{"models": []}', schema_ref="legacy_pinned_ref.v1").content_hash
    calendar_hash = store.publish_bytes(b"date\n2020-01-02\n2021-01-04\n",
                                        schema_ref="legacy_pinned_ref.v1").content_hash
    registry_refs = (lm.format_pinned_ref("engine/models/registry.json", registry_hash),)
    calendar_refs = (lm.format_pinned_ref("data/raw/polygon/gspc_daily.csv", calendar_hash),)
    return registry_refs, calendar_refs


def _build_request(repository, snap, snapshot_object_ref, store, *, direct_scope=None, evidence_scope=None):
    registry_refs, calendar_refs = _pinned_refs(store)
    return lm.build_materialization_request(
        repository, snap, snapshot_object_ref,
        direct_scope=direct_scope or DIRECT_SCOPE, evidence_scope=evidence_scope or EVIDENCE_SCOPE,
        registry_and_model_refs=registry_refs, calendar_refs=calendar_refs,
        expected_population={"earnings_events": 2, "daily_market": 3, "trades": 2,
                             "feature_panel": 2, "tier4_forecasts": 2})


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_round_trip_through_legacy_readers(tmp_path, monkeypatch):
    conn, store, snap = _build_snapshot(tmp_path)
    snapshot_object_ref = _snapshot_object_ref(store)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, snapshot_object_ref, store)
    assert lm.read_plan_complete(request)

    dest_root = tmp_path / "legacy_root"
    manifest = materialize(repository, store, request, dest_root)
    assert manifest  # every declared path got a manifest entry

    import engine.paths as legacy_paths
    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(dest_root))
    importlib.reload(legacy_paths)
    try:
        from engine.data import store as legacy_store

        panel = legacy_store._read_part(legacy_paths.PANEL, columns=None)
        assert sorted(panel["ticker"].tolist()) == ["AAA", "BBB"]
        tier4 = legacy_store._read_part(legacy_paths.TIER4, columns=None)
        assert sorted(tier4["ticker"].tolist()) == ["AAA", "BBB"]

        earnings = legacy_store.read_table("earnings_events")
        assert sorted(earnings["event_id"].tolist()) == ["EE1", "EE2"]

        daily = legacy_store.read_table("daily_market")
        assert sorted(daily["ticker"].tolist()) == ["AAA", "AAA", "BBB"]
        assert legacy_store.table_years("daily_market") == [2020, 2021]

        trades = legacy_store.read_table("trades")
        assert sorted(trades["trade_id"].tolist()) == ["T1", "T2"]

        source_bytes = store.read_verified(_as_artifact_ref(store, snapshot_object_ref))
        assert (dest_root / "data" / "features" / "SNAPSHOT").read_bytes() == source_bytes
    finally:
        monkeypatch.delenv("INVESTING_PLAN_ROOT", raising=False)
        importlib.reload(legacy_paths)


def _as_artifact_ref(store, object_ref: ObjectRef):
    from engine.v2.contracts import ArtifactRef
    from engine.v2.foundation import CONTENT_HASH_PREFIX
    digest = object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    return ArtifactRef(artifact_id=object_ref.object_id, content_hash=object_ref.content_hash,
                       schema_ref="parquet_fragment.v1", byte_size=object_ref.byte_size,
                       storage_key=f"objects/{digest[:2]}/{digest}")


# --------------------------------------------------------------------------
# D13: private, bounded, identity-sensitive
# --------------------------------------------------------------------------


def test_tree_contains_only_declared_paths_and_is_read_only(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    dest_root = tmp_path / "legacy_root"
    manifest = materialize(repository, store, request, dest_root)

    on_disk = {str(p.relative_to(dest_root)) for p in dest_root.rglob("*") if p.is_file()}
    assert on_disk == set(manifest)

    for rel in manifest:
        path = dest_root / rel
        assert not path.is_symlink()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o444
        if os.geteuid() != 0:  # root bypasses file permission bits entirely
            with pytest.raises(PermissionError):
                path.write_bytes(b"x")
    for d in [p for p in dest_root.rglob("*") if p.is_dir()] + [dest_root]:
        assert stat.S_IMODE(d.stat().st_mode) == 0o555


def test_identical_request_yields_identical_hash_and_manifest(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    snapshot_object_ref = _snapshot_object_ref(store)
    registry_refs, calendar_refs = _pinned_refs(store)
    request_a = lm.build_materialization_request(
        repository, snap, snapshot_object_ref, direct_scope=DIRECT_SCOPE, evidence_scope=EVIDENCE_SCOPE,
        registry_and_model_refs=registry_refs, calendar_refs=calendar_refs, expected_population={})
    request_b = lm.build_materialization_request(
        repository, snap, snapshot_object_ref, direct_scope=DIRECT_SCOPE, evidence_scope=EVIDENCE_SCOPE,
        registry_and_model_refs=registry_refs, calendar_refs=calendar_refs, expected_population={})
    assert request_a.request_hash == request_b.request_hash

    manifest_a = materialize(repository, store, request_a, tmp_path / "root_a")
    manifest_b = materialize(repository, store, request_b, tmp_path / "root_b")
    assert manifest_a == manifest_b


def test_changing_scope_or_refs_changes_request_hash(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    snapshot_object_ref = _snapshot_object_ref(store)
    base = _build_request(repository, snap, snapshot_object_ref, store)

    wider_tickers = _build_request(repository, snap, snapshot_object_ref, store,
                                   evidence_scope={"tickers": ["AAA", "BBB", "CCC"], "years": [2020, 2021]})
    assert wider_tickers.request_hash != base.request_hash

    wider_years = _build_request(repository, snap, snapshot_object_ref, store,
                                 evidence_scope={"tickers": ["AAA", "BBB"], "years": [2019, 2020, 2021]})
    assert wider_years.request_hash != base.request_hash

    registry_refs, calendar_refs = _pinned_refs(store)
    other_registry = (lm.format_pinned_ref("engine/models/registry.json",
                                           store.publish_bytes(b'{"models": [1]}',
                                                              schema_ref="legacy_pinned_ref.v1").content_hash),)
    changed_model_ref = lm.build_materialization_request(
        repository, snap, snapshot_object_ref, direct_scope=DIRECT_SCOPE, evidence_scope=EVIDENCE_SCOPE,
        registry_and_model_refs=other_registry, calendar_refs=calendar_refs, expected_population={})
    assert changed_model_ref.request_hash != base.request_hash

    other_calendar = (lm.format_pinned_ref("data/raw/polygon/gspc_daily.csv",
                                           store.publish_bytes(b"date\n2099-01-01\n",
                                                              schema_ref="legacy_pinned_ref.v1").content_hash),)
    changed_calendar_ref = lm.build_materialization_request(
        repository, snap, snapshot_object_ref, direct_scope=DIRECT_SCOPE, evidence_scope=EVIDENCE_SCOPE,
        registry_and_model_refs=registry_refs, calendar_refs=other_calendar, expected_population={})
    assert changed_calendar_ref.request_hash != base.request_hash

    other_snapshot_object_ref = _snapshot_object_ref(store)
    changed_snapshot_object = lm.build_materialization_request(
        repository, snap, other_snapshot_object_ref, direct_scope=DIRECT_SCOPE,
        evidence_scope=EVIDENCE_SCOPE, registry_and_model_refs=registry_refs,
        calendar_refs=calendar_refs, expected_population={})
    assert changed_snapshot_object.request_hash != base.request_hash


def test_changing_a_column_set_changes_request_hash(tmp_path):
    import dataclasses

    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    narrowed = dict(request.table_queries)
    original = narrowed["earnings_events"]
    narrowed["earnings_events"] = dataclasses.replace(
        original, columns=tuple(c for c in original.columns if c != "session"))
    changed = dataclasses.replace(request, table_queries=narrowed)
    doc_a, doc_b = _doc_without_hash(request), _doc_without_hash(changed)
    assert doc_a != doc_b


def _doc_without_hash(request):
    from engine.v2.foundation import to_document
    doc = to_document(request)
    del doc["request_hash"]
    return doc


# --------------------------------------------------------------------------
# evidence scope
# --------------------------------------------------------------------------


def test_evidence_scope_widens_the_analog_pool_table_scan(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store,
                             direct_scope={"tickers": ["AAA"], "years": [2020]},
                             evidence_scope={"tickers": ["AAA", "BBB"], "years": [2020, 2021]})
    daily_query = request.table_queries["daily_market"]
    ticker_predicate = next(p for p in daily_query.key_filter if p.column == "ticker")
    assert set(ticker_predicate.values) == {"AAA", "BBB"}
    assert lm.read_plan_complete(request)


def test_read_plan_incomplete_when_a_table_query_or_ref_is_missing(tmp_path):
    import dataclasses

    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)

    missing_table = dataclasses.replace(
        request, table_queries={k: v for k, v in request.table_queries.items() if k != "trades"})
    assert not lm.read_plan_complete(missing_table)

    no_registry_refs = dataclasses.replace(request, registry_and_model_refs=())
    assert not lm.read_plan_complete(no_registry_refs)

    no_calendar_refs = dataclasses.replace(request, calendar_refs=())
    assert not lm.read_plan_complete(no_calendar_refs)


def test_narrower_evidence_than_direct_scope_is_incomplete(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store,
                             direct_scope={"tickers": ["AAA", "BBB"], "years": [2020, 2021]},
                             evidence_scope={"tickers": ["AAA"], "years": [2020]})
    assert not lm.read_plan_complete(request)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_refuses_a_non_empty_dest_root(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    dest_root = tmp_path / "occupied"
    dest_root.mkdir()
    (dest_root / "stray.txt").write_text("pre-existing")
    with pytest.raises(DataError) as err:
        materialize(repository, store, request, dest_root)
    assert err.value.code == "DEST_ROOT_NOT_EMPTY"


def test_refuses_a_dest_root_inside_the_object_store(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    dest_root = store.root / "attempts" / "inside-store"
    with pytest.raises(DataError) as err:
        materialize(repository, store, request, dest_root)
    assert err.value.code == "DEST_ROOT_UNSAFE"


def test_refuses_a_symlinked_dest_root(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    link = tmp_path / "linked_root"
    link.symlink_to(real_dir)
    with pytest.raises(DataError) as err:
        materialize(repository, store, request, link)
    assert err.value.code == "DEST_ROOT_UNSAFE"


def test_query_exceeding_contract_limits_propagates_the_repository_code(tmp_path):
    import dataclasses

    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    oversized = dict(request.table_queries)
    original = oversized["trades"]
    oversized["trades"] = dataclasses.replace(
        original, max_batch_rows=60_000, max_result_rows=3_000_000)
    bad_request = dataclasses.replace(request, table_queries=oversized)
    with pytest.raises(DataError) as err:
        materialize(repository, store, bad_request, tmp_path / "root_over")
    assert err.value.code == "QUERY_NOT_BOUNDED"


# --------------------------------------------------------------------------
# pinned-ref formatting and other refusal edges
# --------------------------------------------------------------------------


def test_format_and_parse_pinned_ref_round_trip():
    ref = lm.format_pinned_ref("engine/models/registry.json", "sha256:" + "ab" * 32)
    assert lm.parse_pinned_ref(ref) == ("engine/models/registry.json", "sha256:" + "ab" * 32)


def test_format_pinned_ref_rejects_a_separator_in_the_path():
    with pytest.raises(ValueError):
        lm.format_pinned_ref("a::b", "sha256:" + "00" * 32)


def test_parse_pinned_ref_rejects_malformed_strings():
    with pytest.raises(ValueError):
        lm.parse_pinned_ref("no-separator-here")
    with pytest.raises(ValueError):
        lm.parse_pinned_ref("::sha256:" + "00" * 32)
    with pytest.raises(ValueError):
        lm.parse_pinned_ref("engine/models/registry.json::")


def test_refuses_a_dest_root_that_is_a_regular_file(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    dest_root = tmp_path / "not_a_dir"
    dest_root.write_text("i am a file")
    with pytest.raises(DataError) as err:
        materialize(repository, store, request, dest_root)
    assert err.value.code == "DEST_ROOT_UNSAFE"


def test_coerce_failure_surfaces_as_contract_mismatch():
    import pandas as pd

    from engine.v2.data.legacy_adapter import _assert_legacy_coerce_accepts

    bad_frame = pd.DataFrame({"only_column": [1, 2]})  # missing every required trades column
    with pytest.raises(DataError) as err:
        _assert_legacy_coerce_accepts(bad_frame, "trades")
    assert err.value.code == "CONTRACT_MISMATCH"
