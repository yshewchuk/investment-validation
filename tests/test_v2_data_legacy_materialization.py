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

import dataclasses
import importlib
import os
import stat
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import KeyPredicate, ObjectRef, TimeInterval  # noqa: E402
from engine.v2.data import legacy_materialization as lm  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.legacy_adapter import materialize  # noqa: E402
from engine.v2.data.objects import partition_logical_hash  # noqa: E402
from engine.v2.data.reference_inputs import LEGACY_REFERENCE_INPUTS_V1, TIER4_SERVING_DIR  # noqa: E402
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

TABLES = ("earnings_events", "daily_market", "trades", "option_chains", "feature_panel",
         "tier4_forecasts")

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

CHAIN_ROWS = {
    "2020": [dict(ticker="AAA", obs_date=datetime(2020, 1, 3), year=2020, expiry=datetime(2020, 1, 17),
                 dte=14, strike=100.0, right="C", bid=1.2, ask=1.4, delta=0.5, spot=100.0,
                 quote_repaired=False)],
    "2021": [dict(ticker="BBB", obs_date=datetime(2021, 2, 12), year=2021, expiry=datetime(2021, 2, 26),
                 dte=14, strike=50.0, right="P", bid=0.8, ask=1.0, delta=-0.4, spot=50.0,
                 quote_repaired=False)],
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
#: A test-only maximal cutoff (SEND-BACK 2026-09-14 item 2:
#: ``build_materialization_request`` now requires the job's own real
#: ``observation_ceiling`` -- see ``snapshot_planning.pin_snapshot_inputs``
#: for how a real caller derives one from its session date; these tests only
#: need SOME valid cutoff, not a specific one).
FAR_FUTURE_CEILING = "9999-12-31T23:59:59.000000Z"


def _build_snapshot(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    contracts = {name: contract_for(name) for name in TABLES}

    records = {"earnings_events": [], "daily_market": [], "trades": [], "option_chains": [],
               "feature_panel": [], "tier4_forecasts": []}
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
    for year, rows in CHAIN_ROWS.items():
        records["option_chains"].append(
            publish_and_inspect(store, contracts["option_chains"], _ref(contracts["option_chains"]),
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


CALENDAR_PATH = LEGACY_REFERENCE_INPUTS_V1["inputs"]["calendar"]["path"]
PNL_SIM_HISTORY_PATH = LEGACY_REFERENCE_INPUTS_V1["inputs"]["pnl_sim_history"]["path"]
RECALIBRATION_PAIRS_PATH = LEGACY_REFERENCE_INPUTS_V1["inputs"]["recalibration_pairs"]["path"]


def _pinned_refs(store):
    registry_hash = store.publish_bytes(b'{"models": []}', schema_ref="legacy_pinned_ref.v1").content_hash
    calendar_hash = store.publish_bytes(b"date\n2020-01-02\n2021-01-04\n",
                                        schema_ref="legacy_pinned_ref.v1").content_hash
    registry_refs = (lm.format_pinned_ref("engine/models/registry.json", registry_hash),)
    calendar_refs = (lm.format_pinned_ref(CALENDAR_PATH, calendar_hash),)
    return registry_refs, calendar_refs


def _build_request(repository, snap, snapshot_object_ref, store, *, direct_scope=None, evidence_scope=None,
                   extra_registry_refs=(), observation_ceiling=FAR_FUTURE_CEILING):
    """``extra_registry_refs`` (task brief 2026-09-14): callers outside this
    module that need the pnl_sim_history/recalibration_pairs refs too (e.g.
    ``tests/test_v2_ops_snapshot_stages.py::Case``, whose ``pin_snapshot_inputs``
    now requires them pinned) pass them here rather than duplicating this
    function's whole body. ``observation_ceiling`` (SEND-BACK 2026-09-14 item
    2) defaults to :data:`FAR_FUTURE_CEILING`; a caller comparing against a
    REAL ``pin_snapshot_inputs`` call must pass the SAME
    ``f"{session}T23:59:59.000000Z"`` that produces, since it is now part of
    the hashed request."""
    registry_refs, calendar_refs = _pinned_refs(store)
    # Sorted by path, matching reference_catalog.pinned_materialization_refs's
    # own sort -- request_hash is order-sensitive (build_materialization_request
    # stores registry_and_model_refs verbatim, no internal sort).
    combined = tuple(sorted(registry_refs + tuple(extra_registry_refs),
                            key=lambda ref: lm.parse_pinned_ref(ref)[0]))
    return lm.build_materialization_request(
        repository, store, snap, snapshot_object_ref,
        direct_scope=direct_scope or DIRECT_SCOPE, evidence_scope=evidence_scope or EVIDENCE_SCOPE,
        registry_and_model_refs=combined, calendar_refs=calendar_refs,
        expected_population={"earnings_events": 2, "daily_market": 3, "trades": 2,
                             "option_chains": 2, "feature_panel": 2, "tier4_forecasts": 2},
        observation_ceiling=observation_ceiling)


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_round_trip_through_legacy_readers(tmp_path, monkeypatch):
    conn, store, snap = _build_snapshot(tmp_path)
    snapshot_object_ref = _snapshot_object_ref(store)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, snapshot_object_ref, store)
    assert lm.read_plan_complete(request, repository)

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
        repository, store, snap, snapshot_object_ref, direct_scope=DIRECT_SCOPE, evidence_scope=EVIDENCE_SCOPE,
        registry_and_model_refs=registry_refs, calendar_refs=calendar_refs, expected_population={}, observation_ceiling=FAR_FUTURE_CEILING)
    request_b = lm.build_materialization_request(
        repository, store, snap, snapshot_object_ref, direct_scope=DIRECT_SCOPE, evidence_scope=EVIDENCE_SCOPE,
        registry_and_model_refs=registry_refs, calendar_refs=calendar_refs, expected_population={}, observation_ceiling=FAR_FUTURE_CEILING)
    assert request_a.request_hash == request_b.request_hash

    manifest_a = materialize(repository, store, request_a, tmp_path / "root_a")
    manifest_b = materialize(repository, store, request_b, tmp_path / "root_b")
    assert manifest_a == manifest_b


def test_model_output_reference_inputs_materialize_at_their_legacy_paths(tmp_path):
    """Task brief 2026-09-14: pnl_sim_history.parquet/recalibration_pairs.parquet
    pin and materialize exactly like every other ``registry_and_model_refs``
    entry -- through the SAME generic copy loop
    ``legacy_materialization.materialize_tree`` already uses (the
    ``for ref in (*request.registry_and_model_refs, *request.calendar_refs)``
    write, unmodified for this task), so pinning them as two more entries is
    enough; no new materialize-time code is needed."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    registry_refs, calendar_refs = _pinned_refs(store)
    pnl_sim_bytes = b"synthetic pnl_sim_history parquet bytes"
    recal_bytes = b"synthetic recalibration_pairs parquet bytes"
    pnl_sim_hash = store.publish_bytes(pnl_sim_bytes, schema_ref="legacy_pinned_ref.v1").content_hash
    recal_hash = store.publish_bytes(recal_bytes, schema_ref="legacy_pinned_ref.v1").content_hash
    model_output_refs = (
        lm.format_pinned_ref(PNL_SIM_HISTORY_PATH, pnl_sim_hash),
        lm.format_pinned_ref(RECALIBRATION_PAIRS_PATH, recal_hash),
    )
    request = lm.build_materialization_request(
        repository, store, snap, _snapshot_object_ref(store),
        direct_scope=DIRECT_SCOPE, evidence_scope=EVIDENCE_SCOPE,
        registry_and_model_refs=registry_refs + model_output_refs, calendar_refs=calendar_refs,
        expected_population={}, observation_ceiling=FAR_FUTURE_CEILING)

    dest_root = tmp_path / "legacy_root"
    manifest = materialize(repository, store, request, dest_root)

    assert manifest[PNL_SIM_HISTORY_PATH] == pnl_sim_hash
    assert manifest[RECALIBRATION_PAIRS_PATH] == recal_hash
    assert (dest_root / PNL_SIM_HISTORY_PATH).read_bytes() == pnl_sim_bytes
    assert (dest_root / RECALIBRATION_PAIRS_PATH).read_bytes() == recal_bytes


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
        repository, store, snap, snapshot_object_ref, direct_scope=DIRECT_SCOPE, evidence_scope=EVIDENCE_SCOPE,
        registry_and_model_refs=other_registry, calendar_refs=calendar_refs, expected_population={}, observation_ceiling=FAR_FUTURE_CEILING)
    assert changed_model_ref.request_hash != base.request_hash

    other_calendar = (lm.format_pinned_ref(CALENDAR_PATH,
                                           store.publish_bytes(b"date\n2099-01-01\n",
                                                              schema_ref="legacy_pinned_ref.v1").content_hash),)
    changed_calendar_ref = lm.build_materialization_request(
        repository, store, snap, snapshot_object_ref, direct_scope=DIRECT_SCOPE, evidence_scope=EVIDENCE_SCOPE,
        registry_and_model_refs=registry_refs, calendar_refs=other_calendar, expected_population={}, observation_ceiling=FAR_FUTURE_CEILING)
    assert changed_calendar_ref.request_hash != base.request_hash

    other_snapshot_object_ref = _snapshot_object_ref(store)
    changed_snapshot_object = lm.build_materialization_request(
        repository, store, snap, other_snapshot_object_ref, direct_scope=DIRECT_SCOPE,
        evidence_scope=EVIDENCE_SCOPE, registry_and_model_refs=registry_refs,
        calendar_refs=calendar_refs, expected_population={}, observation_ceiling=FAR_FUTURE_CEILING)
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
    # daily_market is whole_table as of review round 3 (item 2); option_chains
    # is the remaining evidence-scoped table, so it demonstrates the widening.
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store,
                             direct_scope={"tickers": ["AAA"], "years": [2020]},
                             evidence_scope={"tickers": ["AAA", "BBB"], "years": [2020, 2021]})
    chains_query = request.table_queries["option_chains"]
    ticker_predicate = next(p for p in chains_query.key_filter if p.column == "ticker")
    assert set(ticker_predicate.values) == {"AAA", "BBB"}
    assert request.table_queries["daily_market"].key_filter == ()
    assert lm.read_plan_complete(request, repository)


def test_read_plan_incomplete_when_a_table_query_or_ref_is_missing(tmp_path):
    import dataclasses

    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)

    missing_table = dataclasses.replace(
        request, table_queries={k: v for k, v in request.table_queries.items() if k != "trades"})
    assert not lm.read_plan_complete(missing_table, repository)

    no_registry_refs = dataclasses.replace(request, registry_and_model_refs=())
    assert not lm.read_plan_complete(no_registry_refs, repository)

    no_calendar_refs = dataclasses.replace(request, calendar_refs=())
    assert not lm.read_plan_complete(no_calendar_refs, repository)


def test_narrower_evidence_than_direct_scope_is_incomplete(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store,
                             direct_scope={"tickers": ["AAA", "BBB"], "years": [2020, 2021]},
                             evidence_scope={"tickers": ["AAA"], "years": [2020]})
    assert not lm.read_plan_complete(request, repository)


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
    # A narrower projection (drop one column) also breaks whole-table-copy
    # eligibility (task brief fix: daily_market's coverage check no longer
    # scans trades unconditionally, so trades must actually take the
    # scan-and-rewrite path itself for Repository.scan's own contract-limit
    # validation to run and raise QUERY_NOT_BOUNDED).
    oversized["trades"] = dataclasses.replace(
        original, columns=tuple(c for c in original.columns if c != "provenance"),
        max_batch_rows=60_000, max_result_rows=3_000_000)
    bad_request = dataclasses.replace(request, table_queries=oversized)
    root_over = tmp_path / "root_over"
    with pytest.raises(DataError) as err:
        materialize(repository, store, bad_request, root_over)
    assert err.value.code == "QUERY_NOT_BOUNDED"
    assert not root_over.exists()  # cleanup-on-failure: refused before any table was written


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


# --------------------------------------------------------------------------
# review round 2: option_chains, sentinel-free intervals, trades span proof,
# pinned-ref existence, ledger exactness
# --------------------------------------------------------------------------


def test_materialized_chains_readable_by_legacy_load_chain_index(tmp_path, monkeypatch):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    dest_root = tmp_path / "legacy_root"
    materialize(repository, store, request, dest_root)

    import engine.paths as legacy_paths
    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(dest_root))
    importlib.reload(legacy_paths)
    try:
        import pandas as pd

        from engine.replay import load_chain_index

        index = load_chain_index([
            ("AAA", pd.Timestamp(2020, 1, 3)), ("BBB", pd.Timestamp(2021, 2, 12)),
        ])
        aaa = index.get("AAA", pd.Timestamp(2020, 1, 3))
        assert aaa is not None and not aaa.empty
        assert aaa["strike"].tolist() == [100.0]
        bbb = index.get("BBB", pd.Timestamp(2021, 2, 12))
        assert bbb is not None and not bbb.empty
        assert bbb["strike"].tolist() == [50.0]
    finally:
        monkeypatch.delenv("INVESTING_PLAN_ROOT", raising=False)
        importlib.reload(legacy_paths)


def test_whole_table_interval_has_no_sentinel_bound(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)

    interval = request.table_queries["earnings_events"].time_interval
    assert interval is not None
    assert interval.start_inclusive not in ("1900-01-01", "1900-01-01T00:00:00.000000")
    assert interval.end_exclusive not in ("2999-12-31", "2999-12-31T00:00:00.000000")
    # Derived exactly from EE_ROWS's own event_date values: min time_min is
    # 2020's, max time_max is 2021's, end made exclusive by one microsecond.
    assert interval.start_inclusive == "2020-01-15T00:00:00.000000"
    assert interval.end_exclusive == "2021-02-10T00:00:00.000001"


def test_trades_span_outside_evidence_scope_is_accepted_when_daily_market_is_whole_table(tmp_path):
    """Task brief real-data defect: ``daily_market`` is whole_table by
    construction (LEGACY_SCORE_READ_PLAN_V1), so trades's real span is
    covered no matter how narrow ``evidence_scope`` is -- a bounded board
    universe (the real failure: 201 tickers, 2023-2026) can never contain the
    real committed trades table's full span (2,724 tickers, 2018-2026), and
    it must not have to. The old behavior -- refusing here -- was the bug;
    this replaces the old negative control of the same name."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    # trades carries BBB (see TRADE_ROWS); an evidence_scope missing it no
    # longer under-covers Scorer._entry_implied_move's own daily_market read,
    # because that read is a whole-table copy regardless of evidence_scope.
    narrow_evidence = {"tickers": ["AAA"], "years": [2020, 2021]}
    request = _build_request(repository, snap, _snapshot_object_ref(store), store,
                             direct_scope={"tickers": ["AAA"], "years": [2020]},
                             evidence_scope=narrow_evidence)
    assert request.table_queries["daily_market"].key_filter == ()  # whole-table: no ticker predicate
    assert lm.evidence_scope_covers_trades(repository, request)
    assert lm.read_plan_complete(request, repository)
    dest_root = tmp_path / "legacy_root"
    manifest = materialize(repository, store, request, dest_root)
    assert manifest


def test_trades_span_inside_evidence_scope_is_accepted(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    assert lm.evidence_scope_covers_trades(repository, request)
    assert lm.trades_span(repository, request) == {"tickers": {"AAA", "BBB"}, "years": {2020, 2021}}


def test_row_scoped_daily_market_excluding_a_trades_ticker_is_refused(tmp_path):
    """Task brief decision, second half: when ``daily_market`` is NOT a
    whole-table read (hypothetically -- today's plan never row-scopes it),
    ``trades``'s real span must be checked against THAT query's own ticker
    predicate/time bound, not the abstract evidence_scope. A synthetic ticker
    predicate excluding a real trades ticker (BBB) must be refused, and the
    refusal must name a COUNT, never the ticker value itself
    (engine.v2.data.errors: no row value may reach a message)."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    whole = request.table_queries["daily_market"]
    # Full calendar-year coverage on the time bound (matching _scope_bounds's
    # own shape for a row-scoped read) isolates the ticker predicate as the
    # only uncovered dimension.
    row_scoped = dataclasses.replace(
        whole, key_filter=(KeyPredicate(column="ticker", operator="in", values=("AAA",)),),
        time_interval=TimeInterval(column=whole.time_interval.column,
                                   start_inclusive="2020-01-01T00:00:00.000000",
                                   end_exclusive="2022-01-01T00:00:00.000000"))
    request = dataclasses.replace(request, table_queries={**request.table_queries, "daily_market": row_scoped})

    assert not lm.evidence_scope_covers_trades(repository, request)
    assert not lm.read_plan_complete(request, repository)
    dest_root = tmp_path / "legacy_root"
    with pytest.raises(DataError) as err:
        materialize(repository, store, request, dest_root)
    assert err.value.code == "EVIDENCE_SCOPE_INCOMPLETE"
    assert not dest_root.exists()  # cleanup-on-failure: refused before any table was written
    assert err.value.problem.details == {"uncovered_ticker_count": 1, "uncovered_year_count": 0}
    assert "BBB" not in err.value.problem.message  # a count, never the value (§7.2 redaction)


def test_row_scoped_daily_market_narrower_years_is_refused_by_count(tmp_path):
    """Same decision, the time-bound half: a row-scoped ``daily_market`` query
    whose own interval excludes a real trades year is refused with a year
    count, independent of the ticker predicate (left wide open here)."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    whole = request.table_queries["daily_market"]
    row_scoped = dataclasses.replace(
        whole,
        key_filter=(KeyPredicate(column="ticker", operator="in", values=("AAA", "BBB")),),
        time_interval=TimeInterval(column=whole.time_interval.column,
                                   start_inclusive="2020-01-01T00:00:00.000000",
                                   end_exclusive="2021-01-01T00:00:00.000000"))
    request = dataclasses.replace(request, table_queries={**request.table_queries, "daily_market": row_scoped})

    assert not lm.evidence_scope_covers_trades(repository, request)
    with pytest.raises(DataError) as err:
        lm.materialize_tree(repository, store, request, tmp_path / "legacy_root2")
    assert err.value.code == "EVIDENCE_SCOPE_INCOMPLETE"
    assert err.value.problem.details == {"uncovered_ticker_count": 0, "uncovered_year_count": 1}


def test_missing_pinned_ref_is_refused(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    _, calendar_refs = _pinned_refs(store)
    never_published = lm.format_pinned_ref("engine/models/registry.json", "sha256:" + "ee" * 32)
    with pytest.raises(DataError) as err:
        lm.build_materialization_request(
            repository, store, snap, _snapshot_object_ref(store), direct_scope=DIRECT_SCOPE,
            evidence_scope=EVIDENCE_SCOPE, registry_and_model_refs=(never_published,),
            calendar_refs=calendar_refs, expected_population={}, observation_ceiling=FAR_FUTURE_CEILING)
    assert err.value.code == "OBJECT_CORRUPT"


def test_pinned_ref_with_wrong_hash_is_refused(tmp_path):
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    registry_refs, calendar_refs = _pinned_refs(store)
    path, _real_hash = lm.parse_pinned_ref(registry_refs[0])
    tampered = lm.format_pinned_ref(path, "sha256:" + "11" * 32)
    with pytest.raises(DataError) as err:
        lm.build_materialization_request(
            repository, store, snap, _snapshot_object_ref(store), direct_scope=DIRECT_SCOPE,
            evidence_scope=EVIDENCE_SCOPE, registry_and_model_refs=(tampered,),
            calendar_refs=calendar_refs, expected_population={}, observation_ceiling=FAR_FUTURE_CEILING)
    assert err.value.code == "OBJECT_CORRUPT"


def test_legacy_adapter_ledger_has_one_entry_per_exact_symbol():
    import json

    ledger = json.loads((ROOT / "checks" / "legacy_adapters.json").read_text())
    adapters = ledger["adapters"]
    assert ledger["count"] == len(adapters)
    pairs = [(a["module"], a["legacy_symbol"]) for a in adapters]
    assert len(pairs) == len(set(pairs)), "duplicate (module, legacy_symbol) ledger entries"
    data_module_symbols = [a["legacy_symbol"] for a in adapters
                           if a["module"] == "engine.v2.data.legacy_adapter"]
    assert "engine.data.features.panel.PANEL_COLUMNS" in data_module_symbols
    assert "engine.data.features.tier4.COLUMNS" in data_module_symbols
    assert "engine.data.features.tier4.KEY_COLUMNS" in data_module_symbols
    assert "engine.data.store._read_part" in data_module_symbols
    assert "engine.data.schemas.coerce" in data_module_symbols
    assert "engine.data.schemas.SOURCE_PRIORITY" in data_module_symbols


def test_daily_market_is_whole_table_covering_the_tier4_cache_miss_window(tmp_path):
    """Review round 3, item 2: daily_market must not be scoped narrowly
    enough that a tier4 serving-model cache miss (im_t1/runup_move's fixed
    IM_T1_YEARS window, or iv_crush's fully unbounded read) could silently
    under-read. Chosen resolution: whole_table, not a pinned-artifact refusal
    (option (a) is not provable — see the module docstring). Direct_scope is
    named narrower than evidence_scope here specifically to prove daily_market
    reads every committed row regardless of EITHER — not merely whatever
    tickers/years this test happened to hand it."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store,
                             direct_scope={"tickers": ["AAA"], "years": [2020]})
    daily_query = request.table_queries["daily_market"]
    assert daily_query.key_filter == ()  # no ticker restriction at all
    scanned = lm.scanned_rows(repository, daily_query, "daily_market")
    # All three committed daily_market rows come back regardless of scope —
    # the query itself, not a trusted scope, is what is complete here.
    assert {row["ticker"] for row in scanned} == {"AAA", "BBB"}
    assert {row["year"] for row in scanned} == {2020, 2021}
    assert lm.read_plan_complete(request, repository)


def test_narrowed_whole_table_query_never_silently_copies_the_whole_table(tmp_path):
    """Review P2-C05, decision 2's own proof: a one-row request (a real
    predicate plus a time_interval matching exactly one of daily_market's
    three manifest rows, ceiling 1) against a table the static read plan
    classifies whole_table must materialize exactly that one row. Before
    the fix, ``materialize_tree`` decided copy-vs-scan from the static plan
    alone (``table_name not in _EVIDENCE_SCOPED_TABLES``), so this narrowed
    query would still take the byte-copy path and silently write all three
    rows, ignoring both its own predicate and its ceiling of 1."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    narrow_query = dataclasses.replace(
        request.table_queries["daily_market"],
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAA",)),),
        time_interval=TimeInterval(column="date", start_inclusive="2020-01-01",
                                   end_exclusive="2020-01-03"),
        max_result_rows=1, max_batch_rows=1)
    # This narrowed daily_market read is deliberately unrelated to trades's
    # real span (it exists only to test row-scoped write mechanics below), so
    # trades is also narrowed to match nothing -- decoupling it from the task
    # brief's daily_market-covers-trades gate, which this test does not exercise.
    no_trades_match = dataclasses.replace(
        request.table_queries["trades"],
        key_filter=(KeyPredicate(column="ticker", operator="in", values=("ZZZ",)),))
    request = dataclasses.replace(
        request, table_queries={**request.table_queries, "daily_market": narrow_query,
                                "trades": no_trades_match})
    dest_root = tmp_path / "legacy_root"
    tree = lm.materialize_tree(repository, store, request, dest_root)

    assert "daily_market" not in tree.copied_tables
    assert tree.row_counts["daily_market"] == 1
    (path,) = tree.curated_files["daily_market"][2020]
    assert lm._parquet_row_count(path) == 1


def test_ceiling_below_actual_rows_is_refused_not_clamped(tmp_path):
    """Decision 2's second half: a request whose ceiling sits below the rows
    its own (non-excluding) predicates would actually match is refused —
    RESULT_LIMIT_EXCEEDED from the scan itself — never silently clamped
    down to fewer rows than what matches."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    narrow_query = dataclasses.replace(
        request.table_queries["daily_market"],
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAA",)),),
        max_result_rows=1, max_batch_rows=1)  # AAA matches 2 rows (2020, 2021), ceiling says 1
    # Decoupled from the task brief's daily_market-covers-trades gate, same
    # as the sibling test above: this test is about the ceiling, not trades.
    no_trades_match = dataclasses.replace(
        request.table_queries["trades"],
        key_filter=(KeyPredicate(column="ticker", operator="in", values=("ZZZ",)),))
    request = dataclasses.replace(
        request, table_queries={**request.table_queries, "daily_market": narrow_query,
                                "trades": no_trades_match})
    dest_root = tmp_path / "legacy_root"
    with pytest.raises(DataError) as err:
        lm.materialize_tree(repository, store, request, dest_root)
    assert err.value.code == "RESULT_LIMIT_EXCEEDED"


def test_whole_table_copy_eligible_rejects_a_narrower_projection(tmp_path):
    """Decision 2's "full contract column set" clause, isolated from the
    key_filter/interval/ceiling checks: dropping even one column from an
    otherwise-honest whole-table query refuses byte-copy eligibility."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    contract = contract_for("daily_market")
    query = request.table_queries["daily_market"]
    narrow_columns = dataclasses.replace(query, columns=query.columns[:-1])
    assert not lm._whole_table_copy_eligible(repository, snap, "daily_market", contract, narrow_columns)
    assert lm._whole_table_copy_eligible(repository, snap, "daily_market", contract, query)


def test_whole_table_copy_eligible_rejects_a_narrower_time_interval(tmp_path):
    """Decision 2's "no time bound narrower than the manifest bounds"
    clause, isolated from the key_filter check: an honest (empty) key_filter
    with only the time_interval narrowed still refuses eligibility."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    contract = contract_for("daily_market")
    query = request.table_queries["daily_market"]
    assert query.key_filter == ()
    narrow_interval = dataclasses.replace(
        query, time_interval=TimeInterval(column="date", start_inclusive="2020-06-01",
                                          end_exclusive="2021-01-01"))
    assert not lm._whole_table_copy_eligible(repository, snap, "daily_market", contract, narrow_interval)


def test_interval_covers_direct_cases():
    """Decision 2's covering rule, unit-tested for the branches an honest
    whole-table query built by _build_table_query never exercises: no
    manifest interval at all (a table with no observation_time_column --
    "none of this plan's tables hit that branch today, but the rule is
    general", per _whole_table_bounds), and a missing/mismatched query
    interval against a real manifest one."""
    manifest = TimeInterval(column="date", start_inclusive="2020-01-01", end_exclusive="2021-01-01")
    assert lm._interval_covers(None, None)
    assert not lm._interval_covers(manifest, None)
    assert not lm._interval_covers(None, manifest)
    assert not lm._interval_covers(
        TimeInterval(column="other", start_inclusive="2020-01-01", end_exclusive="2021-01-01"), manifest)
    assert lm._interval_covers(manifest, manifest)
    assert lm._interval_covers(
        TimeInterval(column="date", start_inclusive="2019-01-01", end_exclusive="2022-01-01"), manifest)
    assert not lm._interval_covers(
        TimeInterval(column="date", start_inclusive="2020-06-01", end_exclusive="2021-01-01"), manifest)
    assert not lm._interval_covers(
        TimeInterval(column="date", start_inclusive="2020-01-01", end_exclusive="2020-06-01"), manifest)


# --------------------------------------------------------------------------
# review round 4: verified byte-for-byte copies, the panel's hash, and
# pinned Tier-4 serving-model cache refs
# --------------------------------------------------------------------------


def _fragment_hash(record) -> str:
    from engine.v2.foundation import CONTENT_HASH_PREFIX
    return record.object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)


def _file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_whole_table_outputs_are_byte_identical_verified_copies(tmp_path):
    """Decision 1 (review round 4): every whole_table output is a verified
    byte-for-byte copy of its own source object, never a Repository-scan
    rewrite -- including a genuinely multi-fragment partition, whose
    part-NNNN files must keep fragment_records's own manifest order.
    option_chains stays evidence_scoped, so it is rewritten, not copied."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    dest_root = tmp_path / "legacy_root"
    tree = lm.materialize_tree(repository, store, request, dest_root)

    assert tree.copied_tables == {"earnings_events", "daily_market", "trades",
                                  "feature_panel", "tier4_forecasts"}
    assert "option_chains" not in tree.copied_tables

    # daily_market's deliberately two-fragment 2020 partition (AAA, then BBB
    # -- see _build_snapshot) keeps its part-NNNN numbering in exactly the
    # order fragment_records itself returns for that year.
    dm_2020_records = [r for r in repository.fragment_records(snap, "daily_market")
                       if r.partition_key == "2020"]
    dm_2020_paths = tree.curated_files["daily_market"][2020]
    assert len(dm_2020_paths) == 2 == len(dm_2020_records)
    assert [p.name for p in dm_2020_paths] == ["part-0000.parquet", "part-0001.parquet"]
    for path, record in zip(dm_2020_paths, dm_2020_records):
        assert _file_sha256(path) == _fragment_hash(record)

    for table_name in ("earnings_events", "daily_market", "trades"):
        for year, paths in tree.curated_files[table_name].items():
            records = [r for r in repository.fragment_records(snap, table_name)
                      if int(r.partition_key) == year]
            assert len(paths) == len(records)
            for path, record in zip(paths, records):
                assert _file_sha256(path) == _fragment_hash(record)

    for table_name in ("feature_panel", "tier4_forecasts"):
        (record,) = repository.fragment_records(snap, table_name)
        assert _file_sha256(tree.single_files[table_name]) == _fragment_hash(record)


def test_materialized_panel_hash_equals_source_panel_object_hash(tmp_path):
    """Decision 1's whole point: a byte-identical panel.parquet means
    engine.data.store.file_sha256(paths.PANEL) -- the Tier-4 serving-model
    cache key -- equals the SOURCE panel object's own hash, not a fresh
    rewrite's incidental one."""
    from engine.data.store import file_sha256

    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    dest_root = tmp_path / "legacy_root"
    materialize(repository, store, request, dest_root)

    (record,) = repository.fragment_records(snap, "feature_panel")
    assert file_sha256(dest_root / "data" / "features" / "panel.parquet") == _fragment_hash(record)


def test_tier4_cache_ref_naming_an_unpublished_object_is_refused(tmp_path):
    """A Tier-4 serving-model cache ref whose FILENAME correctly encodes this
    request's own panel-hash prefix (decision 2's own contract) but whose
    pinned content_hash names an object never published -- a cache artifact
    that is genuinely MISSING from the store, distinct from (d) below's
    STALE, present-but-wrong-hash case -- is refused before the request
    exists (decision 5's existing pinned-ref-must-resolve check, now proven
    for this ref category too). A request that never comes into being can
    never be handed to read_plan_complete as complete."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    registry_refs, calendar_refs = _pinned_refs(store)

    (panel_record,) = repository.fragment_records(snap, "feature_panel")
    panel_prefix = _fragment_hash(panel_record)[:12]
    never_published_hash = "sha256:" + "cd" * 32
    missing_cache_ref = lm.format_pinned_ref(
        f"{TIER4_SERVING_DIR}/size_202001_{panel_prefix}.joblib", never_published_hash)

    with pytest.raises(DataError) as err:
        lm.build_materialization_request(
            repository, store, snap, _snapshot_object_ref(store), direct_scope=DIRECT_SCOPE,
            evidence_scope=EVIDENCE_SCOPE, registry_and_model_refs=(*registry_refs, missing_cache_ref),
            calendar_refs=calendar_refs, expected_population={}, observation_ceiling=FAR_FUTURE_CEILING)
    assert err.value.code == "OBJECT_CORRUPT"


def test_tier4_cache_ref_with_wrong_panel_hash_prefix_is_stale(tmp_path):
    """(d): a pinned Tier-4 cache ref that DOES resolve in the store but
    whose filename's embedded panel-hash prefix does not match this
    request's own panel object -- a stale ref carried over from a different
    snapshot -- fails tier4_cache_refs_match_panel/read_plan_complete and is
    refused by materialize() with TIER4_CACHE_STALE. Cleanup-on-failure
    leaves nothing behind, not even the empty dest_root."""
    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    registry_refs, calendar_refs = _pinned_refs(store)
    dummy_hash = store.publish_bytes(b"not a real joblib file",
                                     schema_ref="legacy_pinned_ref.v1").content_hash
    stale_ref = lm.format_pinned_ref(f"{TIER4_SERVING_DIR}/size_202001_deadbeef0000.joblib", dummy_hash)

    request = lm.build_materialization_request(
        repository, store, snap, _snapshot_object_ref(store), direct_scope=DIRECT_SCOPE,
        evidence_scope=EVIDENCE_SCOPE, registry_and_model_refs=(*registry_refs, stale_ref),
        calendar_refs=calendar_refs, expected_population={}, observation_ceiling=FAR_FUTURE_CEILING)

    assert not lm.tier4_cache_refs_match_panel(repository, request)
    assert not lm.read_plan_complete(request, repository)
    dest_root = tmp_path / "legacy_root"
    with pytest.raises(DataError) as err:
        materialize(repository, store, request, dest_root)
    assert err.value.code == "TIER4_CACHE_STALE"
    assert not dest_root.exists()


def test_tampered_object_byte_is_refused_with_nothing_left_writable(tmp_path):
    """(e): a copied table's source object corrupted on disk after
    verification (simulated bit rot, not a bug in this module's write path)
    is caught by _copy_verified_object's own hash check. daily_market is
    whole-table-copy-eligible here, so evidence_scope_covers_trades's own
    check is vacuous by construction and never scans trades at all (task
    brief decision) -- the corrupted daily_market fragment's own byte copy
    is what raises OBJECT_CORRUPT. Refused as OBJECT_CORRUPT, and -- via
    legacy_adapter.materialize's cleanup-on-failure -- leaves NOTHING behind,
    including earnings_events, copied successfully just before daily_market
    in SCORE_READ_PLAN_TABLES order."""
    from engine.v2.foundation import CONTENT_HASH_PREFIX

    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)

    (dm_record,) = [r for r in repository.fragment_records(snap, "daily_market")
                    if r.partition_key == "2021"]
    digest = dm_record.object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    object_path = Path(store.root) / "objects" / digest[:2] / digest
    corrupted = bytearray(object_path.read_bytes())
    corrupted[0] ^= 0xFF
    object_path.write_bytes(bytes(corrupted))

    dest_root = tmp_path / "legacy_root"
    with pytest.raises(DataError) as err:
        materialize(repository, store, request, dest_root)
    assert err.value.code == "OBJECT_CORRUPT"
    assert not dest_root.exists()


def test_copy_source_object_unreadable_is_refused(tmp_path):
    """The other half of _copy_verified_object's own error handling: the
    source object's file is gone from the store entirely (not merely wrong
    bytes) -- an OSError on open/read, not a hash mismatch."""
    from engine.v2.foundation import CONTENT_HASH_PREFIX

    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)

    (dm_record,) = [r for r in repository.fragment_records(snap, "daily_market")
                    if r.partition_key == "2021"]
    digest = dm_record.object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    object_path = Path(store.root) / "objects" / digest[:2] / digest
    object_path.unlink()

    dest_root = tmp_path / "legacy_root"
    with pytest.raises(DataError) as err:
        materialize(repository, store, request, dest_root)
    assert err.value.code == "OBJECT_CORRUPT"
    assert not dest_root.exists()


def test_multi_fragment_feature_panel_falls_back_to_rewrite_and_skips_tier4_cache_check(tmp_path):
    """A genuinely multi-fragment feature_panel (more than one physical
    object behind the same logical file) cannot be represented as one
    verified byte-for-byte copy -- legacy load_panel()/load_forecasts() open
    a single fixed path, never a part-* directory. _copy_whole_single_file
    correctly declines (returns False) and materialize_tree falls back to
    the existing Repository-scan rewrite, which stays row-correct even
    though it is no longer byte-identical to any one source object.
    panel_object_ref also has no single hash to check a pinned Tier-4 cache
    ref's filename against here, so _check_tier4_cache_refs structurally
    skips validation (tier4_cache_refs_match_panel is vacuously True) --
    documented behavior, not a silent gap."""
    conn, clock, store = catalog_and_store(tmp_path)
    contracts = {name: contract_for(name) for name in TABLES}
    records = {"earnings_events": [], "daily_market": [], "trades": [], "option_chains": [],
              "feature_panel": [], "tier4_forecasts": []}
    for year, rows in EE_ROWS.items():
        records["earnings_events"].append(
            publish_and_inspect(store, contracts["earnings_events"], _ref(contracts["earnings_events"]),
                               rows, partition_key=year))
    for (year, _ticker), rows in sorted(DM_ROWS.items()):
        records["daily_market"].append(
            publish_and_inspect(store, contracts["daily_market"], _ref(contracts["daily_market"]),
                               rows, partition_key=year))
    for year, rows in TRADE_ROWS.items():
        records["trades"].append(
            publish_and_inspect(store, contracts["trades"], _ref(contracts["trades"]),
                               rows, partition_key=year))
    for year, rows in CHAIN_ROWS.items():
        records["option_chains"].append(
            publish_and_inspect(store, contracts["option_chains"], _ref(contracts["option_chains"]),
                               rows, partition_key=year))
    # feature_panel split into TWO fragments under the same "all" partition --
    # the genuinely multi-fragment case _copy_whole_single_file declines.
    for row in PANEL_ROWS:
        records["feature_panel"].append(
            publish_and_inspect(store, contracts["feature_panel"], _ref(contracts["feature_panel"]),
                               [row], partition_key="all"))
    records["tier4_forecasts"].append(
        publish_and_inspect(store, contracts["tier4_forecasts"], _ref(contracts["tier4_forecasts"]),
                           TIER4_ROWS, partition_key="all"))

    dm_2020 = [r for r in records["daily_market"] if r.partition_key == "2020"]
    dm_2020_hash = partition_logical_hash(
        store, [r.object_ref for r in dm_2020], contracts["daily_market"],
        _ref(contracts["daily_market"]), "2020")
    panel_hash = partition_logical_hash(
        store, [r.object_ref for r in records["feature_panel"]], contracts["feature_panel"],
        _ref(contracts["feature_panel"]), "all")

    snap = commit_tables(conn, clock, records, contracts, store=store,
                         partition_logical_hashes={"daily_market": {"2020": dm_2020_hash},
                                                   "feature_panel": {"all": panel_hash}})
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)
    dest_root = tmp_path / "legacy_root"
    tree = lm.materialize_tree(repository, store, request, dest_root)

    assert tree.copied_tables == {"earnings_events", "daily_market", "trades", "tier4_forecasts"}
    assert "feature_panel" not in tree.copied_tables

    assert len(repository.fragment_records(snap, "feature_panel")) == 2
    assert lm.panel_object_ref(repository, snap) is None
    assert lm.tier4_cache_refs_match_panel(repository, request)

    from engine.v2.data.legacy_adapter import read_legacy_part
    panel_frame = read_legacy_part(tree.single_files["feature_panel"], columns=None)
    assert sorted(panel_frame["ticker"].tolist()) == ["AAA", "BBB"]


def test_pinned_ref_object_corrupted_between_build_and_materialize_is_refused(tmp_path):
    """Decision 5 exists so build-time verification and write-time
    resolution can never disagree about what "resolves" means
    (_resolve_pinned_bytes's own docstring). A registry.json object that
    passed _verify_pinned_ref_exists at build time but is corrupted on disk
    before materialize() ever reads it -- the pinned-ref analogue of (e)'s
    tampered curated-table object -- is still caught, by the same function,
    the second time it is opened."""
    from engine.v2.foundation import CONTENT_HASH_PREFIX

    conn, store, snap = _build_snapshot(tmp_path)
    repository = Repository(conn, store)
    request = _build_request(repository, snap, _snapshot_object_ref(store), store)

    registry_path, registry_hash = lm.parse_pinned_ref(request.registry_and_model_refs[0])
    assert registry_path == "engine/models/registry.json"
    digest = registry_hash.removeprefix(CONTENT_HASH_PREFIX)
    object_path = Path(store.root) / "objects" / digest[:2] / digest
    corrupted = bytearray(object_path.read_bytes())
    corrupted[0] ^= 0xFF
    object_path.write_bytes(bytes(corrupted))

    dest_root = tmp_path / "legacy_root"
    with pytest.raises(DataError) as err:
        materialize(repository, store, request, dest_root)
    assert err.value.code == "OBJECT_CORRUPT"
    assert not dest_root.exists()


def test_tier4_cache_hash_prefix_ignores_a_malformed_filename():
    """_tier4_cache_hash_prefix (the filename parser _check_tier4_cache_refs
    relies on) returns None -- "not shaped like one" -- for a path that
    lives in TIER4_SERVING_DIR and ends in .joblib but has no '_' separator to
    carry a model_id/fold/hash-prefix triple, rather than raising or
    guessing. Such a ref is therefore never flagged TIER4_CACHE_STALE no
    matter its actual hash prefix -- it is simply not recognized as a
    Tier-4 cache ref at all."""
    assert lm._tier4_cache_hash_prefix(f"{TIER4_SERVING_DIR}/nounderscore.joblib") is None
    assert lm._tier4_cache_hash_prefix(f"{TIER4_SERVING_DIR}/model_fold_abc123.joblib") == "abc123"
    assert lm._tier4_cache_hash_prefix("engine/models/registry.json") is None
