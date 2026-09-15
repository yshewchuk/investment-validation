"""Pinned legacy reference inputs for snapshot-backed scoring (guide §14, D13/D14).

* Tier 0: every ``LEGACY_REFERENCE_INPUTS_V1`` path equals the legacy constant
  that defines it, relative to ``engine.paths.ROOT``.
* ``plan_import`` resolution and refusals on a synthetic legacy root
  (``tests.test_v2_data_import.build_legacy_store``).
* Imports through a real ``Service``: the exact rows recorded per receipt, and
  a model-only change that reuses the snapshot while recording new refs.
* Schema v5: checksummed, idempotent, append-only, committed receipts only.
* End to end: import, ``ops plan nightly --input-mode snapshot``, submit, and
  the real ``legacy_materialize`` worker writes every reference file at its
  exact legacy path, byte-identical to the source.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from engine.v2.contracts import LegacyFileRef, LegacyInputManifest
from engine.v2.data import legacy_mapping
from engine.v2.data import reference_inputs as ri
from engine.v2.data import schema as data_schema
from engine.v2.data.errors import DataError
from engine.v2.data.import_snapshot import plan_import
from engine.v2.data.reference_catalog import (
    REFERENCE_KINDS,
    ReferenceInput,
    insert_reference_inputs,
    reference_inputs_for_snapshot,
)
from engine.v2.foundation import SystemClock, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.cli import dispatch, parser
from engine.v2.ops.errors import OpsError
from engine.v2.ops.lifecycle import request_cancel
from engine.v2.ops.migrations import Migration, migrate
from engine.v2.ops.snapshot_roots import default_materialization_base, materialization_root
from engine.v2.ops.stages import registry
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY
from tests.test_v2_data_catalog import _assert_immutable, build_chain, catalog, insert_receipt
from tests.test_v2_data_import import (
    REFERENCE_MODEL_ID,
    ROOT,
    _run_until_terminal,
    _submit_and_run,
    build_legacy_store,
    reference_artifact_path,
    reference_cache_path,
    write_reference_inputs,
)

INPUTS = ri.LEGACY_REFERENCE_INPUTS_V1["inputs"]
SCOPE = "shadow"


def _plan(root):
    return plan_import(root, scope=SCOPE, expected_head_snapshot_id=None, expected_head_generation=0)


def _refused(root) -> str:
    with pytest.raises(DataError) as err:
        _plan(root)
    return err.value.code


def _expected_paths(root: Path) -> set[str]:
    exact = {spec["path"] for spec in INPUTS.values() if spec["resolution"] == "exact"}
    return exact | {reference_artifact_path(), reference_cache_path(root)}


# --------------------------------------------------------------------------
# tier 0: paths come from the legacy constants
# --------------------------------------------------------------------------


def test_reference_input_paths_equal_their_legacy_constants():
    from engine import paths, pnl_sim, recalibrate, score, structure_registry
    from engine.data.features import tier4
    from engine.models import registry as model_registry

    def rel(module, path):  # each module's own ``paths`` object, in case a test reloaded it
        return Path(path).relative_to(module.paths.ROOT).as_posix()

    assert set(INPUTS) == set(REFERENCE_KINDS)
    assert INPUTS["calendar"]["path"] == rel(score, paths.GSPC_DAILY)
    assert INPUTS["calendar"]["path"] == "earnings_predictions/data/raw/polygon/gspc_daily.csv"
    assert INPUTS["calendar"]["path"].startswith("earnings_predictions/")
    assert INPUTS["model_registry"]["path"] == rel(model_registry, model_registry.REGISTRY_PATH)
    assert INPUTS["structure_champions"]["path"] == rel(structure_registry,
                                                        structure_registry.CHAMPIONS_PATH)
    assert INPUTS["champion_artifact"]["directory"] == rel(model_registry, model_registry.ARTIFACT_DIR)
    assert INPUTS["tier4_serving_cache"]["directory"] == rel(tier4, tier4.SERVING_DIR)
    assert INPUTS["chooser_analog_pool"]["path"] == rel(
        score, score.paths.FEATURES / score.CHOOSER_ANALOG_POOL)
    assert INPUTS["legacy_snapshot"]["path"] == rel(score, score.paths.SNAPSHOT_FILE)
    assert INPUTS["pnl_sim_history"]["path"] == pnl_sim.HISTORY_PATH
    assert INPUTS["pnl_sim_history"]["path"] == "data/features/pnl_sim_history.parquet"
    assert INPUTS["recalibration_pairs"]["path"] == rel(recalibrate, recalibrate.PAIRS_PATH)
    assert ri.DATA_DIR == rel(score, score.paths.DATA)
    for data_relative, constant in ((legacy_mapping.PANEL_RELATIVE_PATH, score.paths.PANEL),
                                    (legacy_mapping.TIER4_RELATIVE_PATH, score.paths.TIER4),
                                    (legacy_mapping.SNAPSHOT_RELATIVE_PATH, score.paths.SNAPSHOT_FILE)):
        assert f"{ri.DATA_DIR}/{data_relative}" == rel(score, constant)


def test_kind_for_path_classifies_every_declared_input():
    for kind, spec in INPUTS.items():
        if spec["resolution"] == "exact":
            assert ri.kind_for_path(spec["path"]) == kind
    serving = INPUTS["tier4_serving_cache"]["directory"]
    assert ri.kind_for_path(f"{serving}/size_v1_4_202601_{'a' * 12}.joblib") == "tier4_serving_cache"
    assert ri.kind_for_path(f"{INPUTS['champion_artifact']['directory']}/x.joblib") == "champion_artifact"
    assert ri.kind_for_path(f"{serving}/notes.txt") is None
    assert ri.kind_for_path("data/features/panel.parquet") is None


# --------------------------------------------------------------------------
# plan_import resolution and refusals
# --------------------------------------------------------------------------


def test_plan_pins_exactly_the_reference_files_and_excludes_a_stale_cache(tmp_path):
    build_legacy_store(tmp_path)
    stale = f"{ri.TIER4_SERVING_DIR}/{REFERENCE_MODEL_ID}_202401_{'0' * 12}.joblib"
    (tmp_path / stale).write_bytes(b"cache for a different panel")
    manifest = _plan(tmp_path).legacy_input_manifest
    pinned = {ref.path for ref in manifest.file_refs if ri.kind_for_path(ref.path)}
    assert pinned == _expected_paths(tmp_path) and stale not in pinned
    assert manifest.calendar_ref.startswith(INPUTS["calendar"]["path"] + "::")
    assert {ref.split("::")[0] for ref in manifest.registry_and_model_refs} == (
        _expected_paths(tmp_path) - {INPUTS["calendar"]["path"], INPUTS["legacy_snapshot"]["path"]})


def test_missing_registry_referenced_artifact_is_refused(tmp_path):
    build_legacy_store(tmp_path)
    (tmp_path / reference_artifact_path()).unlink()
    assert _refused(tmp_path) == "INPUT_CHANGED"


def test_missing_exact_reference_input_is_refused(tmp_path):
    build_legacy_store(tmp_path)
    (tmp_path / INPUTS["calendar"]["path"]).unlink()
    assert _refused(tmp_path) == "INPUT_CHANGED"


@pytest.mark.parametrize("kind", ["pnl_sim_history", "recalibration_pairs"])
def test_missing_model_output_reference_input_is_refused(tmp_path, kind):
    """Task brief 2026-09-14: the import must refuse typed, not silently pin
    without them, when either Tier-4-derived model output is absent."""
    build_legacy_store(tmp_path)
    (tmp_path / INPUTS[kind]["path"]).unlink()
    assert _refused(tmp_path) == "INPUT_CHANGED"


def test_tier4_champion_without_a_cache_for_this_panel_is_stale(tmp_path):
    build_legacy_store(tmp_path)
    (tmp_path / reference_cache_path(tmp_path)).rename(
        tmp_path / f"{ri.TIER4_SERVING_DIR}/{REFERENCE_MODEL_ID}_202401_{'0' * 12}.joblib")
    assert _refused(tmp_path) == "TIER4_CACHE_STALE"


def test_bad_registry_or_artifact_is_a_contract_mismatch(tmp_path):
    build_legacy_store(tmp_path)
    (tmp_path / reference_artifact_path()).write_bytes(b"retrained without re-registering")
    assert _refused(tmp_path) == "CONTRACT_MISMATCH"
    registry_path = tmp_path / INPUTS["model_registry"]["path"]
    for artifact in ("../outside.joblib", "/abs/model.joblib",
                     f"{ri.TIER4_SERVING_DIR}/{REFERENCE_MODEL_ID}.joblib"):
        write_reference_inputs(tmp_path)
        document = json.loads(registry_path.read_text())
        document["models"][0]["artifact"] = artifact
        registry_path.write_text(json.dumps(document))
        assert _refused(tmp_path) == "CONTRACT_MISMATCH", artifact
    registry_path.write_text("[]")
    assert _refused(tmp_path) == "CONTRACT_MISMATCH"


# --------------------------------------------------------------------------
# imports through a real Service
# --------------------------------------------------------------------------


def _import(tmp_path, store_root, conn, clock, key, head=None, generation=0):
    receipt, state, plan = _submit_and_run(tmp_path / "ops", store_root, conn, clock,
                                           idempotency_key=key, expected_head=head,
                                           expected_generation=generation, scope=SCOPE)
    failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?", (receipt.job_id,)).fetchone()[0]
    assert state == "succeeded", failure
    return plan


def _rows(conn):
    return {row[0]: tuple(row[1:]) for row in conn.execute(
        "SELECT legacy_path, receipt_id, kind, content_hash, byte_size, fold "
        "FROM data_import_reference_inputs")}


def _head(conn):
    return tuple(conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope=?",
                              (SCOPE,)).fetchone())


def _store_bytes(root: Path, content_hash: str) -> bytes:
    digest = content_hash.removeprefix("sha256:")
    return (root / "ops" / "objects" / digest[:2] / digest).read_bytes()


@pytest.fixture
def imported(tmp_path):
    store_root = tmp_path / "legacy_store"
    (tmp_path / "ops").mkdir()
    build_legacy_store(store_root)
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops" / "ops.sqlite", clock=clock)
    try:
        yield tmp_path, store_root, conn, clock
    finally:
        conn.close()


def test_import_records_exactly_the_expected_refs_per_receipt(imported):
    tmp_path, store_root, conn, clock = imported
    stale = f"{ri.TIER4_SERVING_DIR}/{REFERENCE_MODEL_ID}_202401_{'0' * 12}.joblib"
    (store_root / stale).write_bytes(b"cache for a different panel")
    _import(tmp_path, store_root, conn, clock, "refs-1")
    rows = _rows(conn)
    assert set(rows) == _expected_paths(store_root) and stale not in rows
    (receipt_id,) = {row[0] for row in rows.values()}
    status, snapshot_id = conn.execute("SELECT status, result_snapshot_id FROM data_import_receipts "
                                       "WHERE receipt_id=?", (receipt_id,)).fetchone()
    assert status == "committed" and snapshot_id == _head(conn)[0]
    for path, (_, kind, content_hash, byte_size, fold) in rows.items():
        source = (store_root / path).read_bytes()
        assert kind == ri.kind_for_path(path) and byte_size == len(source)
        assert _store_bytes(tmp_path, content_hash) == source
        if kind in ("pnl_sim_history", "recalibration_pairs"):
            assert len(fold) == 6 and fold.isdigit(), (path, fold)
        else:
            assert fold == "", (path, fold)


def test_fold_is_the_snapshot_data_session_not_the_import_clock(tmp_path):
    """SEND-BACK on 1c04cf8: the fold must come from the snapshot's own data
    (its ``generated_at`` decision session, read by ``import_snapshot._check_
    snapshot_shape`` into ``LegacyInputManifest.selected_session``), never
    from ``clock.now()`` at import time. Data session 2026-08-31, import
    clock 2026-09-14 (today) -- a clock-derived fold would read "202609";
    the correct fold, ``legacy_serving_fold("2026-08-31", "2026-08-31")``,
    is "202608"."""
    from engine.v2.data import legacy_adapter

    root, store_root = tmp_path / "ops", tmp_path / "legacy_store"
    root.mkdir()
    store_root.mkdir()
    build_legacy_store(store_root, generated_at="2026-08-31T00:00:00+00:00")

    clock = SystemClock()  # real now() -- 2026-09-14, a different month than the data session
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    try:
        _import(tmp_path, store_root, conn, clock, "fold-cross-month")
        rows = _rows(conn)
        expected = f"{legacy_adapter.legacy_serving_fold('2026-08-31', '2026-08-31'):%Y%m}"
        assert expected == "202608"
        for path, (_, kind, _hash, _size, fold) in rows.items():
            if kind in ("pnl_sim_history", "recalibration_pairs"):
                assert fold == expected, (path, fold)
    finally:
        conn.close()


def test_fold_is_identical_across_reimports_with_different_clocks(tmp_path):
    """SEND-BACK on 1c04cf8, second required test: re-importing the SAME
    snapshot data with a different wall-clock time must record the SAME
    fold -- the fold is a property of the snapshot's data, not of when the
    import happened to run."""
    from tests.ops_support import FakeClock

    root_a, store_a = tmp_path / "ops_a", tmp_path / "legacy_store_a"
    root_b, store_b = tmp_path / "ops_b", tmp_path / "legacy_store_b"
    for root, store in ((root_a, store_a), (root_b, store_b)):
        root.mkdir()
        store.mkdir()
        build_legacy_store(store, generated_at="2026-08-31T00:00:00+00:00")

    early_clock = FakeClock()  # .value == 2026-09-12
    late_clock = FakeClock()
    late_clock.advance(60 * 60 * 24 * 45)  # +45 days -- a different month, and a different clock class

    conn_a = open_catalog(root_a / "ops.sqlite", clock=early_clock)
    conn_b = open_catalog(root_b / "ops.sqlite", clock=late_clock)
    try:
        receipt_a, state_a, _ = _submit_and_run(root_a, store_a, conn_a, early_clock,
                                                idempotency_key="fold-early", scope=SCOPE)
        receipt_b, state_b, _ = _submit_and_run(root_b, store_b, conn_b, late_clock,
                                                idempotency_key="fold-late", scope=SCOPE)
        assert state_a == "succeeded" and state_b == "succeeded"
        folds_a = {kind: fold for _, kind, _, _, fold in _rows(conn_a).values()
                  if kind in ("pnl_sim_history", "recalibration_pairs")}
        folds_b = {kind: fold for _, kind, _, _, fold in _rows(conn_b).values()
                  if kind in ("pnl_sim_history", "recalibration_pairs")}
        assert folds_a and folds_a == folds_b
    finally:
        conn_a.close()
        conn_b.close()


def test_model_only_change_reuses_the_snapshot_and_records_new_refs(imported):
    tmp_path, store_root, conn, clock = imported
    _import(tmp_path, store_root, conn, clock, "model-1")
    head = _head(conn)
    first = reference_inputs_for_snapshot(conn, scope=SCOPE, snapshot_id=head[0])
    write_reference_inputs(store_root, model_bytes=b"synthetic size model v2")
    _import(tmp_path, store_root, conn, clock, "model-2", head=head[0], generation=head[1])
    assert _head(conn) == head
    second = reference_inputs_for_snapshot(conn, scope=SCOPE, snapshot_id=head[0])
    by_path = [{item.legacy_path: item.content_hash for item in refs} for refs in (first, second)]
    changed = {path for path in by_path[0] if by_path[0][path] != by_path[1][path]}
    assert changed == {reference_artifact_path(), INPUTS["model_registry"]["path"]}
    receipts = conn.execute("SELECT receipt_id FROM data_import_receipts WHERE status='committed' "
                            "AND result_snapshot_id=?", (head[0],)).fetchall()
    assert len(receipts) == 2
    assert len({row[0] for row in _rows_all(conn)}) == 2


def _rows_all(conn):
    return conn.execute("SELECT receipt_id, legacy_path FROM data_import_reference_inputs").fetchall()


# --------------------------------------------------------------------------
# schema v5
# --------------------------------------------------------------------------


def _reference(path="engine/models/registry.json", kind="model_registry"):
    return ReferenceInput(kind=kind, legacy_path=path, object_id="art_x",
                          content_hash="sha256:" + "ab" * 32, byte_size=3)


def test_v5_v6_migrations_are_checksummed_idempotent_and_append_only(tmp_path):
    conn, clock = catalog(tmp_path)
    versions = conn.execute("SELECT version, name, checksum FROM schema_versions WHERE owner='data' "
                            "ORDER BY version").fetchall()
    assert tuple(versions[-1])[:2] == (6, "import_reference_input_fold")
    ids = build_chain(conn, clock)
    insert_reference_inputs(conn, ids["receipt_id"], [_reference()])
    row = conn.execute("SELECT fold FROM data_import_reference_inputs WHERE receipt_id=?",
                       (ids["receipt_id"],)).fetchone()
    assert row[0] == ""  # v6's DEFAULT '' -- unfolded kinds never set one
    _assert_immutable(conn, "data_import_reference_inputs", f"receipt_id = '{ids['receipt_id']}'",
                      "byte_size = 4")
    with pytest.raises(sqlite3.IntegrityError):
        insert_reference_inputs(conn, ids["receipt_id"], [_reference()])  # (receipt, path) is the key
    insert_receipt(conn, clock, receipt_id="r-failed", status="failed", result_snapshot_id=None)
    with pytest.raises(sqlite3.IntegrityError, match="committed import receipt"):
        insert_reference_inputs(conn, "r-failed", [_reference()])
    with pytest.raises(DataError):
        insert_reference_inputs(conn, ids["receipt_id"], [_reference("x", kind="unknown")])
    conn.close()

    reopened = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    assert reopened.execute("SELECT version, name, checksum FROM schema_versions WHERE owner='data' "
                            "ORDER BY version").fetchall() == versions
    reopened.close()
    edited = [Migration(v, n, s) for v, n, s in data_schema.MIGRATIONS]
    version, name, statements = data_schema.MIGRATIONS[5]
    edited[5] = Migration(version, name, statements + ("SELECT 1",))
    raw = sqlite3.connect(str(tmp_path / "catalog.sqlite"), isolation_level=None)
    try:
        with pytest.raises(OpsError) as err:
            migrate(raw, data_schema.OWNER, tuple(edited), clock=clock)
        assert err.value.problem.details["reason"] == "checksum_mismatch"
    finally:
        raw.close()


# --------------------------------------------------------------------------
# plan-time guard: legacy_score in snapshot mode needs both model outputs pinned
# --------------------------------------------------------------------------


def _full_reference_set():
    """One ``ReferenceInput`` per declared kind — a minimal, complete pinned set."""
    return [_reference(path=f"p/{kind}", kind=kind) for kind in REFERENCE_KINDS]


def test_pinned_materialization_refs_accepts_a_complete_set():
    from engine.v2.data.reference_catalog import pinned_materialization_refs

    pinned = pinned_materialization_refs(_full_reference_set())
    paths = {ref.split("::")[0] for ref in pinned["registry_and_model_refs"]}
    assert {"p/pnl_sim_history", "p/recalibration_pairs"} <= paths


@pytest.mark.parametrize("missing_kind", ["pnl_sim_history", "recalibration_pairs"])
def test_pinned_materialization_refs_refuses_without_either_model_output(missing_kind):
    """Task brief 2026-09-14: ``legacy_score`` in snapshot mode must refuse (at
    plan time -- this is the function ``ops.snapshot_planning.pin_snapshot_inputs``
    calls) when the pinned set lacks either Tier-4-derived model output, e.g. a
    snapshot imported before this pin existed."""
    from engine.v2.data.reference_catalog import pinned_materialization_refs

    incomplete = [item for item in _full_reference_set() if item.kind != missing_kind]
    with pytest.raises(DataError) as err:
        pinned_materialization_refs(incomplete)
    assert err.value.code == "SNAPSHOT_NOT_READY"


# --------------------------------------------------------------------------
# causality check (read-only): trailing_cutoff / fit_recalibration
# --------------------------------------------------------------------------


def test_pnl_sim_trailing_cutoff_ignores_rows_on_or_after_as_of():
    """``engine.pnl_sim.trailing_cutoff`` (engine/pnl_sim.py:240-259): the
    window is ``[as_of - window_months, as_of)`` -- strictly before, so a row
    dated exactly ``as_of`` must never enter the trailing bar. Proven by the
    ``min_window`` threshold rather than a quantile shift: 99 genuinely prior
    rows plus one row dated exactly ``as_of`` must read as 99 (UNDETERMINED,
    below ``min_window=100``); only moving that same row to one day earlier
    crosses the threshold and yields a real bar."""
    import pandas as pd

    from engine import pnl_sim

    as_of = pd.Timestamp("2026-06-01")
    prior_dates = [as_of - pd.Timedelta(days=d) for d in range(1, 100)]  # 99 dates, all < as_of
    history = pd.DataFrame({
        "event_date": prior_dates + [as_of],
        "exp_pnl_sim": [0.0] * 99 + [999.0],
    })
    assert pnl_sim.trailing_cutoff(history, as_of, min_window=100) is None

    included = history.copy()
    included.loc[included.index[-1], "event_date"] = as_of - pd.Timedelta(days=100)
    assert pnl_sim.trailing_cutoff(included, as_of, min_window=100) is not None


def test_fit_recalibration_restricts_pairs_to_events_closed_before_the_decision_date():
    """``engine.recalibrate.fit_recalibration`` (engine/recalibrate.py:99-131)
    DOES restrict: ``stamp = pd.Timestamp(before).normalize()`` (:118) then
    ``rows = rows[pd.to_datetime(rows["exit_date"]) < stamp]`` (:124) -- a
    pair whose event closed on or after the decision date is excluded, never
    fit into the map that scores that same decision. Called from
    ``engine.score.Scorer.recalibration`` (engine/score.py:903) with
    ``before=stamp`` = the decision timestamp, so ``before`` really is "the
    decision date", not an unrelated cutoff. Read-only: no engine code
    touched, per the causality-check deliverable."""
    import pandas as pd

    from engine import recalibrate

    before = pd.Timestamp("2026-06-01")
    pairs = pd.DataFrame({
        "strategy": ["S1"] * 130,
        "fill_alpha": [0.5] * 130,
        # 129 pairs closed strictly before `before` (raw_win == outcome, a
        # trivial identity map) plus one poisoned pair closed exactly ON
        # `before` whose outcome disagrees -- if it leaked in, the fitted map
        # would stop being the identity at that raw_win value.
        "exit_date": list(pd.date_range(before - pd.Timedelta(days=200), periods=129, freq="D")) + [before],
        "raw_win": [i / 129 for i in range(129)] + [0.999],
        "outcome": [i / 129 for i in range(129)] + [0.0],
    })
    fitted = recalibrate.fit_recalibration("S1", 0.5, before=before, pairs=pairs, min_pairs=100)
    assert fitted is not None and fitted.n == 129  # the same-day pair never entered the fit
    calibrated = fitted.transform([0.999])[0]
    assert calibrated > 0.9  # the poisoned 0.0 outcome at raw_win=0.999 did not pull this down


# --------------------------------------------------------------------------
# end to end: import -> plan nightly --input-mode snapshot -> real materialize worker
# --------------------------------------------------------------------------


def _plan_argv(tmp_path, store_root):
    tickers = sorted(set(pq.read_table(next((store_root / ri.DATA_DIR / "curated" / "trades").rglob(
        "*.parquet")), columns=["ticker"]).column("ticker").to_pylist()))
    fixture = store_root / INPUTS["structure_champions"]["path"]
    from engine.v2.data.legacy_nightly_read_plan import NIGHTLY_CAPTURE_IMPLEMENTATION_REF
    from engine.v2.ops.fingerprints import file_hash
    manifest = tmp_path / "legacy_manifest.json"
    # This plan's own barrier-only kinds (legacy_finality/decisions/...) are
    # all cancelled below before they ever run -- only legacy_materialize's
    # own snapshot_ref/materialization_request bindings matter to this test
    # (nightly.py:_stage_inputs never binds materialize to legacy_manifest.json
    # at all). The plan-time guard (engine.v2.data.legacy_nightly_read_plan,
    # capture_inputs deliverable) still runs at plan time regardless, so these
    # entries are placeholder PATHS satisfying its presence check, not files
    # this test's materialize worker ever reads.
    placeholder_refs = tuple(
        LegacyFileRef(path=f"{ri.DATA_DIR}/curated/{table}/year=2024/part-0000.parquet",
                      content_hash=file_hash(fixture), byte_size=fixture.stat().st_size)
        for table in ("daily_market", "option_chains", "earnings_events", "trades")
    ) + (
        LegacyFileRef(path="data/raw/fetch/orats/ab/placeholder.meta.json",
                      content_hash=file_hash(fixture), byte_size=fixture.stat().st_size),
        LegacyFileRef(path=INPUTS["structure_champions"]["path"], content_hash=file_hash(fixture),
                      byte_size=fixture.stat().st_size),
    )
    manifest.write_text(json.dumps(to_document(LegacyInputManifest(
        manifest_id="m1", file_refs=placeholder_refs,
        table_contract_refs=(), registry_and_model_refs=("placeholder::sha256:" + "0" * 64,),
        calendar_ref="placeholder::sha256:" + "0" * 64,
        selected_session="2026-09-12", finality_receipt_refs=(), knowledge_mode_by_table={},
        availability_evidence_refs=(), read_set_complete=True,
        capture_implementation_ref=NIGHTLY_CAPTURE_IMPLEMENTATION_REF))))
    population = tmp_path / "population.json"
    population.write_text(json.dumps([f"{tickers[0]}|S1|2024-01-15"]))
    return ["plan", "nightly", "--as-of", "2026-09-12", "--input-mode", "snapshot",
            "--snapshot-scope", SCOPE, "--input-manifest", str(manifest),
            "--expected-population", str(population), "--tickers", ",".join(tickers),
            "--year-start", "2024", "--year-end", "2024"]


def test_import_plan_submit_materializes_reference_files_byte_identical(imported):
    tmp_path, store_root, conn, clock = imported
    ops_root = tmp_path / "ops"
    _import(tmp_path, store_root, conn, clock, "e2e-import")
    planned = dispatch(parser().parse_args(_plan_argv(tmp_path, store_root)), ops_root, conn, clock)
    request_hash = planned["plan"]["snapshot_inputs"]["materialization_request_hash"]
    submitted = dispatch(parser().parse_args(["submit", "--plan", planned["plan_ref"],
                                              "--idempotency-key", "e2e"]), ops_root, conn, clock)
    kinds = {job["job_id"]: json.loads(conn.execute("SELECT spec_json FROM jobs WHERE job_id=?",
                                                    (job["job_id"],)).fetchone()[0])["kind"]
             for job in submitted["jobs"]}
    (materialize_id,) = [job for job, kind in kinds.items() if kind == "legacy_materialize"]
    for job in kinds:
        if job != materialize_id:
            request_cancel(conn, job, None, clock=clock)
    service = Service(conn, ops_root, registry(), TEST_POLICY, clock=clock, code_source=ROOT,
                      store_root=store_root)
    try:
        service.start()
        state = _run_until_terminal(service, conn, materialize_id, timeout=120)
    finally:
        service.close()
    failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?", (materialize_id,)).fetchone()[0]
    assert state == "succeeded", failure

    dest = materialization_root(default_materialization_base(ops_root), request_hash)
    rows = _rows(conn)
    assert set(rows) == _expected_paths(store_root)
    for path in rows:
        assert (dest / path).is_file() and not (dest / path).is_symlink()
        assert (dest / path).read_bytes() == (store_root / path).read_bytes(), path
