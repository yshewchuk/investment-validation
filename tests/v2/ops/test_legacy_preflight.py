"""Synthetic end-to-end coverage for the post-import preflight."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.data import store as data_store
from engine.v2.ops import cli
from engine.v2.ops.errors import OpsError
from engine.v2.ops.legacy_adapter import run_snapshot_preflight as run_preflight
from tests.v2.ops.test_v2_ops_snapshot_stages import case as _case  # noqa: F401
from tests.v2.ops.test_v2_ops_tier4_coverage import _accept_generation


@pytest.fixture
def case(request):
    """Expose the imported synthetic catalog fixture under its usual name."""
    return request.getfixturevalue("_case")


def _roots(tmp_path, *, source_value, pinned_value):
    source = tmp_path / "source"
    pinned = tmp_path / "pinned"
    source_features = source / "data/features"
    pinned_features = pinned / "data/features"
    source_features.mkdir(parents=True)
    pinned_features.mkdir(parents=True)
    (source / "data/raw").mkdir(parents=True)
    (source / "data/raw/source.csv").write_text("accepted\n")
    def panel(value):
        return pd.DataFrame({"ticker": ["AAA"], "date": ["2026-01-02"],
                             "value": [value]})
    tier4 = pd.DataFrame({"ticker": ["AAA"], "event_date": ["2026-01-02"],
                          "forecast": [0.5], "tier3_snapshot": ["a" * 64]})
    panel(source_value).to_parquet(source_features / "panel.parquet", index=False)
    panel(pinned_value).to_parquet(pinned_features / "panel.parquet", index=False)
    tier4.to_parquet(source_features / "tier4_forecasts.parquet", index=False)
    tier4.to_parquet(pinned_features / "tier4_forecasts.parquet", index=False)
    return source, pinned


def _accept(case, source, receipt_id):
    digest = "sha256:" + str(data_store.file_sha256(
        source / "data/raw/source.csv")).removeprefix("sha256:")
    _accept_generation(case, [("data/raw/source.csv", digest)], receipt_id=receipt_id)


def test_preflight_uses_real_generation_and_numeric_comparison(case, tmp_path):
    source, pinned = _roots(tmp_path, source_value=0.5 + 1e-13, pinned_value=0.5)
    _accept(case, source, "preflight-match")
    before_changes = case.conn.total_changes
    feature_files = (source / "data/features/panel.parquet",
                     source / "data/features/tier4_forecasts.parquet",
                     pinned / "data/features/panel.parquet",
                     pinned / "data/features/tier4_forecasts.parquet")
    before_files = {path: path.read_bytes() for path in feature_files}
    result = run_preflight(case.conn, case.store, source_root=source,
                           materialization_root=pinned, receipt_id="preflight-match")
    assert result["generation"] == "match"
    assert result["features"]["verdict"] == "match"
    assert result["features"]["tables"]["panel"]["verdict"] == "match"
    assert result["features"]["tables"]["tier4"]["verdict"] == "match"
    assert case.conn.total_changes == before_changes
    assert {path: path.read_bytes() for path in feature_files} == before_files


def test_preflight_preserves_features_stale_code(case, tmp_path):
    source, pinned = _roots(tmp_path, source_value=0.75, pinned_value=0.5)
    _accept(case, source, "preflight-features-stale")
    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned, receipt_id="preflight-features-stale")
    assert err.value.code == "FEATURES_STALE"
    assert err.value.problem.details["reason"] == "mismatch"


def test_preflight_refuses_source_change_during_feature_comparison(
        case, tmp_path, monkeypatch):
    from engine.v2.ops import features_compare

    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-source-race")
    original_compare = features_compare.compare_tables
    changed = False

    def compare_then_change(name, rebuilt, pinned_path):
        nonlocal changed
        result = original_compare(name, rebuilt, pinned_path)
        if name == "panel" and not changed:
            changed = True
            panel = pd.read_parquet(rebuilt)
            panel.loc[0, "value"] = 0.75
            panel.to_parquet(rebuilt, index=False)
        return result

    monkeypatch.setattr(features_compare, "compare_tables", compare_then_change)
    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned, receipt_id="preflight-source-race")

    assert changed
    assert err.value.code == "FEATURES_STALE"
    assert err.value.problem.details["reason"] == "comparison_failed"
    assert err.value.problem.details["exception_type"] == "ValueError"


def test_preflight_refuses_pinned_change_during_feature_comparison(
        case, tmp_path, monkeypatch):
    from engine.v2.ops import features_compare

    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-pinned-race")
    original_compare = features_compare.compare_tables
    changed = False

    def compare_then_change(name, rebuilt, pinned_path):
        nonlocal changed
        result = original_compare(name, rebuilt, pinned_path)
        if name == "panel" and not changed:
            changed = True
            panel = pd.read_parquet(pinned_path)
            panel.loc[0, "value"] = 0.75
            panel.to_parquet(pinned_path, index=False)
        return result

    monkeypatch.setattr(features_compare, "compare_tables", compare_then_change)
    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned, receipt_id="preflight-pinned-race")

    assert changed
    assert err.value.code == "FEATURES_STALE"
    assert err.value.problem.details["reason"] == "comparison_failed"
    assert err.value.problem.details["exception_type"] == "OpsError"


def test_preflight_preserves_generation_input_changed_code(case, tmp_path):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-generation-stale")
    (source / "data/raw/source.csv").write_text("changed\n")
    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned, receipt_id="preflight-generation-stale")
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_mismatch"


def test_preflight_rejects_symlinked_read_set_ancestor(case, tmp_path):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-symlinked-ancestor")
    raw_dir = source / "data/raw"
    replacement = tmp_path / "replacement-raw"
    replacement.mkdir()
    (replacement / "source.csv").write_text("accepted\n")
    raw_dir.rename(tmp_path / "original-raw")
    raw_dir.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned,
                      receipt_id="preflight-symlinked-ancestor")

    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["path"] == "data/raw/source.csv"


def test_preflight_maps_generation_read_error_to_input_changed(case, tmp_path, monkeypatch):
    from engine.v2.ops import legacy_adapter

    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-generation-read-error")
    original_open = legacy_adapter.os.open
    source_file = source / "data/raw/source.csv"

    def open_wrapper(path, *args, **kwargs):
        if path == source_file or str(path) == str(source_file):
            raise PermissionError("synthetic unreadable source")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(legacy_adapter.os, "open", open_wrapper)
    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned,
                      receipt_id="preflight-generation-read-error")

    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_read_failed"
    assert err.value.problem.details["path"] == "data/raw/source.csv"


def test_refuse_generation_mismatch_explicit_accepted_none_is_missing(case, tmp_path):
    from engine.v2.ops.generation_binding import refuse_generation_mismatch

    source, _ = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "explicit-none-accepted")
    with pytest.raises(OpsError) as err:
        refuse_generation_mismatch(case.conn, case.store, receipt_id="explicit-none-accepted",
                                   barrier_manifest={"file_refs": []},
                                   accepted=None)

    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_receipt_missing"
    assert err.value.problem.details["receipt_id"] == "explicit-none-accepted"


def test_cli_registers_and_dispatches_snapshot_preflight(monkeypatch, tmp_path):
    source = tmp_path / "source"
    pinned = tmp_path / "pinned"
    seen = {}

    def run(conn, store, **kwargs):
        seen.update(kwargs)
        seen["conn"] = conn
        seen["store_type"] = type(store).__name__
        return {"sentinel": "preflight-dispatched"}

    monkeypatch.setattr("engine.v2.ops.legacy_adapter.run_snapshot_preflight", run)
    args = cli.parser().parse_args([
        "snapshot", "preflight", "--source-root", str(source),
        "--materialization-root", str(pinned), "--receipt-id", "cli-receipt",
    ])
    conn = object()
    clock = object()
    result = cli.dispatch(args, tmp_path, conn, clock)

    assert result == {"sentinel": "preflight-dispatched"}
    assert seen == {"source_root": source, "materialization_root": pinned,
                    "receipt_id": "cli-receipt", "conn": conn,
                    "store_type": "ArtifactStore"}


def test_preflight_rejects_empty_receipt_id(case, tmp_path):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned, receipt_id="")
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_receipt_missing"


def test_preflight_rejects_uncommitted_receipt_id(case, tmp_path):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned,
                      receipt_id="not-a-committed-receipt")

    assert err.value.code == "INPUT_CHANGED"


def test_preflight_preserves_features_stale_code_for_tier4_difference(case, tmp_path):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-tier4-stale")
    tier4_path = source / "data/features/tier4_forecasts.parquet"
    tier4 = pd.read_parquet(tier4_path)
    tier4.loc[0, "forecast"] = 0.75
    tier4.to_parquet(tier4_path, index=False)
    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned, receipt_id="preflight-tier4-stale")
    assert err.value.code == "FEATURES_STALE"
    assert err.value.problem.details["reason"] == "mismatch"
    assert err.value.problem.details["comparison"]["tables"]["tier4"]["verdict"] == "mismatch"


def test_preflight_preserves_features_stale_code_for_corrupt_pinned_table(case, tmp_path):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-corrupt-pinned")
    (pinned / "data/features/panel.parquet").write_bytes(
        b"not-a-parquet-file-with-invalid-footer")

    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned, receipt_id="preflight-corrupt-pinned")

    assert err.value.code == "FEATURES_STALE"
    assert err.value.problem.details["reason"] == "comparison_failed"
    assert isinstance(err.value.problem.details["exception_type"], str)
    assert err.value.problem.details["exception_type"]


def test_preflight_reject_identical_tables_over_whole_table_cap(
        case, tmp_path, monkeypatch):
    from engine.v2.ops import features_compare

    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-identical-table-over-cap")
    monkeypatch.setattr(features_compare, "TABLE_MAX_ROWS", 0)

    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned,
                      receipt_id="preflight-identical-table-over-cap")

    assert err.value.code == "FEATURES_STALE"
    mismatch = err.value.problem.details["comparison"]["tables"]["panel"]
    assert mismatch["mismatches"][0]["reason"] == "table_rows_over_cap"


def test_digest_enforces_streaming_hash_byte_cap(tmp_path):
    from engine.v2.ops.legacy_adapter import _digest

    path = tmp_path / "payload"
    path.write_bytes(b"12345")
    with pytest.raises(OpsError) as err:
        _digest(path, max_bytes=4)

    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "hash_file_over_cap"


def test_preflight_bounds_committed_generation_manifest(case, tmp_path, monkeypatch):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-manifest-over-cap")
    monkeypatch.setattr(
        "engine.v2.ops.generation_binding.MAX_LEGACY_MANIFEST_BYTES", 0)

    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned,
                      receipt_id="preflight-manifest-over-cap")

    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_manifest_over_cap"
    assert err.value.problem.details["byte_size"] > 0
    assert err.value.problem.details["max_bytes"] == 0


def test_preflight_bounds_generation_read_set_hash(case, tmp_path, monkeypatch):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-read-set-over-cap")
    monkeypatch.setattr("engine.v2.ops.legacy_adapter.PREFLIGHT_MAX_HASH_BYTES", 4)

    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned,
                      receipt_id="preflight-read-set-over-cap")

    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "hash_file_over_cap"


def test_preflight_bounds_feature_hash(case, tmp_path, monkeypatch):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    (source / "data/raw/source.csv").write_text("x")
    _accept(case, source, "preflight-feature-over-cap")
    monkeypatch.setattr("engine.v2.ops.legacy_adapter.PREFLIGHT_MAX_HASH_BYTES", 4)

    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned,
                      receipt_id="preflight-feature-over-cap")

    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details == {"reason": "hash_file_over_cap",
                                         "max_bytes": 4}


def test_preflight_refuses_byte_identical_corrupt_panel(case, tmp_path):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-identical-corrupt-panel")
    corrupt = b"not-a-parquet-file-with-invalid-footer"
    (source / "data/features/panel.parquet").write_bytes(corrupt)
    (pinned / "data/features/panel.parquet").write_bytes(corrupt)

    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned,
                      receipt_id="preflight-identical-corrupt-panel")

    assert err.value.code == "FEATURES_STALE"
    assert err.value.problem.details["reason"] == "comparison_failed"
