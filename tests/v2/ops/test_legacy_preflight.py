"""Synthetic end-to-end coverage for the post-import preflight."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.data import store as data_store
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


def test_preflight_preserves_generation_input_changed_code(case, tmp_path):
    source, pinned = _roots(tmp_path, source_value=0.5, pinned_value=0.5)
    _accept(case, source, "preflight-generation-stale")
    (source / "data/raw/source.csv").write_text("changed\n")
    with pytest.raises(OpsError) as err:
        run_preflight(case.conn, case.store, source_root=source,
                      materialization_root=pinned, receipt_id="preflight-generation-stale")
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.details["reason"] == "generation_mismatch"
