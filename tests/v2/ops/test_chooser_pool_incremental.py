import pandas as pd
import pytest

from engine.v2.models.training.fold_store import TrainingRefused
from tools import refresh_chooser_pool_candidates as refresh


def _row(event_id, value, event_date="2026-09-01", entry_date="2026-09-01"):
    return {
        "event_id": event_id, "arm": "N7q2_-1_-1_1", "variant": 0,
        "fill_alpha": 0.5, "event_date": pd.Timestamp(event_date),
        "entry_date": pd.Timestamp(entry_date), "exit_date": pd.Timestamp("2026-10-15"),
        "marker": value,
    }


def test_month_requires_canonical_value():
    assert refresh.require_month("2026-10") == "2026-10"
    with pytest.raises(TrainingRefused, match="invalid_exit_month"):
        refresh.require_month("2026-13")


def test_append_keeps_existing_key_and_adds_only_unseen_rows(tmp_path):
    path = tmp_path / "candidates.parquet"
    pd.DataFrame([_row("existing", 1)]).to_parquet(path, index=False)
    fresh = pd.DataFrame([_row("existing", 99), _row("new", 2), _row("new", 3)])

    assert refresh.append_candidates(path, fresh) == 1
    result = pd.read_parquet(path).set_index("event_id")
    assert result.loc["existing", "marker"] == 1
    assert result.loc["new", "marker"] == 2
    assert len(result) == 2


def test_append_refuses_future_feature_date_without_changing_table(tmp_path):
    path = tmp_path / "candidates.parquet"
    original = pd.DataFrame([_row("existing", 1)])
    original.to_parquet(path, index=False)

    with pytest.raises(TrainingRefused, match="candidate_causality_failed"):
        refresh.append_candidates(
            path, pd.DataFrame([_row("new", 2, event_date="2026-10-02", entry_date="2026-10-01")])
        )

    pd.testing.assert_frame_equal(pd.read_parquet(path), original)


def test_missing_candidate_table_is_typed_refusal(tmp_path):
    with pytest.raises(TrainingRefused, match="candidate_table_missing"):
        refresh.append_candidates(tmp_path / "missing.parquet", pd.DataFrame())
