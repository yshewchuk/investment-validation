"""S1 monthly retrain training mode: four pool members, receipt and boundary."""
import json

import pandas as pd
import pytest

from engine import paths
from engine.v2.models.chooser_analog_pool import CHOOSER_ANALOG_DIMS
from engine.v2.models.frozen_state import FrozenStateLoader, FrozenStateRef
from engine.v2.models.training import chooser_pool_source
from engine.v2.ops import training
from engine.v2.ops.errors import OpsError
from tools import phase5_training_job as job

MONTH = "2026-02"
CUTOFF = "2026-03-01"
_DIMS = {"exp_pnl_sim": 0.5, "width_over_forecast": 1.0, "n_legs": 5.0,
         "anchor_over_spot": 1.0, "rel_spread": 0.2}


def _params(**changes):
    fields = {"expected_ids": ("training",), "mode": "monthly_pools",
              "month": MONTH, "cutoffs": (CUTOFF,)}
    fields.update(changes)
    return training.TrainingParameters(**fields)


def _problems(**changes):
    return training.training_parameter_problems(None, _params(**changes))


def _worker_params(**changes):
    fields = {"expected_ids": ["training"], "mode": "monthly_pools", "month": MONTH,
              "cutoffs": [CUTOFF], "recipe": "", "state": "", "alpha": None,
              "strategies": [], "pairs_path": "", "ticker_chunk": 1000}
    fields.update(changes)
    return fields


def _row(strategy, day, pnl):
    return {"strategy": strategy, "exit_date": pd.Timestamp(day), "pnl": float(pnl),
            **_DIMS}


def _write_source(tmp_path, rows):
    source = tmp_path / "features"
    source.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(source / "chooser_analog_pool.parquet", index=False)
    return source


def test_valid_month_and_cutoff_combinations():
    assert _problems() == []
    assert _problems(month="2024-02", cutoffs=("2024-03-01",)) == []
    assert _problems(month="2026-12", cutoffs=("2027-01-01",)) == []


@pytest.mark.parametrize("month", ["", "2026-2", "2026-13", "2026-00", "2026-02-01",
                                   "not-a-month"])
def test_invalid_month_is_refused(month):
    assert any("month" in problem for problem in _problems(month=month))


@pytest.mark.parametrize("cutoff", ["2026-02-28", "2026-03-02", "20260301",
                                    "2026-03-01T00:00:00", ""])
def test_cutoff_must_be_the_day_after_the_month_end(cutoff):
    assert _problems(cutoffs=(cutoff,))


def test_exactly_one_cutoff_is_required():
    assert any("exactly one cutoff" in problem for problem in _problems(cutoffs=()))
    assert any("exactly one cutoff" in problem
               for problem in _problems(cutoffs=(CUTOFF, "2026-03-02")))


@pytest.mark.parametrize("changes", [
    {"recipe": "size:*:champion"},
    {"state": "paired_residual_pool"},
    {"alpha": 0.5},
    {"strategies": ("STR-THRU",)},
    {"pairs_path": "data/pairs.parquet"},
    {"catalog_path": "/tmp/catalog"},
    {"objects_root": "/tmp/objects"},
    {"snapshot_id": "snap"},
    {"receipt_id": "receipt"},
    {"input_mode": "native_d14"},
])
def test_forbidden_mode_fields_are_refused(changes):
    assert _problems(**changes)


def test_worker_builds_four_members_and_deterministic_receipt(tmp_path, monkeypatch):
    calls = []

    def fake(member, out, *, plan_only=False, **kwargs):
        calls.append((member, kwargs.get("cutoff")))
        out.mkdir(parents=True, exist_ok=True)
        name = member.replace(":", "__") + ".json"
        (out / name).write_text(member)
        return {"state": member, "content_hash": "sha256:" + member, "file": name,
                "bytes": len(member), "status": "written"}

    monkeypatch.setattr(job, "run_state_job", fake)
    root = tmp_path / "attempt"
    training.run_training_worker(_worker_params(), root)

    assert calls == [
        ("driver_residual_pool:size", None),
        ("driver_residual_pool:implied_t1", None),
        ("driver_residual_pool:runup_move", None),
        ("chooser_analog_pool", CUTOFF),
    ]
    receipt = json.loads((root / "training_result.json").read_text())
    assert receipt["mode"] == "monthly_pools"
    assert receipt["month"] == MONTH
    assert receipt["retrain_cutoff"] == CUTOFF
    assert receipt["experiment_inputs"] == "none"
    assert set(receipt["members"]) == set(job.MONTHLY_POOL_MEMBERS)
    assert all(member["content_hash"] for member in receipt["members"].values())

    first = (root / "training_result.json").read_bytes()
    training.run_training_worker(_worker_params(), root)
    assert (root / "training_result.json").read_bytes() == first


def test_chooser_boundary_includes_m_end_and_excludes_next_day(tmp_path, monkeypatch):
    source = _write_source(tmp_path, [
        _row("CTR5", "2026-02-27", 1.0),
        _row("CTR5", "2026-02-28", 2.0),
        _row("CTR5", "2026-03-01", 3.0),
    ])
    monkeypatch.setattr(paths, "FEATURES", source)
    out = tmp_path / "out"

    summary = job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)

    assert summary["n_rows"] == 2
    pool = FrozenStateLoader(out).load(
        FrozenStateRef(path=summary["file"], content_hash=summary["content_hash"]))
    assert [row[0] for row in pool.rows_for("CTR5")] == ["2026-02-27", "2026-02-28"]


def _run_with_real_chooser(tmp_path, monkeypatch, source):
    real = job.run_state_job

    def dispatch(member, out, *, plan_only=False, **kwargs):
        if member == job.CHOOSER_POOL_STATE:
            return real(member, out, plan_only=plan_only, **kwargs)
        out.mkdir(parents=True, exist_ok=True)
        return {"state": member, "content_hash": "sha256:" + member,
                "file": member.replace(":", "__") + ".json", "bytes": 1,
                "status": "written"}

    monkeypatch.setattr(job, "run_state_job", dispatch)
    monkeypatch.setattr(paths, "FEATURES", source)
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(_worker_params(), tmp_path / "attempt")
    return excinfo.value


def test_missing_source_is_validation_failed(tmp_path, monkeypatch):
    source = tmp_path / "features"
    source.mkdir()
    assert _run_with_real_chooser(tmp_path, monkeypatch, source).code == "VALIDATION_FAILED"


def test_oversized_source_is_validation_failed(tmp_path, monkeypatch):
    source = _write_source(tmp_path, [_row("CTR5", "2026-02-27", 1.0)])
    monkeypatch.setattr(chooser_pool_source, "CHOOSER_POOL_MAX_BYTES", 1)
    assert _run_with_real_chooser(tmp_path, monkeypatch, source).code == "VALIDATION_FAILED"


def test_empty_chooser_pool_is_validation_failed(tmp_path, monkeypatch):
    source = _write_source(tmp_path, [_row("CTR5", "2026-03-01", 1.0)])
    assert _run_with_real_chooser(tmp_path, monkeypatch, source).code == "VALIDATION_FAILED"
