"""#1102 slice A: the ``mode=state`` ``chooser_analog_pool`` producer job.

Fast, no real data: a tiny parquet stands in for ``tools/build_chooser_pool.py``
output, ``engine.paths.FEATURES`` is monkeypatched to a tmp dir, and the frozen
state round-trips through the real ``FrozenStateLoader``.
"""
import re

import pandas as pd
import pytest

from engine import paths
from engine.v2.models.chooser_analog_pool import CHOOSER_ANALOG_DIMS
from engine.v2.models.frozen_state import FrozenStateLoader, FrozenStateRef
from engine.v2.ops import training
from tools import phase5_training_job as job
from tools.phase5_prepare_release import CHOOSER_POOL_ID

CUTOFF = "2026-02-13"
_DIMS = {"exp_pnl_sim": 0.5, "width_over_forecast": 1.0, "n_legs": 5.0,
         "anchor_over_spot": 1.0, "rel_spread": 0.2}

_NO_CUTOFF = "chooser_analog_pool needs a cutoff"
_BAD_CUTOFF = "chooser_analog_pool cutoff must be an ISO date (YYYY-MM-DD)"
_MISSING_SOURCE = "chooser analog pool source is missing"
_TOO_LARGE = "chooser analog pool source is too large"
_AT_MOST_ONE = "at most one cutoff, only for paired_residual_pool or chooser_analog_pool"
_EXACTLY_ONE = "chooser_analog_pool needs exactly one cutoff"


def _exact(message):
    return "^" + re.escape(message) + "$"


def _row(strategy, day, pnl):
    return {"strategy": strategy, "exit_date": pd.Timestamp(day), "pnl": float(pnl), **_DIMS}


def _pool_row(day, pnl):
    return (day, pnl, *[_DIMS[name] for name in CHOOSER_ANALOG_DIMS])


def _pool_frame(rows):
    return pd.DataFrame(rows)


def _write_source(tmp_path, frame):
    source = tmp_path / "features"
    source.mkdir(exist_ok=True)
    frame.to_parquet(source / "chooser_analog_pool.parquet", index=False)
    return source


def _mixed_frame():
    # Two strategies; exits on/after the cutoff (02-13, 02-14) must be dropped.
    return _pool_frame([
        _row("CTR5", "2026-02-10", 1.0),
        _row("CTR7", "2026-02-11", 4.0),
        _row("CTR5", "2026-02-12", 2.0),
        _row("CTR5", "2026-02-13", 3.0),
        _row("CTR7", "2026-02-14", 5.0),
    ])


def _params(**changes):
    fields = {"expected_ids": ("training",), "mode": "state", "state": "chooser_analog_pool",
              "cutoffs": (CUTOFF,)}
    fields.update(changes)
    return training.TrainingParameters(**fields)


def _problems(**changes):
    return training.training_parameter_problems(None, _params(**changes))


def test_writes_loadable_pool_excluding_post_cutoff_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    out = tmp_path / "out"

    summary = job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)

    assert summary["status"] == "written"
    assert (out / "chooser_analog_pool.json").is_file()
    assert summary["cutoff"] == CUTOFF
    assert summary["source_rows"] == 5 and summary["n_rows"] == 3
    assert summary["n_strategies"] == 2
    pool = FrozenStateLoader(out).load(
        FrozenStateRef(path=summary["file"], content_hash=summary["content_hash"]))
    assert pool.key == (CHOOSER_POOL_ID, CUTOFF)
    assert pool.rows_for("CTR5") == (_pool_row("2026-02-10", 1.0), _pool_row("2026-02-12", 2.0))
    assert pool.rows_for("CTR7") == (_pool_row("2026-02-11", 4.0),)
    assert pool.rows_for("CTR9") is None


def test_identical_rerun_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    out = tmp_path / "out"

    job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)
    again = job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)

    assert again["status"] == "resumed"


def test_changed_source_refuses_resume_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    out = tmp_path / "out"
    job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)

    changed = _pool_frame([_row("CTR5", "2026-02-10", 99.0), _row("CTR7", "2026-02-11", 4.0),
                           _row("CTR5", "2026-02-12", 2.0)])
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, changed))
    with pytest.raises(SystemExit, match="RESUME_MISMATCH"):
        job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)


def test_missing_cutoff_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    with pytest.raises(SystemExit, match=_exact(_NO_CUTOFF)):
        job.run_state_job("chooser_analog_pool", tmp_path / "out")


@pytest.mark.parametrize("cutoff", ["2026-02-13T12:00", "nope"])
def test_non_iso_cutoff_refuses(tmp_path, monkeypatch, cutoff):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    with pytest.raises(SystemExit, match=_exact(_BAD_CUTOFF)):
        job.run_state_job("chooser_analog_pool", tmp_path / "out", cutoff=cutoff)


def test_missing_source_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", tmp_path / "empty")
    with pytest.raises(SystemExit, match=_exact(_MISSING_SOURCE)):
        job.run_state_job("chooser_analog_pool", tmp_path / "out", cutoff=CUTOFF)


def test_too_large_source_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    monkeypatch.setattr(job, "_CHOOSER_POOL_MAX_BYTES", 10)
    with pytest.raises(SystemExit, match=_exact(_TOO_LARGE)):
        job.run_state_job("chooser_analog_pool", tmp_path / "out", cutoff=CUTOFF)


def test_all_rows_after_cutoff_is_empty(tmp_path, monkeypatch):
    late = _pool_frame([_row("CTR5", "2026-02-13", 3.0), _row("CTR7", "2026-02-14", 5.0)])
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, late))
    out = tmp_path / "out"

    summary = job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)

    assert summary["status"] == "empty"
    assert summary["n_rows"] == 0
    assert not (out / "chooser_analog_pool.json").exists()


def test_cli_refuses_non_iso_cutoff_before_running(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        job.main(["--state", "chooser_analog_pool", "--cutoff", "nope",
                  "--out", str(tmp_path / "out")])
    assert excinfo.value.code == 2
    assert _BAD_CUTOFF in capsys.readouterr().err


def test_state_problems_require_exactly_one_cutoff_for_chooser():
    assert _problems() == []
    assert _problems(cutoffs=()) == [_EXACTLY_ONE]
    assert _problems(cutoffs=(CUTOFF, "2026-02-14")) == [_AT_MOST_ONE, _EXACTLY_ONE]


def test_state_problems_leave_other_states_unchanged():
    assert _problems(state="paired_residual_pool", cutoffs=(CUTOFF,)) == []
    assert _problems(state="paired_residual_pool", cutoffs=()) == []
    assert _problems(state="driver_residual_pool:size", cutoffs=(CUTOFF,)) == [_AT_MOST_ONE]
