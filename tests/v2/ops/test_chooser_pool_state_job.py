"""#1102 slice A: the ``mode=state`` ``chooser_analog_pool`` producer job.

Fast, no real data: a tiny parquet stands in for ``tools/build_chooser_pool.py``
output, ``engine.paths.FEATURES`` is monkeypatched to a tmp dir, and the frozen
state round-trips through the real ``FrozenStateLoader``.
"""
import json
import os
import re

import pandas as pd
import pytest

from engine import paths
from engine.v2.foundation import to_document
from engine.v2.models.chooser_analog_pool import CHOOSER_ANALOG_DIMS
from engine.v2.models.frozen_state import FrozenStateLoader, FrozenStateRef
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.training import chooser_pool_source as pool_source
from engine.v2.models.training.chooser_pool import build_chooser_analog_pool_artifact
from engine.v2.ops import training
from engine.v2.ops.errors import OpsError
from tools import phase5_prepare_release as release
from tools import phase5_training_job as job
from tools.phase5_prepare_release import CHOOSER_POOL_ID

CUTOFF = "2026-02-13"
_DIMS = {"exp_pnl_sim": 0.5, "width_over_forecast": 1.0, "n_legs": 5.0,
         "anchor_over_spot": 1.0, "rel_spread": 0.2}

_NO_CUTOFF = "chooser_analog_pool needs a cutoff"
_BAD_CUTOFF = "chooser_analog_pool cutoff must be an ISO date (YYYY-MM-DD)"
_MISSING_SOURCE = "chooser analog pool source is missing"
_TOO_LARGE = "chooser analog pool source is too large"
_NOT_REGULAR = "chooser analog pool source is not a regular file"
_LACKS_COLUMNS = "chooser analog pool source lacks required columns"
_BUILDER_CUTOFF = "chooser analog pool cutoff must be YYYY-MM-DD"
_AT_MOST_ONE = "at most one cutoff, only for paired_residual_pool or chooser_analog_pool"
_EXACTLY_ONE = "chooser_analog_pool needs exactly one cutoff"
_NOT_ISO = "cutoffs must be valid ISO dates (YYYY-MM-DD)"


def _exact(message):
    return "^" + re.escape(message) + "$"


def _row(strategy, day, pnl, **dims):
    return {"strategy": strategy, "exit_date": pd.Timestamp(day), "pnl": float(pnl),
            **{**_DIMS, **dims}}


def _pool_row(day, pnl, **dims):
    values = {**_DIMS, **dims}
    return (day, pnl, *[values[name] for name in CHOOSER_ANALOG_DIMS])


def _pool_frame(rows):
    return pd.DataFrame(rows)


def _write_source(tmp_path, frame):
    source = tmp_path / "features"
    source.mkdir(exist_ok=True)
    frame.to_parquet(source / "chooser_analog_pool.parquet", index=False)
    return source


def _mixed_frame():
    # Two strategies; exits on/after the cutoff (02-13, 02-14) must be dropped.
    # Input order is deliberately not date order and the rows carry DISTINCT
    # dimension values: the pool must keep the given order, row for row.
    return _pool_frame([
        _row("CTR7", "2026-02-11", 4.0, exp_pnl_sim=4.5),
        _row("CTR5", "2026-02-12", 2.0, exp_pnl_sim=2.5, n_legs=6.0),
        _row("CTR5", "2026-02-13", 3.0),
        _row("CTR5", "2026-02-10", 1.0, exp_pnl_sim=1.5),
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
    persisted = json.loads((out / "chooser_analog_pool.summary.json").read_text())
    assert persisted["status"] == "written"
    assert not [p.name for p in out.iterdir()
                if ".tmp-" in p.name or p.name.endswith(".partial")]
    assert summary["cutoff"] == CUTOFF
    assert summary["source_rows"] == 5 and summary["n_rows"] == 3
    assert summary["n_strategies"] == 2
    pool = FrozenStateLoader(out).load(
        FrozenStateRef(path=summary["file"], content_hash=summary["content_hash"]))
    assert pool.key == (CHOOSER_POOL_ID, CUTOFF)
    # Stored order is the INPUT order within a strategy (02-12 before 02-10),
    # not date order, and the distinct dimension values survived.
    assert pool.rows_for("CTR5") == (_pool_row("2026-02-12", 2.0, exp_pnl_sim=2.5, n_legs=6.0),
                                     _pool_row("2026-02-10", 1.0, exp_pnl_sim=1.5))
    assert pool.rows_for("CTR7") == (_pool_row("2026-02-11", 4.0, exp_pnl_sim=4.5),)
    assert pool.rows_for("CTR9") is None
    assert len(pool.lineage.data) == 1
    assert pool.lineage.data[0].table == CHOOSER_POOL_ID
    assert pool.lineage.data[0].end_exclusive == CUTOFF


def test_identical_rerun_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    out = tmp_path / "out"

    job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)
    written = (out / "chooser_analog_pool.json").read_bytes()
    again = job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)

    assert again["status"] == "resumed"
    assert (out / "chooser_analog_pool.json").read_bytes() == written


def test_changed_source_refuses_resume_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    out = tmp_path / "out"
    job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)

    changed = _pool_frame([_row("CTR5", "2026-02-10", 99.0), _row("CTR7", "2026-02-11", 4.0),
                           _row("CTR5", "2026-02-12", 2.0)])
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, changed))
    with pytest.raises(SystemExit, match="RESUME_MISMATCH"):
        job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)


def test_existing_output_of_other_size_refuses_resume_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    out = tmp_path / "out"
    job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)
    path = out / "chooser_analog_pool.json"
    path.write_bytes(path.read_bytes() + b"x")

    with pytest.raises(SystemExit, match="RESUME_MISMATCH"):
        job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)


def test_existing_output_of_same_size_different_content_refuses_resume_mismatch(
        tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    out = tmp_path / "out"
    job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)
    path = out / "chooser_analog_pool.json"
    tampered = bytearray(path.read_bytes())
    tampered[0] ^= 0xFF  # same size, different content
    path.write_bytes(bytes(tampered))

    with pytest.raises(SystemExit, match="RESUME_MISMATCH"):
        job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)
    assert path.read_bytes() == bytes(tampered)


def test_existing_output_symlink_refuses_resume_mismatch(tmp_path, monkeypatch):
    """A symlinked output is refused even when it links to byte-identical
    content: the bounded resume read is no-follow, so the comparison and the
    file the job would keep can never be two different objects."""
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    out = tmp_path / "out"
    job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)
    path = out / "chooser_analog_pool.json"
    stored = tmp_path / "stored.json"
    path.rename(stored)
    path.symlink_to(stored)

    stored_content = path.read_bytes()
    with pytest.raises(SystemExit, match=_exact(
            "RESUME_MISMATCH: chooser_analog_pool.json exists with different content")):
        job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)
    assert path.is_symlink() and stored.is_file()
    assert stored.read_bytes() == stored_content


def test_prepare_release_reexports_the_runtime_admission():
    """The names stay importable from the preparer while the admission itself
    lives in the no-``checks`` runtime module the job reads."""
    assert release.CHOOSER_POOL_ID is pool_source.CHOOSER_POOL_ID
    assert release.read_chooser_pool_frame is pool_source.read_chooser_pool_frame
    assert release.require_chooser_pool_columns is pool_source.require_chooser_pool_columns


def test_builder_refuses_non_iso_cutoff_and_takes_a_timestamp(tmp_path):
    lineage = Lineage(data=(DataDependency(table=CHOOSER_POOL_ID, end_exclusive=CUTOFF),))
    rows = [_row("CTR5", "2026-02-10", 1.0)]
    for cutoff in ("20260213", "2026-W07-5", "nope"):
        with pytest.raises(ValueError, match=_exact(_BUILDER_CUTOFF)):
            build_chooser_analog_pool_artifact(rows, pool_id=CHOOSER_POOL_ID,
                                               cutoff=cutoff, lineage=lineage)
    # A timestamp/datetime whose first 10 chars ARE an ISO date stays accepted.
    for cutoff in ("2026-02-13T12:00:00", pd.Timestamp("2026-02-13 08:00")):
        artifact = build_chooser_analog_pool_artifact(rows, pool_id=CHOOSER_POOL_ID,
                                                      cutoff=cutoff, lineage=lineage)
        assert artifact.cutoff == CUTOFF


def test_missing_cutoff_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    with pytest.raises(SystemExit, match=_exact(_NO_CUTOFF)):
        job.run_state_job("chooser_analog_pool", tmp_path / "out")


@pytest.mark.parametrize("cutoff", ["2026-02-13T12:00", "nope", "20260213", "2026-W07-5"])
def test_non_iso_cutoff_refuses(tmp_path, monkeypatch, cutoff):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    with pytest.raises(SystemExit, match=_exact(_BAD_CUTOFF)):
        job.run_state_job("chooser_analog_pool", tmp_path / "out", cutoff=cutoff)


def test_iso_cutoff_requires_canonical_date():
    assert job._iso_cutoff(CUTOFF) == CUTOFF
    for value in ("20260213", "2026-W07-5"):
        with pytest.raises(ValueError):
            job._iso_cutoff(value)


def test_missing_source_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", tmp_path / "empty")
    with pytest.raises(SystemExit, match=_exact(_MISSING_SOURCE)):
        job.run_state_job("chooser_analog_pool", tmp_path / "out", cutoff=CUTOFF)


def test_too_large_source_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    monkeypatch.setattr(pool_source, "CHOOSER_POOL_MAX_BYTES", 10)
    with pytest.raises(SystemExit, match=_exact(_TOO_LARGE)):
        job.run_state_job("chooser_analog_pool", tmp_path / "out", cutoff=CUTOFF)


@pytest.mark.parametrize("column", ["strategy", "exit_date", "pnl", *CHOOSER_ANALOG_DIMS])
def test_source_missing_required_column_refuses(tmp_path, monkeypatch, column):
    frame = _mixed_frame().drop(columns=[column])
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, frame))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=_exact(_LACKS_COLUMNS)):
        job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)
    assert not (out / "chooser_analog_pool.json").exists()


def test_size_cap_boundary(tmp_path, monkeypatch):
    source = _write_source(tmp_path, _mixed_frame())
    size = (source / "chooser_analog_pool.parquet").stat().st_size
    monkeypatch.setattr(paths, "FEATURES", source)

    monkeypatch.setattr(pool_source, "CHOOSER_POOL_MAX_BYTES", size)
    assert job.run_state_job("chooser_analog_pool", tmp_path / "out",
                             cutoff=CUTOFF)["status"] == "written"

    monkeypatch.setattr(pool_source, "CHOOSER_POOL_MAX_BYTES", size - 1)
    with pytest.raises(SystemExit, match=_exact(_TOO_LARGE)):
        job.run_state_job("chooser_analog_pool", tmp_path / "other", cutoff=CUTOFF)


def test_prepare_release_reader_refuses_oversized_source(tmp_path, monkeypatch):
    source = _write_source(tmp_path, _mixed_frame())
    monkeypatch.setattr(pool_source, "CHOOSER_POOL_MAX_BYTES", 10)
    with pytest.raises(ValueError, match=_exact(_TOO_LARGE)):
        release.chooser_pool_payloads(source / "chooser_analog_pool.parquet")


def test_reader_missing_source_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        release.read_chooser_pool_frame(tmp_path / "absent.parquet")


def test_reader_reads_regular_source_within_cap(tmp_path, monkeypatch):
    path = _write_source(tmp_path, _mixed_frame()) / "chooser_analog_pool.parquet"
    monkeypatch.setattr(pool_source, "CHOOSER_POOL_MAX_BYTES", path.stat().st_size)
    assert len(release.read_chooser_pool_frame(path)) == 5
    monkeypatch.setattr(pool_source, "CHOOSER_POOL_MAX_BYTES", path.stat().st_size - 1)
    with pytest.raises(ValueError, match=_exact(_TOO_LARGE)):
        release.read_chooser_pool_frame(path)


def test_reader_refuses_symlink_source(tmp_path):
    real = _write_source(tmp_path, _mixed_frame()) / "chooser_analog_pool.parquet"
    link = tmp_path / "link.parquet"
    link.symlink_to(real)
    with pytest.raises(ValueError, match=_exact(_NOT_REGULAR)):
        release.read_chooser_pool_frame(link)
    with pytest.raises(ValueError, match=_exact(_NOT_REGULAR)):
        release.chooser_pool_payloads(link)


def test_reader_refuses_directory_source(tmp_path):
    with pytest.raises(ValueError, match=_exact(_NOT_REGULAR)):
        release.read_chooser_pool_frame(tmp_path)
    with pytest.raises(ValueError, match=_exact(_NOT_REGULAR)):
        release.chooser_pool_payloads(tmp_path)


def test_reader_refuses_fifo_source(tmp_path):
    fifo = tmp_path / "chooser_analog_pool.parquet"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match=_exact(_NOT_REGULAR)):
        release.read_chooser_pool_frame(fifo)
    with pytest.raises(ValueError, match=_exact(_NOT_REGULAR)):
        release.chooser_pool_payloads(fifo)


def test_job_refuses_symlink_source(tmp_path, monkeypatch):
    real = tmp_path / "real.parquet"
    _mixed_frame().to_parquet(real, index=False)
    source = tmp_path / "features"
    source.mkdir()
    (source / "chooser_analog_pool.parquet").symlink_to(real)
    monkeypatch.setattr(paths, "FEATURES", source)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=_exact(_NOT_REGULAR)):
        job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)
    assert not (out / "chooser_analog_pool.json").exists()


def test_empty_rerun_refuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    out = tmp_path / "out"
    job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)

    # Every exit is on/after this cutoff: the pool is empty and an "empty"
    # rerun must not leave the earlier non-empty state file standing.
    with pytest.raises(SystemExit, match=_exact(
            "RESUME_MISMATCH: chooser_analog_pool.json exists but the pool is empty")):
        job.run_state_job("chooser_analog_pool", out, cutoff="2026-01-01")
    assert (out / "chooser_analog_pool.json").is_file()


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


def test_cli_refuses_two_cutoffs_for_chooser_state(tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        job.main(["--state", "chooser_analog_pool", "--cutoff", CUTOFF,
                  "--cutoff", "2026-02-14", "--out", str(tmp_path / "out")])
    assert excinfo.value.code == 2
    assert "--cutoff: at most one" in capsys.readouterr().err


def test_state_problems_require_exactly_one_cutoff_for_chooser():
    assert _problems() == []
    assert _problems(cutoffs=()) == [_EXACTLY_ONE]
    assert _problems(cutoffs=(CUTOFF, "2026-02-14")) == [_AT_MOST_ONE, _EXACTLY_ONE]


def test_state_problems_reject_noncanonical_cutoff():
    assert _problems(cutoffs=("20260213",)) == [_NOT_ISO]
    assert _problems(cutoffs=("2026-W07-5",)) == [_NOT_ISO]


def test_state_problems_leave_other_states_unchanged():
    assert _problems(state="paired_residual_pool", cutoffs=(CUTOFF,)) == []
    assert _problems(state="paired_residual_pool", cutoffs=()) == []
    assert _problems(state="driver_residual_pool:size", cutoffs=(CUTOFF,)) == [_AT_MOST_ONE]


def test_worker_maps_missing_source_to_typed_validation_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", tmp_path / "empty")
    with pytest.raises(OpsError) as err:
        training.run_training_worker(to_document(_params()), tmp_path)
    assert err.value.problem.code == "VALIDATION_FAILED"
    assert err.value.problem.message == "training tool refused the job"
    assert not (tmp_path / "training_result.json").exists()


def test_worker_refuses_two_cutoffs_before_dispatch(tmp_path):
    with pytest.raises(OpsError) as err:
        training.run_training_worker(
            to_document(_params(cutoffs=(CUTOFF, "2026-02-14"))), tmp_path)
    assert err.value.problem.code == "INVALID_REQUEST"
    assert err.value.problem.details == {"problems": [_AT_MOST_ONE, _EXACTLY_ONE]}
    assert not (tmp_path / "training_result.json").exists()
    assert not (tmp_path / "training").exists()


def test_worker_refuses_noncanonical_cutoff_before_dispatch(tmp_path):
    with pytest.raises(OpsError) as err:
        training.run_training_worker(to_document(_params(cutoffs=("20260213",))), tmp_path)
    assert err.value.problem.code == "INVALID_REQUEST"
    assert err.value.problem.details == {"problems": [_NOT_ISO]}
    assert not (tmp_path / "training_result.json").exists()
    assert not (tmp_path / "training").exists()


def test_worker_refuses_paired_pool_noncanonical_cutoff_before_dispatch(tmp_path):
    """The worker admits with the planner's validator for EVERY state, so the
    paired pool's tool-normalized cutoffs are format-gated here exactly as
    ``training_plan`` gates them: no dispatch, no data reads."""
    with pytest.raises(OpsError) as err:
        training.run_training_worker(
            to_document(_params(state="paired_residual_pool", cutoffs=("20260213",))), tmp_path)
    assert err.value.problem.code == "INVALID_REQUEST"
    assert err.value.problem.details == {"problems": [_NOT_ISO]}
    assert not (tmp_path / "training_result.json").exists()
    assert not (tmp_path / "training").exists()


def test_worker_maps_resume_mismatch_to_checkpoint_incompatible(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, _mixed_frame()))
    training.run_training_worker(to_document(_params()), tmp_path)

    changed = _pool_frame([_row("CTR5", "2026-02-10", 99.0)])
    monkeypatch.setattr(paths, "FEATURES", _write_source(tmp_path, changed))
    with pytest.raises(OpsError) as err:
        training.run_training_worker(to_document(_params()), tmp_path)
    assert err.value.problem.code == "CHECKPOINT_INCOMPATIBLE"
    assert err.value.problem.message == "training output already exists with different content"
