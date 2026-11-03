"""S1 monthly retrain training mode: four pool members, receipt and boundary."""
import hashlib
import json
import pickle

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


def _row(strategy, day, pnl, event_date=None):
    return {"strategy": strategy, "event_date": pd.Timestamp(event_date or day),
            "exit_date": pd.Timestamp(day), "pnl": float(pnl), **_DIMS}


def _write_source(tmp_path, rows):
    source = tmp_path / "features"
    source.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(source / "chooser_analog_pool.parquet", index=False)
    return source


def _identity(member):
    if member.startswith("driver_residual_pool:"):
        return {"model_id": "model-" + member, "artifact_sha256": "sha256:" + member}
    return {"source": "features/chooser_analog_pool.parquet",
            "pool_id": "features.chooser_analog_pool", "sha256": "sha256:" + member}


def _expected_member(member):
    entry = {"file": member.replace(":", "__") + ".json",
             "content_hash": "sha256:" + member, "bytes": len(member),
             "cutoff": CUTOFF if member == job.CHOOSER_POOL_STATE else None,
             "source_identity": _identity(member)}
    if member in job.MONTHLY_DRIVER_MEMBERS:
        entry.update({
            "status": "carried_refrozen",
            "model_id": "model-" + member,
            "artifact_sha256": "sha256:" + member,
            "champion_data_date_cutoff": None,
            "admission": {"mode": "inherited_from_champion",
                          "artifact_id": "model-" + member},
        })
    return entry


def _fake_state_job():
    def fake(member, out, *, plan_only=False, **kwargs):
        out.mkdir(parents=True, exist_ok=True)
        name = member.replace(":", "__") + ".json"
        (out / name).write_text(member)
        return {"state": member, "content_hash": "sha256:" + member, "file": name,
                "bytes": len(member), "status": "written",
                "source_identity": _identity(member)}

    return fake


def test_valid_month_and_cutoff_combinations():
    assert _problems() == []
    assert _problems(month="2024-02", cutoffs=("2024-03-01",)) == []
    assert _problems(month="2026-12", cutoffs=("2027-01-01",)) == []


@pytest.mark.parametrize("month", ["", "2026-2", "2026-13", "2026-00", "2026-02-01",
                                   "not-a-month", "9999-12"])
def test_invalid_month_is_refused(month):
    assert _problems(month=month) == ["month must be a canonical YYYY-MM calendar month"]


@pytest.mark.parametrize("changes", [
    {"mode": "recipe", "recipe": "size:*:champion", "cutoffs": ()},
    {"mode": "state", "state": "paired_residual_pool", "cutoffs": ()},
    {"mode": "board_analog", "alpha": 0.5, "cutoffs": (CUTOFF,)},
    {"mode": "trailing_cutoff", "cutoffs": (CUTOFF,)},
])
def test_month_is_refused_for_every_non_monthly_mode(changes):
    assert _problems(**changes) == ["month applies to monthly_pools only"]


def test_plan_rejects_month_for_a_non_monthly_mode():
    with pytest.raises(OpsError) as excinfo:
        training.training_plan(mode="state", state="paired_residual_pool", month=MONTH)
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.problem.details["problems"] == ["month applies to monthly_pools only"]


@pytest.mark.parametrize("cutoff, problems", [
    ("2026-02-28", ["cutoff must be the day after the month's end"]),
    ("2026-03-02", ["cutoff must be the day after the month's end"]),
    ("20260301", ["cutoffs must be valid ISO dates (YYYY-MM-DD)",
                  "cutoff must be the day after the month's end"]),
    ("2026-03-01T00:00:00", ["cutoffs must be valid ISO dates (YYYY-MM-DD)",
                             "cutoff must be the day after the month's end"]),
    ("", ["cutoffs must be valid ISO dates (YYYY-MM-DD)",
          "cutoff must be the day after the month's end"]),
])
def test_cutoff_must_be_the_day_after_the_month_end(cutoff, problems):
    assert _problems(cutoffs=(cutoff,)) == problems


def test_exactly_one_cutoff_is_required():
    assert _problems(cutoffs=()) == ["mode=monthly_pools needs exactly one cutoff"]
    assert _problems(cutoffs=(CUTOFF, "2026-03-02")) == [
        "mode=monthly_pools needs exactly one cutoff"]


@pytest.mark.parametrize("changes, problems", [
    ({"recipe": "size:*:champion"}, ["mode=monthly_pools must not name a recipe"]),
    ({"state": "paired_residual_pool"}, ["mode=monthly_pools must not name a state"]),
    ({"alpha": 0.5}, ["mode=monthly_pools does not take alpha"]),
    ({"strategies": ("STR-THRU",)}, ["mode=monthly_pools does not take strategies"]),
    ({"pairs_path": "data/pairs.parquet"},
     ["mode=monthly_pools does not take pairs_path"]),
    ({"catalog_path": "/tmp/catalog"}, ["native pins require input_mode=native_d14"]),
    ({"objects_root": "/tmp/objects"}, ["native pins require input_mode=native_d14"]),
    ({"snapshot_id": "snap"}, ["native pins require input_mode=native_d14"]),
    ({"receipt_id": "receipt"}, ["native pins require input_mode=native_d14"]),
    ({"input_mode": "native_d14"},
     ["native_d14 requires an implied_t1/runup_move monthly recipe",
      "native_d14 requires catalog_path, objects_root, snapshot_id and receipt_id"]),
])
def test_forbidden_mode_fields_are_refused(changes, problems):
    assert _problems(**changes) == problems


@pytest.mark.parametrize("changes, problems", [
    ({"month": "2026-13"}, ["month must be a canonical YYYY-MM calendar month"]),
    ({"cutoffs": ["2026-03-02"]}, ["cutoff must be the day after the month's end"]),
])
def test_worker_rejects_invalid_month_or_cutoff(tmp_path, changes, problems):
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(_worker_params(**changes), tmp_path / "attempt")
    assert excinfo.value.code == "INVALID_REQUEST"
    assert excinfo.value.problem.details["problems"] == problems


def test_worker_builds_four_members_and_deterministic_receipt(tmp_path, monkeypatch):
    calls = []

    def fake(member, out, *, plan_only=False, **kwargs):
        calls.append((member, kwargs.get("cutoff"), plan_only))
        out.mkdir(parents=True, exist_ok=True)
        name = member.replace(":", "__") + ".json"
        (out / name).write_text(member)
        identity = ({"model_id": "model-" + member, "artifact_sha256": "sha256:" + member}
                    if member.startswith("driver_residual_pool:") else
                    {"source": "features/chooser_analog_pool.parquet",
                     "pool_id": "features.chooser_analog_pool", "sha256": "sha256:" + member})
        return {"state": member, "content_hash": "sha256:" + member, "file": name,
                "bytes": len(member), "status": "written", "source_identity": identity}

    monkeypatch.setattr(job, "run_state_job", fake)
    root = tmp_path / "attempt"
    training.run_training_worker(_worker_params(), root)

    assert calls == [
        ("driver_residual_pool:size", None, False),
        ("driver_residual_pool:implied_t1", None, False),
        ("driver_residual_pool:runup_move", None, False),
        ("chooser_analog_pool", CUTOFF, False),
    ]
    receipt = json.loads((root / "training_result.json").read_text())
    assert receipt["mode"] == "monthly_pools"
    assert receipt["month"] == MONTH
    assert receipt["retrain_cutoff"] == CUTOFF
    assert receipt["experiment_inputs"] == "none"
    assert sorted(receipt["members"]) == sorted(job.MONTHLY_POOL_MEMBERS)
    assert receipt["members"] == {
        member: _expected_member(member) for member in job.MONTHLY_POOL_MEMBERS}

    first = (root / "training_result.json").read_bytes()
    training.run_training_worker(_worker_params(), root)
    assert (root / "training_result.json").read_bytes() == first

    other_root = tmp_path / "other"
    training.run_training_worker(
        _worker_params(month="2026-03", cutoffs=["2026-04-01"]), other_root)
    other = json.loads((other_root / "training_result.json").read_text())
    assert other["month"] == "2026-03"
    assert other["retrain_cutoff"] == "2026-04-01"
    assert other["members"][job.CHOOSER_POOL_STATE]["cutoff"] == "2026-04-01"
    assert other != receipt


def test_driver_receipts_record_inherited_champion_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "run_state_job", _fake_state_job())
    root = tmp_path / "attempt"
    training.run_training_worker(_worker_params(), root)

    receipt = json.loads((root / "training_result.json").read_text())
    for member in job.MONTHLY_DRIVER_MEMBERS:
        entry = receipt["members"][member]
        assert entry["status"] == "carried_refrozen"
        assert entry["source_identity"] == _identity(member)
        assert entry["champion_data_date_cutoff"] is None
        assert entry["admission"] == {
            "mode": "inherited_from_champion", "artifact_id": "model-" + member}
        assert entry["model_id"] == "model-" + member
        assert entry["artifact_sha256"] == "sha256:" + member


def test_chooser_event_date_boundary_keeps_m_end_and_drops_m_end_plus_one(
        tmp_path, monkeypatch):
    source = _write_source(tmp_path, [
        _row("CTR5", "2026-02-27", 1.0, event_date="2026-02-28"),
        _row("CTR5", "2026-02-27", 2.0, event_date="2026-03-01"),
    ])
    monkeypatch.setattr(paths, "FEATURES", source)
    out = tmp_path / "out"

    summary = job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF, monthly=True)

    assert summary["n_rows"] == 1
    pool = FrozenStateLoader(out).load(
        FrozenStateRef(path=summary["file"], content_hash=summary["content_hash"]))
    dims = [_DIMS[name] for name in CHOOSER_ANALOG_DIMS]
    assert pool.rows_for("CTR5") == (("2026-02-27", 1.0, *dims),)


def test_chooser_boundary_includes_m_end_and_excludes_next_day(tmp_path, monkeypatch):
    source = _write_source(tmp_path, [
        _row("CTR5", "2026-02-27", 1.0),
        _row("CTR5", "2026-02-28", 2.0),
        _row("CTR5", "2026-03-01", 3.0),
    ])
    monkeypatch.setattr(paths, "FEATURES", source)
    out = tmp_path / "out"

    summary = job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF, monthly=True)

    assert summary["n_rows"] == 2
    pool = FrozenStateLoader(out).load(
        FrozenStateRef(path=summary["file"], content_hash=summary["content_hash"]))
    dims = [_DIMS[name] for name in CHOOSER_ANALOG_DIMS]
    assert pool.rows_for("CTR5") == (
        ("2026-02-27", 1.0, *dims),
        ("2026-02-28", 2.0, *dims),
    )


def test_chooser_boundary_filters_event_date_too(tmp_path, monkeypatch):
    source = _write_source(tmp_path, [
        _row("CTR5", "2026-02-27", 1.0, event_date="2026-02-20"),
        _row("CTR5", "2026-02-27", 2.0, event_date="2026-03-02"),
    ])
    monkeypatch.setattr(paths, "FEATURES", source)
    out = tmp_path / "out"

    summary = job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF, monthly=True)

    assert summary["n_rows"] == 1
    pool = FrozenStateLoader(out).load(
        FrozenStateRef(path=summary["file"], content_hash=summary["content_hash"]))
    dims = [_DIMS[name] for name in CHOOSER_ANALOG_DIMS]
    assert pool.rows_for("CTR5") == (("2026-02-27", 1.0, *dims),)


def test_chooser_source_identity_hashes_the_exact_bounded_bytes(tmp_path, monkeypatch):
    source = _write_source(tmp_path, [_row("CTR5", "2026-02-27", 1.0)])
    monkeypatch.setattr(paths, "FEATURES", source)
    out = tmp_path / "out"

    summary = job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)

    path = source / "chooser_analog_pool.parquet"
    assert summary["source_identity"] == {
        "source": "features/chooser_analog_pool.parquet",
        "pool_id": "features.chooser_analog_pool",
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def test_chooser_identity_ignores_the_staged_feature_root(tmp_path, monkeypatch):
    rows = [_row("CTR5", "2026-02-27", 1.0)]
    (tmp_path / "attempt-a").mkdir()
    (tmp_path / "attempt-b").mkdir()
    first = _write_source(tmp_path / "attempt-a", rows)
    second = _write_source(tmp_path / "attempt-b", rows)
    payload = (first / "chooser_analog_pool.parquet").read_bytes()
    assert payload == (second / "chooser_analog_pool.parquet").read_bytes()

    monkeypatch.setattr(paths, "FEATURES", first)
    one = job.run_state_job("chooser_analog_pool", tmp_path / "out-a", cutoff=CUTOFF)
    monkeypatch.setattr(paths, "FEATURES", second)
    two = job.run_state_job("chooser_analog_pool", tmp_path / "out-b", cutoff=CUTOFF)

    expected = {"source": "features/chooser_analog_pool.parquet",
                "pool_id": "features.chooser_analog_pool",
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest()}
    assert one["source_identity"] == two["source_identity"] == expected


def _run_with_real_chooser(tmp_path, monkeypatch, source):
    real = job.run_state_job

    def dispatch(member, out, *, plan_only=False, **kwargs):
        if member == job.CHOOSER_POOL_STATE:
            return real(member, out, plan_only=plan_only, **kwargs)
        out.mkdir(parents=True, exist_ok=True)
        return {"state": member, "content_hash": "sha256:" + member,
                "file": member.replace(":", "__") + ".json", "bytes": 1,
                "status": "written",
                "source_identity": {"model_id": "model-" + member,
                                    "artifact_sha256": "sha256:" + member}}

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


def test_unreadable_source_is_validation_failed(tmp_path, monkeypatch):
    source = tmp_path / "features"
    source.mkdir()

    def unreadable(_path, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(chooser_pool_source, "read_chooser_pool_frame", unreadable)
    assert _run_with_real_chooser(tmp_path, monkeypatch, source).code == "VALIDATION_FAILED"


@pytest.mark.parametrize("field", ["event_date", "exit_date"])
def test_malformed_chooser_date_is_validation_failed(tmp_path, monkeypatch, field):
    row = _row("CTR5", "2026-02-27", 1.0)
    row[field] = "not-a-date"
    source = _write_source(tmp_path, [row])
    assert _run_with_real_chooser(tmp_path, monkeypatch, source).code == "VALIDATION_FAILED"


def test_corrupt_chooser_parquet_is_validation_failed(tmp_path, monkeypatch):
    source = tmp_path / "features"
    source.mkdir()
    (source / "chooser_analog_pool.parquet").write_bytes(b"not a parquet file")
    assert _run_with_real_chooser(tmp_path, monkeypatch, source).code == "VALIDATION_FAILED"


def test_invalid_driver_champion_source_is_validation_failed(tmp_path, monkeypatch):
    from engine.models.registry import RegistryError

    def broken(*_args, **_kwargs):
        raise RegistryError("no champion")

    monkeypatch.setattr("engine.models.registry.load_registry", broken)
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(_worker_params(), tmp_path / "attempt")
    assert excinfo.value.code == "VALIDATION_FAILED"


def _champion_registry(tmp_path, monkeypatch):
    """A one-champion registry whose artifact is a small file under ``tmp_path``."""
    from engine.models import registry as registry_mod

    payload = b"x" * 64
    artifact = tmp_path / "size-champion.joblib"
    artifact.write_bytes(payload)
    entry = registry_mod.RegistryEntry(
        id="size-champion", role="size", strategy="*", artifact=str(artifact),
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        features=["abs_move"], target="abs_move", train_window="2015-2025",
        champion=True)
    registry = registry_mod.Registry(entries=[entry])
    monkeypatch.setattr(registry_mod, "load_registry", lambda *args, **kwargs: registry)
    return registry


@pytest.mark.parametrize("max_bytes", [None, 1024])
def test_champion_digest_mismatch_refuses_before_deserialization(
        tmp_path, monkeypatch, max_bytes):
    from engine.models import registry as registry_mod

    registry = _champion_registry(tmp_path, monkeypatch)
    registry.entries[0].artifact_sha256 = "0" * 64
    calls = []

    def deserializer(*_args, **_kwargs):
        calls.append("deserialized")
        raise AssertionError("the deserializer must not run for a digest mismatch")

    monkeypatch.setattr(registry_mod, "load_artifact", deserializer)
    with pytest.raises(registry_mod.RegistryError, match="hash mismatch"):
        registry.load(registry.entries[0], max_bytes=max_bytes)
    assert calls == []


@pytest.mark.parametrize("max_bytes", [-1, 1.5, "8", True])
def test_champion_load_refuses_an_invalid_size_bound(tmp_path, monkeypatch, max_bytes):
    from engine.models import registry as registry_mod

    registry = _champion_registry(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="nonnegative integer"):
        registry.load(registry.entries[0], max_bytes=max_bytes)


@pytest.mark.parametrize("error", [
    ValueError("private detail"),
    pickle.UnpicklingError("private detail"),
    ImportError("private detail"),
])
def test_corrupt_champion_artifact_is_validation_failed_without_exception_text(
        tmp_path, monkeypatch, error):
    from engine.models import registry as registry_mod

    _champion_registry(tmp_path, monkeypatch)

    def corrupt(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(registry_mod, "load_artifact", corrupt)
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(_worker_params(), tmp_path / "attempt")
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.message == "training tool refused the job"
    assert "private detail" not in str(excinfo.value)


def test_oversized_champion_is_rejected_before_the_deserializer_runs(tmp_path, monkeypatch):
    from engine.models import registry as registry_mod

    _champion_registry(tmp_path, monkeypatch)
    calls = []

    def deserializer(*_args, **_kwargs):
        calls.append("deserialized")
        raise AssertionError("the deserializer must not run for an oversized artifact")

    monkeypatch.setattr(registry_mod, "load_artifact", deserializer)
    monkeypatch.setattr(job, "MONTHLY_CHAMPION_ARTIFACT_MAX_BYTES", 8)
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(_worker_params(), tmp_path / "attempt")
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert calls == []


def test_unreadable_champion_source_is_validation_failed(tmp_path, monkeypatch):
    from engine.models import registry as registry_mod

    _champion_registry(tmp_path, monkeypatch)

    def unreadable(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(registry_mod, "_artifact_fd", unreadable)
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(_worker_params(), tmp_path / "attempt")
    assert excinfo.value.code == "VALIDATION_FAILED"


def test_empty_required_member_is_validation_failed(tmp_path, monkeypatch):
    def empty(member, out, *, plan_only=False, **kwargs):
        out.mkdir(parents=True, exist_ok=True)
        return {"state": member, "status": "empty"}

    monkeypatch.setattr(job, "run_state_job", empty)
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(_worker_params(), tmp_path / "attempt")
    assert excinfo.value.code == "VALIDATION_FAILED"


def test_monthly_retrain_cutoff_rolls_over_the_year_end():
    assert job.monthly_retrain_cutoff("2026-12") == "2027-01-01"
    assert job.monthly_retrain_cutoff("2024-02") == "2024-03-01"


def test_legacy_chooser_state_accepts_a_source_without_event_date(tmp_path, monkeypatch):
    rows = [{"strategy": "CTR5", "exit_date": pd.Timestamp(day), "pnl": float(pnl), **_DIMS}
            for day, pnl in (("2026-02-27", 1.0), ("2026-03-01", 3.0))]
    source = _write_source(tmp_path, rows)
    monkeypatch.setattr(paths, "FEATURES", source)
    out = tmp_path / "out"

    summary = job.run_state_job("chooser_analog_pool", out, cutoff=CUTOFF)

    assert summary["n_rows"] == 1
    pool = FrozenStateLoader(out).load(
        FrozenStateRef(path=summary["file"], content_hash=summary["content_hash"]))
    dims = [_DIMS[name] for name in CHOOSER_ANALOG_DIMS]
    assert pool.rows_for("CTR5") == (("2026-02-27", 1.0, *dims),)


def test_monthly_chooser_refuses_a_source_without_event_date(tmp_path, monkeypatch):
    rows = [{"strategy": "CTR5", "exit_date": pd.Timestamp("2026-02-27"), "pnl": 1.0,
             **_DIMS}]
    source = _write_source(tmp_path, rows)
    assert _run_with_real_chooser(tmp_path, monkeypatch, source).code == "VALIDATION_FAILED"


def test_tool_refuses_a_cutoff_that_is_not_the_day_after_the_month_end(tmp_path):
    with pytest.raises(SystemExit, match="day after the month's end"):
        job.run_monthly_pools_job(tmp_path, month=MONTH, cutoff="2026-03-02")


def test_monthly_plan_pins_the_manifest_and_non_monthly_plans_omit_month():
    manifest = "sha256:" + "a" * 64
    plan = training.training_plan(
        mode="monthly_pools", month=MONTH, cutoffs=(CUTOFF,), manifest_ref=manifest)
    assert plan["parameters"]["month"] == MONTH
    assert plan["parameters"]["input_bindings"] == {"legacy_manifest.json": manifest}
    assert plan["input_refs"] == [manifest]
    assert plan["blocked_prerequisites"] == []

    other = training.training_plan(
        mode="state", state="paired_residual_pool", manifest_ref=manifest)
    assert "month" not in other["parameters"]
    assert other["input_refs"] == [manifest]
