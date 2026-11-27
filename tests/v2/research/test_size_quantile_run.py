"""EXP-190 Q6c: the real-data entry point and smoke mode, on synthetic data only (no data/, no catalog, no ledger writes).
Results are checked against literal values, hashlib and content_hash written in this file, never the code under test."""
import contextlib
import hashlib
import inspect
import json
import time
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import lib
from experiments import size_pnl_tables
from experiments import size_quantile_run as run_mod
from tests.v2.research import test_size_quantile_stage_a as base

CFG = {"catalog": "c.db", "store_root": "store", "spec": "spec.json", "plan_ref": "ref.json", "snapshot_id": "SNAP",
       "crush_snapshot_id": "CRUSH", "production_hash": base.TABLE_HASH}
YEAR_Y0 = {"y0": "2022", "sealed_binding": "Y0B", "chain_share_min": 0.5, "training_years_min": 2, "years": []}
ALL_YEARS = ["2022", "2023", "2024", "2025"]


class _Conn:
    closed = False

    def close(self):
        self.closed = True


def _no_config(_path):
    raise AssertionError("the sources config must not be read")


def _world():
    """The stage-A world with a ticker on every sealed event (the sealed events carry only event_id in base)."""
    world = base._world()
    tickers = {e["id"]: e["ticker"] for e in world["events"]}
    for event in world["sealed"]["events"]:
        event["ticker"] = tickers[event["event_id"]]
    return world


def _paths(guard):
    return {"spec_path": guard["spec_path"], "ledger_path": guard["ledger_path"],
            "run_log_path": guard["run_log_path"], "experiments_dir": guard["experiments_dir"]}


def _inputs(world, out, calls=None, avail=None):
    """What the real load_inputs returns for this world; fetch appends each call to calls (when given) and takes
    label availability overrides from avail."""
    return {"sealed": world["sealed"], "admitted": world["admitted"],
            "fetch": base._fake_fetch(world, out, [] if calls is None else calls, avail=avail),
            "production": {"binding": base.PRODUCTION_BINDING, "table_hash": base.TABLE_HASH},
            "table_hash": base.TABLE_HASH}


def _env(monkeypatch, world, out, *, y0=None, regress=None, calls=None, avail=None):
    base._install(monkeypatch, world, regress=regress)
    monkeypatch.setattr(run_mod.sources, "_config", lambda _path: CFG)
    monkeypatch.setattr(run_mod.sources, "_open", lambda _cfg: (_Conn(), object()))
    monkeypatch.setattr(run_mod, "load_inputs",
                        lambda _cfg, _conn, _store: _inputs(world, out, calls=calls, avail=avail))
    monkeypatch.setattr(run_mod, "y0_report", y0 or (lambda _sealed, _spec: dict(YEAR_Y0)))


def _recorder(calls, name, value, boom=None, error=None):
    """A fake collaborator: records its name, and raises `error` when it is the one named by `boom`."""
    def fake(*args, **kwargs):
        if name is not None:
            calls.append(name)
        if name is not None and name == boom:
            raise error
        return value(*args, **kwargs) if callable(value) else value
    return fake


def _patch_load(monkeypatch, calls, *, boom=None, error=None, snapshot_id="SNAP", fetch=None):
    sealed = {"snapshot": SimpleNamespace(snapshot_id=snapshot_id), "repository": "REPO", "as_of_month": "2025-12-01",
              "plan": {"plan_id": "P", "anchor": {"run_id": "RUN"}}, "binding": "SB"}
    table = SimpleNamespace(fetch=fetch or (lambda columns, ids: pd.DataFrame()), table_hash=base.TABLE_HASH)
    steps = [(run_mod.sources, "_read_json", None, {}), (run_mod, "experiment_spec_from_document", None, "SPEC"),
             (run_mod, "from_document", None, "REF"), (run_mod, "load_sealed_panel", "sealed", sealed),
             (run_mod.sources, "_bound_snapshot", "bound", None), (run_mod.sources, "_snapshot", "snapshot", "CRUSH"),
             (run_mod, "admit_events", "admit", "ADMIT"), (size_pnl_tables, "tables", "tables", table),
             (run_mod, "read_production", "production", {"binding": "PB", "table_hash": base.TABLE_HASH})]
    for module, attr, name, value in steps:
        monkeypatch.setattr(module, attr, _recorder(calls, name, value, boom, error))


ORDER = ["sealed", "bound", "snapshot", "admit", "tables", "production"]


def test_real_load_inputs_runs_in_spec_order(monkeypatch):
    calls, reads = [], []

    def fetch(columns, ids):
        reads.append((columns, ids))
        return pd.DataFrame()
    _patch_load(monkeypatch, calls, fetch=fetch)
    out = run_mod.load_inputs(CFG, _Conn(), object())
    assert calls == ORDER
    assert reads == []  # loading reads no table rows itself here, so no label is read
    assert set(out) == {"sealed", "admitted", "fetch", "production", "table_hash"}
    assert "b3" not in out


@pytest.mark.parametrize("boom, reached", [
    ("load_sealed_panel", ["sealed"]),
    ("read_production", ORDER[:6]),
])
def test_a_refusal_at_each_gate_propagates_unchanged_and_stops_the_order(monkeypatch, boom, reached):
    calls, error = [], fail("EXPERIMENT_IDENTITY_CONFLICT", f"{boom} refused", details={"reason": boom})
    _patch_load(monkeypatch, calls, boom={"load_sealed_panel": "sealed", "read_production": "production"}[boom],
                error=error)
    with pytest.raises(OpsError) as exc:
        run_mod.load_inputs(CFG, _Conn(), object())
    assert exc.value is error
    assert calls == reached


def test_snapshot_mismatch_refuses_before_admission(monkeypatch):
    calls = []
    _patch_load(monkeypatch, calls, snapshot_id="OTHER")
    with pytest.raises(OpsError) as exc:
        run_mod.load_inputs(CFG, _Conn(), object())
    assert base._refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_mismatch")
    assert calls == ["sealed"]


def test_fetch_memo_reads_each_key_once_and_returns_copies(monkeypatch):
    reads = []

    def underlying(columns, ids):
        reads.append((tuple(columns), tuple(ids)))
        return pd.DataFrame({"event_id": ["x"], "a": [1.0]})
    _patch_load(monkeypatch, [], fetch=underlying)
    fetch = run_mod.load_inputs(CFG, _Conn(), object())["fetch"]
    first = fetch(["a"], ["x"])
    first.loc[0, "a"] = 99.0
    assert fetch(["a"], ["x"])["a"].tolist() == [1.0]
    fetch(["a"], ["y"])
    assert reads == [(("a",), ("x",)), (("a",), ("y",))]


@pytest.mark.parametrize("tickers", [None, 3])
def test_unsettled_spec_refuses_before_the_config_is_read(monkeypatch, tmp_path, tickers):
    guard = base._guard(tmp_path, settled=False)
    monkeypatch.setattr(run_mod.sources, "_config", _no_config)
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", tmp_path / "out", tickers, **_paths(guard))
    assert base._refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "stage_b_pending")
    assert not (tmp_path / "out").exists()


def test_full_run_writes_run_inputs_before_the_first_fit(monkeypatch, tmp_path):
    world, guard, out = _world(), base._guard(tmp_path), tmp_path / "out"
    seen, real, calls = {}, base._fake_fit(), []

    def regress(recipe, train_x, y, eval_x, *, columns):
        seen.setdefault("run_inputs_exists", (out / "run_inputs.json").exists())
        return real(recipe, train_x, y, eval_x, columns=columns)
    _env(monkeypatch, world, out, regress=regress, calls=calls)
    result = run_mod.run("cfg.json", out, **_paths(guard))
    run_inputs = json.loads((out / "run_inputs.json").read_text(encoding="utf-8"))
    report = json.loads((out / base.REPORT).read_text(encoding="utf-8"))
    assert seen == {"run_inputs_exists": True}
    assert result == {"mode": "full", "status": report["status"], "reasons": report["reasons"],
                      "content_hash": report["content_hash"], "report": str(out / base.REPORT)}
    assert run_inputs["schema_version"] == "size_quantile_run_inputs.v1" and run_inputs["mode"] == "full"
    assert run_inputs["years"] == ALL_YEARS and run_inputs["smoke_tickers"] is None
    assert run_inputs["y0"] == YEAR_Y0
    assert "b3" not in run_inputs and "b0_labels_hash" not in run_inputs
    assert run_inputs["content_hash"] == content_hash({k: v for k, v in run_inputs.items() if k != "content_hash"})
    assert report["inputs"]["run_inputs_hash"] == run_inputs["content_hash"]
    assert run_inputs["spec_hash"] == lib.spec_hash(yaml.safe_load(guard["spec_path"].read_text(encoding="utf-8")))
    assert run_inputs["plan_id"] == world["sealed"]["plan"]["plan_id"]
    assert run_inputs["production_binding"] == base.PRODUCTION_BINDING
    assert run_inputs["production_table_hash"] == base.TABLE_HASH
    assert run_inputs["sealed_binding"] == base.BINDING
    assert "y0_check" in report["not_included"]
    labels = world["labels"]
    b0 = content_hash(sorted([e, float(labels.at[e, "abs_move"]), float(labels.at[e, "crush_pct_iv30"]),
                              (date.fromisoformat(world["dates"][e]) + timedelta(days=1)).isoformat()]
                             for e in labels.index))
    assert report["inputs"]["b0_labels_hash"] == b0
    label_calls = base._label_fetches(calls)
    assert label_calls
    assert all(c["fits"] for c in label_calls)  # every label read happens after every fit_<year>.json exists
    assert sorted(p.name for p in out.iterdir()) == ["fits", "run_inputs.json", base.REPORT]


def test_a_bad_b0_label_refuses_only_after_the_fits_and_publishes_no_report(monkeypatch, tmp_path):
    world, guard, out = _world(), base._guard(tmp_path), tmp_path / "out"
    event = world["labels"].index[0]
    _env(monkeypatch, world, out, avail={event: world["dates"][event]})
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", out, **_paths(guard))
    assert base._refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "label_availability_missing")
    assert (out / "run_inputs.json").exists()
    assert all((out / "fits" / f"fit_{year}.json").exists() for year in ALL_YEARS)
    assert not (out / base.REPORT).exists()


def test_y0_refusal_starts_no_fit_and_writes_nothing(monkeypatch, tmp_path):
    world, guard, out = _world(), base._guard(tmp_path), tmp_path / "out"

    def y0_none(_sealed, _spec):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "no Y0", details={"reason": "y0_none"})

    def no_fit(*_args, **_kwargs):
        raise AssertionError("no fit may run")
    _env(monkeypatch, world, out, y0=y0_none)
    monkeypatch.setattr(run_mod, "fit_walk_forward", no_fit)
    monkeypatch.setattr(run_mod, "run_stage_a", no_fit)
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", out, **_paths(guard))
    assert base._refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "y0_none")
    assert not out.exists()


def test_smoke_selects_the_three_smallest_hash_tickers_and_writes_no_stage_a_report(monkeypatch, tmp_path):
    world, guard, out = _world(), base._guard(tmp_path), tmp_path / "out"
    all_tickers = sorted({e["ticker"] for e in world["events"]})
    expected = sorted(all_tickers, key=lambda t: (hashlib.sha256(f"{t}|exp190-smoke".encode()).hexdigest(), t))[:3]

    def no_stage_a(*_args, **_kwargs):
        raise AssertionError("smoke must not run stage A")
    calls = []
    _env(monkeypatch, world, out, calls=calls)
    monkeypatch.setattr(run_mod, "run_stage_a", no_stage_a)
    ledger_before = guard["ledger_path"].read_bytes()
    run_mod.run("cfg.json", out, 3, **_paths(guard))
    assert base._label_fetches(calls) == []
    run_inputs = json.loads((out / "run_inputs.json").read_text(encoding="utf-8"))
    smoke = json.loads((out / "smoke_record.json").read_text(encoding="utf-8"))
    assert run_inputs["mode"] == "smoke" and run_inputs["smoke_tickers"] == expected
    assert run_inputs["sealed_binding"] == content_hash({"sealed": base.BINDING, "smoke_tickers": expected})
    assert run_inputs["sealed_binding"] != base.BINDING
    assert smoke["schema_version"] == "size_quantile_smoke.v1" and smoke["tickers"] == 3
    assert smoke["run_inputs_hash"] == run_inputs["content_hash"] and smoke["peak_rss_bytes"] > 0
    assert set(smoke) == {"schema_version", "run_inputs_hash", "events", "tickers", "fit_seconds",
                          "peak_rss_bytes", "fit_status", "fit_receipt_hashes", "content_hash"}
    assert smoke["events"] == sum(e["ticker"] in expected for e in world["events"])
    assert smoke["fit_status"]
    for year, status in smoke["fit_status"].items():
        receipt = json.loads((out / "fits" / f"fit_{year}.json").read_text(encoding="utf-8"))
        assert smoke["fit_status"][year] == receipt["status"] == status
        assert smoke["fit_receipt_hashes"][year] == receipt["content_hash"]
    assert run_inputs["spec_hash"] == lib.spec_hash(yaml.safe_load(guard["spec_path"].read_text(encoding="utf-8")))
    assert not (out / base.REPORT).exists() and (out / "fits").is_dir()
    kept = {e["id"] for e in world["events"] if e["ticker"] in expected}
    assert (out / "fits" / "fit_2022.json").exists()
    for year in ALL_YEARS:
        path = out / "fits" / f"fit_{year}.json"
        if path.exists():  # the walk stops at the first fit_failed year
            assert set(json.loads(path.read_text(encoding="utf-8"))["event_ids"]) <= kept
    kept_2022 = {e["id"] for e in world["events"] if e["ticker"] in expected and e["date"].startswith("2022")}
    receipt_2022 = json.loads((out / "fits" / "fit_2022.json").read_text(encoding="utf-8"))
    assert receipt_2022["event_ids"] and set(receipt_2022["event_ids"]) == kept_2022
    assert guard["ledger_path"].read_bytes() == ledger_before


@pytest.mark.parametrize("tickers", [0, True])
def test_invalid_smoke_tickers_refuse_before_the_config_is_read(monkeypatch, tmp_path, tickers):
    guard = base._guard(tmp_path)
    monkeypatch.setattr(run_mod.sources, "_config", _no_config)
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", tmp_path / "out", tickers, **_paths(guard))
    assert base._refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "smoke_tickers_invalid")


def test_smoke_into_a_non_empty_directory_refuses_before_the_config_is_read(monkeypatch, tmp_path):
    guard, out = base._guard(tmp_path), tmp_path / "out"
    out.mkdir()
    (out / "old.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(run_mod.sources, "_config", _no_config)
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", out, 3, **_paths(guard))
    assert base._refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "smoke_out_dir_not_fresh")


@pytest.mark.parametrize("kind", ["file", "file_parent", "dangling_link"])
@pytest.mark.parametrize("tickers", [None, 3])
def test_out_dir_that_cannot_be_a_directory_refuses_before_the_config_is_read(
        monkeypatch, tmp_path, tickers, kind):
    guard = base._guard(tmp_path)
    if kind == "file":
        out, kept = tmp_path / "out", tmp_path / "out"
        out.write_text("x", encoding="utf-8")
    elif kind == "file_parent":
        kept = tmp_path / "plain"
        kept.write_text("x", encoding="utf-8")
        out = kept / "out"
    else:
        out = tmp_path / "out"
        out.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(run_mod.sources, "_config", _no_config)
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", out, tickers, **_paths(guard))
    assert base._refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "out_dir_not_directory")
    if kind == "dangling_link":
        assert not (tmp_path / "missing").exists()
    else:
        assert kept.read_text(encoding="utf-8") == "x"


def test_watchdog_breach_refuses_resource_limit_and_publishes_no_report(monkeypatch, tmp_path):
    world, guard, out = _world(), base._guard(tmp_path), tmp_path / "out"
    _env(monkeypatch, world, out)

    def busy(_cfg, _conn, _store):
        end = time.monotonic() + 5
        while time.monotonic() < end:
            pass
    monkeypatch.setattr(run_mod, "load_inputs", busy)
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", out, cap_bytes=1, **_paths(guard))
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not (out / base.REPORT).exists()


def test_memory_error_in_a_smoke_fit_refuses_resource_limit_with_no_record(monkeypatch, tmp_path):
    world, guard, out = _world(), base._guard(tmp_path), tmp_path / "out"
    _env(monkeypatch, world, out)

    def out_of_memory(*_args):
        raise MemoryError
    monkeypatch.setattr(run_mod, "fit_walk_forward", out_of_memory)
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", out, 3, **_paths(guard))
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not (out / "smoke_record.json").exists()


def test_default_cap_is_eight_gib_and_run_passes_it_to_the_watchdog(monkeypatch, tmp_path):
    assert run_mod.CAP_BYTES == 8 * 2**30 == inspect.signature(run_mod.run).parameters["cap_bytes"].default
    seen = []

    @contextlib.contextmanager
    def recording(limit_bytes, *_args, **_kwargs):
        seen.append(limit_bytes)
        yield
    world, guard, out = _world(), base._guard(tmp_path), tmp_path / "out"
    _env(monkeypatch, world, out)
    monkeypatch.setattr(run_mod, "rss_watchdog", recording)
    run_mod.run("cfg.json", out, **_paths(guard))
    assert seen == [8 * 2**30]


def test_main_refusal_prints_the_code_to_stderr_and_returns_1(monkeypatch, capsys):
    def refuse(*_args, **_kwargs):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "spec changed", details={"reason": "spec_changed"})
    monkeypatch.setattr(run_mod, "run", refuse)
    assert run_mod.main(["--sources-config", "c.json", "--out-dir", "o"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("refused: EXPERIMENT_IDENTITY_CONFLICT")


def test_main_success_prints_one_json_line_and_returns_0(monkeypatch, capsys):
    monkeypatch.setattr(run_mod, "run", lambda *_args: {"mode": "smoke", "content_hash": "sha256:x"})
    assert run_mod.main(["--sources-config", "c.json", "--out-dir", "o", "--tickers", "3"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and json.loads(lines[0]) == {"mode": "smoke", "content_hash": "sha256:x"}


def test_main_accepts_no_ledger_as_a_noop_and_passes_the_same_args(monkeypatch):
    recorded = []
    monkeypatch.setattr(run_mod, "run", lambda *args: recorded.append(args) or {"mode": "smoke"})
    assert run_mod.main(["--sources-config", "c.json", "--out-dir", "o", "--tickers", "3", "--no-ledger"]) == 0
    assert recorded == [("c.json", "o", 3)]


def test_an_altered_run_inputs_identity_refuses_on_rerun_and_is_left_untouched(monkeypatch, tmp_path):
    world, guard, out = _world(), base._guard(tmp_path), tmp_path / "out"
    _env(monkeypatch, world, out)
    run_mod.run("cfg.json", out, **_paths(guard))
    path = out / "run_inputs.json"
    planted = json.loads(path.read_text(encoding="utf-8"))
    planted["spec_hash"] = "sha256:planted"
    path.write_text(json.dumps(planted), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", out, **_paths(guard))
    assert base._refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "run_inputs_changed")
    assert path.read_bytes() == before


def test_an_altered_smoke_record_is_never_overwritten_by_a_rerun(monkeypatch, tmp_path):
    world, guard, out = _world(), base._guard(tmp_path), tmp_path / "out"
    _env(monkeypatch, world, out)
    run_mod.run("cfg.json", out, 3, **_paths(guard))
    path = out / "smoke_record.json"
    planted = json.loads(path.read_text(encoding="utf-8"))
    planted["run_inputs_hash"] = "sha256:planted"
    path.write_text(json.dumps(planted), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", out, 3, **_paths(guard))
    assert base._refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "smoke_out_dir_not_fresh")
    assert path.read_bytes() == before


def test_stage_a_refuses_a_spec_that_differs_from_the_one_bound_in_run_inputs(monkeypatch, tmp_path):
    world, guard, out = _world(), base._guard(tmp_path), tmp_path / "out"
    _env(monkeypatch, world, out)
    received, real = [], run_mod.run_stage_a

    def tampered(*args, **kwargs):
        received.append(kwargs["bound_spec_hash"])
        kwargs["bound_spec_hash"] = "sha256:other"
        return real(*args, **kwargs)
    monkeypatch.setattr(run_mod, "run_stage_a", tampered)
    with pytest.raises(OpsError) as exc:
        run_mod.run("cfg.json", out, **_paths(guard))
    assert base._refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "spec_changed")
    assert not (out / "fits").exists() and not (out / base.REPORT).exists()
    assert received == [lib.spec_hash(yaml.safe_load(guard["spec_path"].read_text(encoding="utf-8")))]
