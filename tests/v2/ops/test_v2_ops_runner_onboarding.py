"""Runner onboarding contract: reviewed inventory plus mechanical evidence (§11.1)."""
from pathlib import Path

import pytest

from engine.v2.ops.diagnostics import audit_writes, snapshot_sensitive
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import runner_manifest
from engine.v2.ops.legacy_adapter import run_legacy_script

REGISTERED = "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py"

FAKE_RUNNER = r"""\
import sys
from pathlib import Path

if "--no-ledger" not in sys.argv:
    raise SystemExit("ledger writes must be disabled")
if [a for a in sys.argv[1:] if a != "--no-ledger"]:
    raise SystemExit("unexpected argument")
Path("REPORT.md").write_text("# fake\n")
print("done")
"""

LEAKING_RUNNER = r"""\
import sys
from pathlib import Path

if "--no-ledger" not in sys.argv:
    raise SystemExit("ledger writes must be disabled")
if [a for a in sys.argv[1:] if a != "--no-ledger"]:
    raise SystemExit("unexpected argument")
Path("ledger/predictions/leak.jsonl").write_text("{}\n")
Path("REPORT.md").write_text("# fake\n")
print("done")
"""


def _synthetic_root(root: Path, source: str) -> Path:
    run_dir = root / "experiments" / "EXP-182_d_1_gated_execution_parity_registered"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.py").write_text(source)
    return run_dir / "run.py"


def test_disabled_runner_manifest_refuses_with_reason():
    from engine.v2.ops.experiments import DISABLED_RUNNERS

    root = Path(__file__).resolve().parents[3]
    reason = DISABLED_RUNNERS[REGISTERED]
    with pytest.raises(OpsError) as excinfo:
        runner_manifest(root, REGISTERED)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    assert excinfo.value.problem.details == {"runner": REGISTERED, "reason": reason}


def test_unaudited_runner_is_refused(tmp_path):
    with pytest.raises(OpsError, match="not audited"):
        runner_manifest(tmp_path, "experiments/EXP-999_nope/run.py")
    with pytest.raises(OpsError, match="unaudited"):
        run_legacy_script(tmp_path, "experiments/EXP-999_nope/run.py")


def test_disabled_runner_adapter_refuses_before_subprocess(tmp_path):
    from engine.v2.ops.experiments import DISABLED_RUNNERS

    script = _synthetic_root(tmp_path, FAKE_RUNNER)
    reason = DISABLED_RUNNERS[REGISTERED]
    with pytest.raises(OpsError) as excinfo:
        run_legacy_script(tmp_path, REGISTERED)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    assert excinfo.value.problem.details == {"runner": REGISTERED, "reason": reason}
    assert not (tmp_path / "REPORT.md").exists()
    assert script.is_file()


def test_disabled_runner_cannot_write_even_if_a_script_exists(tmp_path):
    prod = tmp_path / "root"
    (prod / "ledger" / "predictions").mkdir(parents=True)
    script = _synthetic_root(prod, LEAKING_RUNNER)
    before = snapshot_sensitive(prod)
    with pytest.raises(OpsError) as excinfo:
        run_legacy_script(prod, REGISTERED)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    findings = audit_writes(prod, before, disclosed=(prod / "experiments",))
    assert findings == []
    assert script.is_file()
    assert not (prod / "ledger" / "predictions" / "leak.jsonl").exists()


def test_exp185_wrapper_patches_here_and_results_before_main():
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    wrapper_dir = root / (
        "experiments/EXP-185_str_runup_t14_corrected_calendar_gate_rebaseline_registered"
    )
    source = wrapper_dir / "run.py"
    module_spec = importlib.util.spec_from_file_location("exp185_wrapper_under_test", source)
    wrapper = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(wrapper)

    # HERE/RESULTS are not the wrapper's own top-level names -- run.py sets
    # them on the EXP-144 module it loads (the ``module`` name in its
    # namespace), which is the object ``main()`` would run against. HERE also
    # happens to exist as a plain wrapper-level assignment, so asserting on
    # ``wrapper.module`` is what actually checks the patch, not the
    # coincidental same-named local.
    assert wrapper.module.HERE == wrapper_dir
    assert wrapper.module.RESULTS == wrapper_dir / "results"


RETIRED_RUNNERS = (
    "experiments/EXP-184_str_thru_gate_promotion_confirmatory_val_registered/run.py",
    "experiments/EXP-185_str_runup_t14_corrected_calendar_gate_rebaseline_registered/run.py",
)


@pytest.mark.parametrize("retired", RETIRED_RUNNERS)
def test_retired_runner_is_refused_before_any_subprocess(retired, monkeypatch):
    import subprocess
    from engine.v2.ops import legacy_adapter
    from engine.v2.ops.experiments import RUNNER_INVENTORY

    root = Path(__file__).resolve().parents[3]
    spawn_calls = []

    def fail_spawn(*a, **k):
        spawn_calls.append((a, k))
        raise AssertionError("retired runner spawned a subprocess")

    monkeypatch.setattr(subprocess, "Popen", fail_spawn)
    assert retired not in RUNNER_INVENTORY
    assert retired not in legacy_adapter.REGISTERED_RUNNERS
    with pytest.raises(OpsError) as manifest_error:
        runner_manifest(root, retired)
    assert manifest_error.value.code == "INVALID_REQUEST"
    with pytest.raises(OpsError) as run_error:
        legacy_adapter.run_legacy_script(root, retired)
    assert run_error.value.code == "INVALID_REQUEST"
    assert spawn_calls == []


def test_exp182_registration_is_disabled_with_audited_reason():
    from engine.v2.ops import legacy_adapter
    from engine.v2.ops.experiments import DISABLED_RUNNERS, RUNNER_INVENTORY

    assert REGISTERED in RUNNER_INVENTORY
    assert RUNNER_INVENTORY[REGISTERED]["enabled"] is False
    assert REGISTERED not in legacy_adapter.REGISTERED_RUNNERS
    assert DISABLED_RUNNERS[REGISTERED] == (
        "disabled 2026-10-10: reads EXP-179 parquet directly "
        "(no v2_snapshot_id), bypassing holdouts")
