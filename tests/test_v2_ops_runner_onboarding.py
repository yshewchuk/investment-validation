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


def test_runner_manifest_mixes_reviewed_inventory_with_mechanical_evidence():
    root = Path(__file__).resolve().parents[1]
    manifest = runner_manifest(root, REGISTERED)
    assert manifest["schema_version"] == "runner_capability_manifest.v1.0"
    assert manifest["no_ledger_support"] is True
    assert manifest["spec_hash"].startswith("sha256:")
    closure = manifest["source_closure"]
    assert "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py" in closure
    assert "experiments/EXP-181_d_1_gated_execution_parity/run.py" in closure
    assert manifest["report_path"] == "REPORT.md"
    assert "promotion" in manifest["registry_effects"]
    assert runner_manifest(root, REGISTERED) == manifest


def test_unaudited_runner_is_refused(tmp_path):
    with pytest.raises(OpsError, match="not audited"):
        runner_manifest(tmp_path, "experiments/EXP-999_nope/run.py")
    with pytest.raises(OpsError, match="unaudited"):
        run_legacy_script(tmp_path, "experiments/EXP-999_nope/run.py")


def test_registered_runner_executes_no_ledger_only_in_its_synthetic_root(tmp_path):
    script = _synthetic_root(tmp_path, FAKE_RUNNER)
    result = run_legacy_script(tmp_path, REGISTERED)
    assert result.returncode == 0
    assert (tmp_path / "REPORT.md").is_file()
    with pytest.raises(OpsError, match="ledger"):
        run_legacy_script(tmp_path, REGISTERED, args=("--record",))
    script.unlink()
    with pytest.raises(OpsError):
        runner_manifest(tmp_path, REGISTERED)


def test_leaking_runner_is_caught_by_the_write_audit(tmp_path):
    prod = tmp_path / "root"
    (prod / "ledger" / "predictions").mkdir(parents=True)
    _synthetic_root(prod, LEAKING_RUNNER)
    before = snapshot_sensitive(prod)
    run_legacy_script(prod, REGISTERED)
    findings = audit_writes(prod, before, disclosed=(prod / "experiments",))
    assert {"path": "ledger/predictions/leak.jsonl", "kind": "new"} in findings


def test_exp185_wrapper_patches_here_and_results_before_main():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
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

    root = Path(__file__).resolve().parents[1]
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append((a, k)))
    assert retired not in RUNNER_INVENTORY
    assert retired not in legacy_adapter.REGISTERED_RUNNERS
    with pytest.raises(OpsError) as manifest_error:
        runner_manifest(root, retired)
    assert manifest_error.value.code == "INVALID_REQUEST"
    with pytest.raises(OpsError) as run_error:
        legacy_adapter.run_legacy_script(root, retired)
    assert run_error.value.code == "INVALID_REQUEST"
    assert calls == []


def test_active_registration_keeps_its_audited_source_closure():
    from engine.v2.ops import legacy_adapter
    from engine.v2.ops.experiments import RUNNER_INVENTORY

    root = Path(__file__).resolve().parents[1]
    assert set(RUNNER_INVENTORY) == {REGISTERED}
    assert legacy_adapter.REGISTERED_RUNNERS == frozenset({REGISTERED})
    manifest = runner_manifest(root, REGISTERED)
    assert manifest["no_ledger_support"] is True
    assert REGISTERED in manifest["source_closure"]
    assert "experiments/EXP-181_d_1_gated_execution_parity/run.py" in manifest["source_closure"]
