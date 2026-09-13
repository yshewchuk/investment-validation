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
