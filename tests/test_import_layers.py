"""The layer check's negative controls.

`rearchitecture_phase0_baseline.md` §4 acceptance: the checker fails a planted
upward import inside v2, fails a planted undeclared legacy import, fails a
planted ``engine/* -> engine/v2/*`` import, and passes the real tree.

A check that has never failed is not known to work, so every rule here is
proved by planting the violation it exists to catch — not by asserting the
real tree is green, which it would also be if the check did nothing.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import import_layers as il  # noqa: E402
from checks.layer_map import CONTAINERS, PACKAGES, package_of  # noqa: E402


def rules(report) -> set[str]:
    return {v.rule for v in report.violations}


# --------------------------------------------------------------------------
# the real tree
# --------------------------------------------------------------------------


def test_real_tree_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "import_layers.py"), "--all"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr


def test_real_tree_runs_in_under_two_seconds():
    started = time.monotonic()
    subprocess.run(
        [sys.executable, str(ROOT / "checks" / "import_layers.py"), "--all", "--quiet"],
        capture_output=True, cwd=ROOT, check=True,
    )
    assert time.monotonic() - started < 2.0


def test_every_declared_package_exists():
    assert il.missing_skeleton(ROOT) == []


def test_adapter_ledger_is_empty():
    """Phase 0 exit gate: the layer check runs green with an empty ledger."""
    ledger = il.load_adapters()
    assert ledger["count"] == 0
    assert ledger["adapters"] == []
    assert il.check_ledger(ledger) == []


# --------------------------------------------------------------------------
# rule 1 — inside v2, imports point down only
# --------------------------------------------------------------------------


def test_planted_upward_import_fails():
    files = {
        "engine/v2/features/panel.py": b"from engine.v2.scoring import kernel\n",
    }
    report = il.check_files(files)
    assert "upward-import" in rules(report)


def test_planted_peer_import_fails():
    """4a peers may not import each other — the split is what enforces it."""
    files = {
        "engine/v2/domain/scenarios/pool.py":
            b"from engine.v2.domain.valuation import mark\n",
    }
    assert "upward-import" in rules(il.check_files(files))


def test_downward_import_passes():
    files = {
        "engine/v2/scoring/kernel.py": b"from engine.v2.features import frame\n",
        "engine/v2/domain/simulation/draws.py":
            b"from engine.v2.domain.valuation import mark\n",
        "engine/v2/foundation/canonical.py": b"from engine.v2.contracts import Score\n",
    }
    assert il.check_files(files).violations == []


def test_intra_package_import_passes():
    files = {"engine/v2/scoring/kernel.py": b"from engine.v2.scoring import stages\n"}
    assert il.check_files(files).violations == []


def test_nothing_may_import_diagnosis():
    files = {"engine/v2/serving/api.py": b"from engine.v2.diagnosis import compare\n"}
    assert "diagnosis-imported" in rules(il.check_files(files))


def test_diagnosis_may_import_everything_below_it():
    files = {
        "engine/v2/diagnosis/parity.py": (
            b"from engine.v2.scoring import kernel\n"
            b"from engine.v2.data import store\n"
            b"from engine.v2.serving import release\n"
        ),
    }
    assert il.check_files(files).violations == []


def test_dashboard_may_import_layer_seven_only():
    ok = {"engine/v2/dashboard/board.py": b"from engine.v2.serving import rows\n"}
    assert il.check_files(ok).violations == []
    bad = {"engine/v2/dashboard/board.py": b"from engine.v2.scoring import kernel\n"}
    assert "upward-import" in rules(il.check_files(bad))


def test_container_import_fails():
    files = {"engine/v2/scoring/kernel.py": b"import engine.v2.domain\n"}
    assert "container-import" in rules(il.check_files(files))


def test_unmapped_package_fails():
    files = {"engine/v2/scoring/kernel.py": b"from engine.v2.telemetry import log\n"}
    assert "unmapped-package" in rules(il.check_files(files))


# --------------------------------------------------------------------------
# rule 2 — v2 reaches legacy only through declared adapters
# --------------------------------------------------------------------------


def test_planted_undeclared_legacy_import_fails():
    files = {"engine/v2/features/panel.py": b"from engine.features import live\n"}
    assert "undeclared-legacy-import" in rules(il.check_files(files))


def test_declared_legacy_import_passes():
    ledger = {
        "count": 1,
        "adapters": [{
            "package": "engine.v2.features",
            "module": "engine.v2.features.legacy_adapter",
            "legacy_symbol": "engine.features",
            "reason": "panel columns until engine/v2/data lands",
            "declared_on": "2026-09-12",
        }],
    }
    files = {
        "engine/v2/features/legacy_adapter.py": b"from engine.features import live\n",
    }
    assert il.check_files(files, ledger).violations == []


def test_declaration_does_not_travel_outside_its_adapter_module():
    """One adapter module per package: a sibling importing the same symbol fails."""
    ledger = {
        "count": 1,
        "adapters": [{
            "package": "engine.v2.features",
            "module": "engine.v2.features.legacy_adapter",
            "legacy_symbol": "engine.features",
            "reason": "r", "declared_on": "2026-09-12",
        }],
    }
    files = {"engine/v2/features/panel.py": b"from engine.features import live\n"}
    assert "undeclared-legacy-import" in rules(il.check_files(files, ledger))


def test_two_adapter_modules_in_one_package_fails():
    ledger = {
        "count": 2,
        "adapters": [
            {"package": "engine.v2.features", "module": "engine.v2.features.a",
             "legacy_symbol": "engine.features", "reason": "r",
             "declared_on": "2026-09-12"},
            {"package": "engine.v2.features", "module": "engine.v2.features.b",
             "legacy_symbol": "engine.score", "reason": "r",
             "declared_on": "2026-09-12"},
        ],
    }
    assert any("allows one per package" in p for p in il.check_ledger(ledger))


def test_miscounted_ledger_fails():
    ledger = {"count": 0, "adapters": [
        {"package": "p", "module": "m", "legacy_symbol": "s", "reason": "r",
         "declared_on": "2026-09-12"},
    ]}
    assert any("count is 0" in p for p in il.check_ledger(ledger))


def test_ledger_entry_missing_a_field_fails():
    ledger = {"count": 1, "adapters": [{"package": "p", "module": "m"}]}
    assert any("missing" in p for p in il.check_ledger(ledger))


# --------------------------------------------------------------------------
# rule 3 — legacy never imports v2
# --------------------------------------------------------------------------


def test_planted_legacy_importing_v2_fails():
    files = {"engine/score.py": b"from engine.v2.scoring import kernel\n"}
    assert "legacy-imports-v2" in rules(il.check_files(files))


def test_legacy_importing_legacy_passes():
    files = {"engine/score.py": b"from engine.features import live\n"}
    assert il.check_files(files).violations == []


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel,expected",
    [
        ("engine/v2/scoring/__init__.py", "engine.v2.scoring"),
        ("engine/v2/scoring/kernel.py", "engine.v2.scoring.kernel"),
        ("checks/import_layers.py", "checks.import_layers"),
        ("guides/x.md", None),
    ],
)
def test_module_name(rel, expected):
    assert il.module_name(rel) == expected


def test_relative_imports_resolve():
    files = {"engine/v2/features/panel.py": b"from ..scoring import kernel\n"}
    assert "upward-import" in rules(il.check_files(files))


def test_plain_import_statement_is_seen():
    files = {"engine/v2/features/panel.py": b"import engine.v2.scoring.kernel\n"}
    assert "upward-import" in rules(il.check_files(files))


def test_syntax_error_is_skipped_rather_than_crashing():
    files = {"engine/v2/features/panel.py": b"def (\n"}
    assert il.check_files(files).violations == []


def test_longest_prefix_wins_for_nested_training_package():
    """models/training is layer 6, not layer 3 — a scorer must not reach it."""
    assert package_of("engine.v2.models.training.gate").layer == 6.0
    assert package_of("engine.v2.models.registry").layer == 3.0
    files = {"engine/v2/scoring/kernel.py":
             b"from engine.v2.models.training import gate\n"}
    assert "upward-import" in rules(il.check_files(files))


def test_features_may_not_import_training():
    """The Tier-4 feature-model cycle, dissolved by construction (§4.1)."""
    files = {"engine/v2/features/tier4.py":
             b"from engine.v2.models.training import fit\n"}
    assert "upward-import" in rules(il.check_files(files))


def test_features_may_not_import_inference_either():
    """The §4.1 table gives features "0-1", and the table is what is enforced.

    The §4.1 prose says "features may import inference but never training".
    The table one paragraph above it says features may import layers 0-1, and
    models/ is layer 3. The check holds the table: it is strictly stronger, it
    never contradicts "never training", and enforcing the looser prose instead
    would create a 2 <-> 3 cycle, since models may import features. Recorded in
    checks/layer_map.py as a design question to settle before Tier-4
    materialization is written, not silently resolved here.
    """
    files = {"engine/v2/features/tier4.py": b"from engine.v2.models import champion\n"}
    assert "upward-import" in rules(il.check_files(files))


def test_no_package_shares_a_dotted_path_with_a_container():
    declared = {p.dotted for p in PACKAGES}
    assert declared.isdisjoint(set(CONTAINERS))
