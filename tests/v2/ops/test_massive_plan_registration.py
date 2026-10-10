"""Source-only regressions for the Massive planner's repository registrations."""
from pathlib import Path

from tools import mutation_pilot, phase6_inventory

ROOT = Path(__file__).resolve().parents[3]
PLANNER = "engine/v2/ops/massive_plan.py"
CLI = "cli:tools/v2_massive_plan.py"
TEST = "tests/v2/ops/test_massive_plan.py"


def test_planner_has_one_mutation_owner_with_its_offline_tests():
    config = mutation_pilot.load_config()
    owners = [name for name in mutation_pilot.enabled_modules(config)
              if PLANNER in mutation_pilot.mutate_files(config, name)]
    assert owners == ["ops_nightly"]
    assert TEST in config["modules"][owners[0]]["tests"]


def test_cli_has_a_native_plan_only_inventory_row():
    document = phase6_inventory.build_document(ROOT)
    rows = [row for row in document["rows"] if CLI in row["new"]]
    assert len(rows) == 1
    row = rows[0]
    assert row["disposition"] == "native"
    assert row["area"] == "operations"
    assert row["old"] == []
    assert TEST in row["tests"]
    assert row["producer"] == "massive_plan.plan (network-free, plan-only)"
