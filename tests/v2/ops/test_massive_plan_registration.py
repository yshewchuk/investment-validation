"""Source-only regressions for the Massive planner's repository registrations."""
import re
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


def test_planner_failure_heading_has_markdown_spacing():
    document = (ROOT / "engine/v2/ops/ARCHITECTURE.md").read_text()
    assert "\n\n### Massive Basic planning\n\n" in document


def test_planner_failure_table_names_each_refusal_separately():
    document = (ROOT / "engine/v2/ops/ARCHITECTURE.md").read_text()
    section = document.split("### Massive Basic planning\n", 1)[1].split("\n### ", 1)[0]
    rows = [line.split("|")[2] for line in section.splitlines() if line.startswith("| ")][1:]
    codes = [re.findall(r"`([A-Z_]+)`", outcome) for outcome in rows]
    assert all(len(row) == 1 for row in codes)
    assert sorted(row[0] for row in codes) == sorted({
        "INVALID_REQUEST", "SOURCE_INVALID", "INTEGRITY_FAILED", "MANIFEST_CORRUPT",
        "RESOURCE_LIMIT_EXCEEDED", "MISSING", "UNSAFE_PATH", "INPUT_CHANGED",
        "RESOURCE_UNAVAILABLE",
    })
