"""Guard the ``decisions-supersede`` Phase 6 declaration this task adds.

``tools/phase6_capabilities.toml``'s row now claims a CLI entrypoint, the
supervised job kind and the native commit function. ``checks/phase6_inventory.py``
resolves each of those against the real source, so a rename or a dropped
declaration would surface as ``UNOWNED_ENTRYPOINT``/``NEW_ENTRYPOINT_MISSING``
in the real-checkout inventory. This focused, source-only test pins that the
row's own claims resolve and that both discovered CLI levels are owned.
"""
from __future__ import annotations

from tools import phase6_inventory as inv

NEW = ["cli:ops decisions", "cli:ops decisions supersede", "job:decisions_supersede",
       "py:engine/v2/ops/decision_commit.py::commit_supersede"]


def test_decisions_supersede_row_claims_all_resolve():
    document = inv.build_document()
    row = next(item for item in document["rows"] if item["id"] == "decisions-supersede")
    assert row["new"] == NEW
    assert row["new_resolved"] == {ref: True for ref in NEW}


def test_both_decisions_cli_levels_are_discovered_and_owned():
    discovered = inv.discover_entrypoints()
    assert {"cli:ops decisions", "cli:ops decisions supersede"} <= set(discovered)
    covered = set()
    for row in inv.build_document(discovered=discovered)["rows"]:
        covered |= set(row["old"]) | set(row["new"])
    assert {"cli:ops decisions", "cli:ops decisions supersede"} <= covered
