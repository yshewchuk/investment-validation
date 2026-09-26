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


def test_provider_account_cli_is_discovered_and_owned():
    """The provider-account CLI the scheduler reserves against is discovered
    by the inventory and owned by a declared row (operations-quotas), so the
    new ``decisions`` parser cannot leave it as an UNOWNED_ENTRYPOINT."""
    discovered = inv.discover_entrypoints()
    assert "cli:ops provider-account" in discovered
    covered = set()
    for row in inv.build_document(discovered=discovered)["rows"]:
        covered |= set(row["old"]) | set(row["new"])
    assert "cli:ops provider-account" in covered


def test_generated_job_kinds_resolve_against_the_runtime_registry():
    """``ledger_export`` (built by ``stages.registry``'s outbox loop) and
    ``incremental_refresh`` (built by ``incremental_data.refresh_job_kind``)
    have no literal ``JobKind(name=...)`` call for the AST fallback to find;
    on the real checkout the runtime registry is what resolves them."""
    assert inv.resolve_entrypoint(inv.ROOT, "job:ledger_export", {}) is True
    assert inv.resolve_entrypoint(inv.ROOT, "job:incremental_refresh", {}) is True


def test_synthetic_trees_keep_the_literal_jobkind_fallback(tmp_path):
    """A tree that is not this checkout must never import this checkout's
    registry: it gets the AST scan, so a literal ``JobKind(...)`` resolves
    and a loop-generated kind does not."""
    stages = tmp_path / inv.V2_STAGES
    stages.parent.mkdir(parents=True)
    stages.write_text('x = JobKind(name="literal_kind")\n')
    assert inv.resolve_entrypoint(tmp_path, "job:literal_kind", {}) is True
    assert inv.resolve_entrypoint(tmp_path, "job:ledger_export", {}) is False
