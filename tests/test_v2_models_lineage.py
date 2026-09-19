"""P5-4 correction propagation: a 3B changeset invalidates by declared dependency.

Acceptance (guides/rearchitecture_phase5_models.md, P5-4): "historical
correction invalidates every dependent later fold/state". The graph here is
synthetic but shaped like the real one: monthly Tier-4 folds that read the
panel strictly before their fold start, a driver residual pool per fold, a
paired pool over every fold, and an unrelated calibration table.
"""
from __future__ import annotations

import hashlib
import random

import pytest

from engine.v2.contracts.data import DatasetVersionRef, TableContractRef, TimeInterval
from engine.v2.contracts.incremental import ChangeSet, DependencyImpact, RowChange
from engine.v2.models.lineage import (
    DataDependency,
    Lineage,
    LineageError,
    StateNode,
    lineage_from_document,
    propagate_corrections,
    rebuild_order,
)

PANEL = "tier3.panel"
FORECASTS = "tier4.forecasts"
CRUSH = "tier2.crush"
FOLDS = ("2024-03-01", "2024-04-01", "2024-05-01")


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _node(state_id: str, *, data=(), upstream=()) -> StateNode:
    return StateNode(state_id=state_id, content_hash=_hash(state_id),
                     lineage=Lineage(data=tuple(data), upstream=tuple(upstream)))


def _graph() -> list[StateNode]:
    nodes = []
    for fold in FOLDS:
        nodes.append(_node(f"fold:size:{fold}",
                           data=[DataDependency(table=PANEL, end_exclusive=fold)]))
        nodes.append(_node(
            f"driver_pool:size:{fold}",
            data=[DataDependency(table=FORECASTS, end_exclusive=fold)],
            upstream=[f"fold:size:{fold}"],
        ))
    nodes.append(_node(
        "paired_pool:2024-06-01",
        data=[DataDependency(table=PANEL, end_exclusive="2024-06-01"),
              DataDependency(table=CRUSH, end_exclusive="2024-06-01")],
        upstream=[f"fold:size:{fold}" for fold in FOLDS],
    ))
    nodes.append(_node("admissible_table:v1",
                       data=[DataDependency(table="dyn_sv.menu_events",
                                            end_exclusive="2026-09-10")]))
    return nodes


def _changeset(table: str, *changes: RowChange, changeset_id: str = "cs-1",
               disposition: str = "exact", outcome: str = "changed",
               impacts=()) -> ChangeSet:
    contract = TableContractRef(contract_id=table, definition_hash="sha256:" + "0" * 64)
    version = lambda name: DatasetVersionRef(  # noqa: E731
        dataset_version_id=name, table_contract_ref=contract,
        manifest_hash="sha256:" + "1" * 64)
    return ChangeSet(
        changeset_id=changeset_id, table_contract_ref=contract,
        base_dataset_version_ref=version("v1"), result_dataset_version_ref=version("v2"),
        acquisition_receipt_refs=(), coverage_receipt_refs=(), changes=tuple(changes),
        changed_partitions=(), dependency_impacts=tuple(impacts), unknown_dependencies=(),
        dependency_disposition=disposition, outcome=outcome,
        normalized_payloads=len(changes), rewritten_partitions=0,
    )


def _change(key: str, day: str | None, kind: str = "correction") -> RowChange:
    interval = None if day is None else TimeInterval(column="event_date", start_inclusive=day)
    return RowChange(logical_key=key, partition_key=day or "", columns=("abs_move",),
                     time_range=interval, old_hash="sha256:" + "2" * 64,
                     new_hash="sha256:" + "3" * 64, revision_kind=kind, revision_id="r1")


def test_historical_correction_invalidates_every_later_fold_and_its_dependents():
    report = propagate_corrections(
        _graph(), [_changeset(PANEL, _change("AAA|2024-03-15", "2024-03-15"))],
    )

    assert report.invalid_ids == {
        "fold:size:2024-04-01", "fold:size:2024-05-01",
        "driver_pool:size:2024-04-01", "driver_pool:size:2024-05-01",
        "paired_pool:2024-06-01",
    }
    # The fold that trained strictly before the corrected date, the pool built
    # on it, and the unrelated calibration table all stay valid.
    assert set(report.valid) == {
        "fold:size:2024-03-01", "driver_pool:size:2024-03-01", "admissible_table:v1",
    }
    reasons = {item.state_id: (item.reason, item.cause) for item in report.invalid}
    assert reasons["fold:size:2024-04-01"] == ("direct", "cs-1:AAA|2024-03-15")
    assert reasons["driver_pool:size:2024-04-01"] == ("upstream", "fold:size:2024-04-01")
    assert reasons["paired_pool:2024-06-01"][0] == "direct"
    assert report.conservative is False


def test_correction_dated_after_every_cutoff_invalidates_nothing():
    report = propagate_corrections(
        _graph(), [_changeset(PANEL, _change("AAA|2026-12-01", "2026-12-01"))],
    )
    assert report.invalid == ()


def test_correction_on_the_fold_start_itself_is_after_that_fold():
    # end_exclusive: a row dated exactly on the fold start was never read.
    report = propagate_corrections(
        _graph(), [_changeset(PANEL, _change("AAA|2024-04-01", "2024-04-01"))],
    )
    assert "fold:size:2024-04-01" not in report.invalid_ids
    assert "fold:size:2024-05-01" in report.invalid_ids


def test_keyed_dependency_ignores_corrections_to_keys_it_never_read_but_not_appends():
    nodes = [_node("pool:keyed", data=[DataDependency(
        table=PANEL, end_exclusive="2024-06-01", keys=("AAA|2024-01-05",))])]
    other_key = propagate_corrections(
        nodes, [_changeset(PANEL, _change("BBB|2024-01-05", "2024-01-05"))])
    own_key = propagate_corrections(
        nodes, [_changeset(PANEL, _change("AAA|2024-01-05", "2024-01-05"))])
    appended = propagate_corrections(
        nodes, [_changeset(PANEL, _change("CCC|2024-02-01", "2024-02-01", "append"))])

    assert other_key.invalid == ()
    assert own_key.invalid_ids == {"pool:keyed"}
    assert appended.invalid_ids == {"pool:keyed"}


def test_undated_change_and_schema_change_hit_conservatively():
    undated = propagate_corrections(_graph(), [_changeset(PANEL, _change("AAA", None))])
    schema = propagate_corrections(
        _graph(), [_changeset(PANEL, _change("*", "2030-01-01", "schema_change"))])
    everything_on_panel = {
        "fold:size:2024-03-01", "fold:size:2024-04-01", "fold:size:2024-05-01",
        "paired_pool:2024-06-01",
    }
    assert everything_on_panel <= undated.invalid_ids
    assert everything_on_panel <= schema.invalid_ids


def test_conservative_full_changeset_invalidates_every_dependent_and_says_so():
    report = propagate_corrections(
        _graph(), [_changeset(CRUSH, _change("AAA", "2030-01-01"),
                              disposition="conservative_full")])
    assert report.invalid_ids == {"paired_pool:2024-06-01"}
    assert report.conservative is True


def test_explicit_dependency_impact_names_a_state():
    report = propagate_corrections(_graph(), [_changeset(
        "unrelated.table", impacts=(DependencyImpact(
            dependency_id="admissible_table:v1", scope="full", affected_keys=()),),
    )])
    assert report.invalid_ids == {"admissible_table:v1"}


def test_noop_changeset_is_skipped_and_refused_changeset_raises():
    noop = propagate_corrections(
        _graph(), [_changeset(PANEL, _change("AAA", "2024-01-01"), outcome="noop")])
    assert noop.invalid == () and noop.changeset_ids == ()
    with pytest.raises(LineageError) as refused:
        propagate_corrections(_graph(), [_changeset(PANEL, outcome="refused")])
    assert refused.value.code == "REFUSED_CHANGESET"


@pytest.mark.parametrize("nodes, code", [
    ([_node("a", upstream=["missing"])], "UNKNOWN_UPSTREAM"),
    ([StateNode(state_id="a", content_hash=_hash("a"), lineage=Lineage())], "UNDECLARED_LINEAGE"),
    ([_node("a", upstream=["b"]), _node("b", upstream=["a"])], "DEPENDENCY_CYCLE"),
    ([_node("a", data=[DataDependency(table=PANEL)])] * 2, "DUPLICATE_STATE"),
])
def test_a_graph_that_cannot_be_judged_raises_instead_of_guessing(nodes, code):
    with pytest.raises(LineageError) as error:
        propagate_corrections(nodes, [])
    assert error.value.code == code


def test_report_is_independent_of_node_and_changeset_order():
    changesets = [
        _changeset(PANEL, _change("AAA|2024-03-15", "2024-03-15"), changeset_id="cs-a"),
        _changeset(CRUSH, _change("BBB|2024-05-20", "2024-05-20"), changeset_id="cs-b"),
    ]
    baseline = propagate_corrections(_graph(), changesets)
    rng = random.Random(7)
    for _ in range(5):
        nodes = _graph()
        rng.shuffle(nodes)
        shuffled = list(changesets)
        rng.shuffle(shuffled)
        assert propagate_corrections(nodes, shuffled) == baseline


def test_rebuild_order_puts_every_upstream_first():
    nodes = _graph()
    report = propagate_corrections(
        nodes, [_changeset(PANEL, _change("AAA|2024-03-15", "2024-03-15"))])
    order = rebuild_order(nodes, report.invalid_ids)

    assert set(order) == report.invalid_ids
    position = {state: index for index, state in enumerate(order)}
    by_id = {node.state_id: node for node in nodes}
    for state in order:
        for parent in by_id[state].lineage.upstream:
            if parent in position:
                assert position[parent] < position[state]


def test_lineage_document_round_trips_and_is_order_independent():
    first = Lineage(
        data=(DataDependency(table=PANEL, end_exclusive="2024-04-01T00:00:00",
                             keys=("b", "a")),
              DataDependency(table=CRUSH)),
        upstream=("y", "x"),
    )
    second = Lineage(
        data=(DataDependency(table=CRUSH),
              DataDependency(table=PANEL, end_exclusive="2024-04-01", keys=("a", "b"))),
        upstream=("x", "y"),
    )
    assert first.document() == second.document()
    assert lineage_from_document(first.document()).document() == first.document()
