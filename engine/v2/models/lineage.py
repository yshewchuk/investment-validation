"""Frozen-state lineage and correction propagation (P5-4).

Every frozen serving state -- a Tier-4 monthly fold, a residual pool, a
payoff artifact, a calibration table -- declares WHAT it was built from: the
data tables it read (each bounded by the causal cutoff it respected, and
optionally by the exact logical keys it consumed) and the other frozen
states it was derived from. That declaration is :class:`Lineage`, and it is
part of the artifact's content hash, so a state cannot silently change what
it claims to depend on.

A Phase 3B :class:`~engine.v2.contracts.incremental.ChangeSet` records what a
data revision actually changed. :func:`propagate_corrections` intersects the
two: a state is invalid when a change lands inside one of its declared data
dependencies (right table, before its cutoff, and on a key it consumed when
it named its keys), or when any state upstream of it is invalid. Nothing is
inferred from names or dates the state did not declare, and a graph that
cannot be judged -- an unknown upstream id, a cycle, a state that declares no
lineage at all, a refused changeset -- raises :class:`LineageError` instead
of producing a partial answer.

The time rule is what makes "a historical correction invalidates every
dependent LATER fold" hold by construction: a fold trained on rows dated
before its fold start declares ``end_exclusive=<fold start>``, so a
correction dated D hits exactly the folds whose start is after D, and every
state built on one of those folds follows through ``upstream``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from engine.v2.contracts.incremental import ChangeSet, RowChange

__all__ = [
    "LINEAGE_V1",
    "DataDependency",
    "Invalidation",
    "InvalidationReport",
    "Lineage",
    "LineageError",
    "StateNode",
    "lineage_from_document",
    "propagate_corrections",
    "rebuild_order",
    "state_node",
]

LINEAGE_V1 = "frozen_state_lineage.v1.0"

#: Revision kinds that rewrite a key that already existed. An ``append``
#: adds a key, so it can only matter to a dependency that selected by time
#: range rather than by a fixed key list; see :func:`_change_hits`.
_REWRITES = frozenset({"correction", "tombstone"})


class LineageError(ValueError):
    """The dependency graph or a changeset cannot be judged exactly."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _day(value: Any) -> str | None:
    return None if value is None else str(value)[:10]


@dataclass(frozen=True, kw_only=True)
class DataDependency:
    """One table a frozen state read, and the part of it that mattered.

    ``end_exclusive`` is the causal bound the build respected: only rows
    dated strictly before it were read (``None``: no bound, every row).
    ``keys`` is the exact set of logical keys consumed, when the builder
    knows it; ``None`` means "every key in the time range", so any change
    in that range -- including an append -- reaches this state.
    """

    table: str
    end_exclusive: str | None = None
    keys: tuple[str, ...] | None = None

    def document(self) -> dict[str, Any]:
        return {
            "table": str(self.table),
            "end_exclusive": _day(self.end_exclusive),
            "keys": None if self.keys is None else sorted({str(k) for k in self.keys}),
        }


@dataclass(frozen=True, kw_only=True)
class Lineage:
    """What one frozen state was built from; hashed with the state itself."""

    data: tuple[DataDependency, ...] = ()
    upstream: tuple[str, ...] = ()
    schema_version: str = LINEAGE_V1

    @property
    def declared(self) -> bool:
        return bool(self.data or self.upstream)

    def canonical(self) -> "Lineage":
        """The same lineage in its one canonical order (what artifacts store)."""
        return lineage_from_document(self.document())

    def document(self) -> dict[str, Any]:
        """Canonical, order-independent form (sorted deps, sorted upstream)."""
        data = sorted(
            (item.document() for item in self.data),
            key=lambda doc: (doc["table"], doc["end_exclusive"] or "", str(doc["keys"])),
        )
        return {
            "schema_version": self.schema_version,
            "data": data,
            "upstream": sorted({str(item) for item in self.upstream}),
        }


def lineage_from_document(document: Mapping[str, Any] | None) -> Lineage:
    """Inverse of :meth:`Lineage.document` (used by artifact loaders)."""
    if not document:
        return Lineage()
    if document.get("schema_version") != LINEAGE_V1:
        raise LineageError("UNSUPPORTED_LINEAGE", repr(document.get("schema_version")))
    data = tuple(
        DataDependency(
            table=item["table"], end_exclusive=item.get("end_exclusive"),
            keys=None if item.get("keys") is None else tuple(item["keys"]),
        )
        for item in document.get("data", ())
    )
    return Lineage(data=data, upstream=tuple(document.get("upstream", ())))


@dataclass(frozen=True, kw_only=True)
class StateNode:
    """One frozen state in the dependency graph."""

    state_id: str
    content_hash: str
    lineage: Lineage


def state_node(state_id: str, state: Any) -> StateNode:
    """The graph node for any frozen state carrying ``content_hash``/``lineage``."""
    return StateNode(state_id=state_id, content_hash=state.content_hash, lineage=state.lineage)


@dataclass(frozen=True, order=True)
class Invalidation:
    """Why one state is invalid: a direct data hit, or an invalid upstream."""

    state_id: str
    reason: str
    cause: str


@dataclass(frozen=True, kw_only=True)
class InvalidationReport:
    changeset_ids: tuple[str, ...]
    invalid: tuple[Invalidation, ...]
    valid: tuple[str, ...]
    conservative: bool

    @property
    def invalid_ids(self) -> frozenset[str]:
        return frozenset(item.state_id for item in self.invalid)


def _check_graph(nodes: Sequence[StateNode]) -> dict[str, StateNode]:
    by_id: dict[str, StateNode] = {}
    for node in nodes:
        if node.state_id in by_id:
            raise LineageError("DUPLICATE_STATE", node.state_id)
        if not node.lineage.declared:
            raise LineageError("UNDECLARED_LINEAGE", node.state_id)
        by_id[node.state_id] = node
    for node in nodes:
        for parent in node.lineage.upstream:
            if parent not in by_id:
                raise LineageError("UNKNOWN_UPSTREAM", f"{node.state_id} <- {parent}")
    rebuild_order(nodes, by_id)  # raises on a cycle
    return by_id


def _change_start(change: RowChange) -> str | None:
    interval = change.time_range
    return None if interval is None else _day(interval.start_inclusive)


def _change_hits(dependency: DataDependency, change: RowChange) -> bool:
    """True when ``change`` lands inside what ``dependency`` read."""
    if change.revision_kind == "schema_change":
        return True
    start = _change_start(change)
    end = _day(dependency.end_exclusive)
    if start is not None and end is not None and start >= end:
        return False  # dated at or after the state's cutoff: never read
    if change.revision_kind in _REWRITES and dependency.keys is not None:
        return change.logical_key in dependency.keys
    return True


def _direct_hits(
    node: StateNode, changeset: ChangeSet,
) -> list[Invalidation]:
    table = changeset.table_contract_ref.contract_id
    hits: list[Invalidation] = []
    for impact in changeset.dependency_impacts:
        if impact.dependency_id == node.state_id:
            hits.append(Invalidation(
                node.state_id, "direct",
                f"{changeset.changeset_id}:impact:{impact.scope}",
            ))
    full = changeset.dependency_disposition == "conservative_full"
    for dependency in node.lineage.data:
        if dependency.table != table:
            continue
        if full:
            hits.append(Invalidation(
                node.state_id, "direct", f"{changeset.changeset_id}:conservative_full",
            ))
            continue
        for change in changeset.changes:
            if _change_hits(dependency, change):
                hits.append(Invalidation(
                    node.state_id, "direct",
                    f"{changeset.changeset_id}:{change.logical_key}",
                ))
                break
    return hits


def _usable(changesets: Iterable[ChangeSet]) -> tuple[list[ChangeSet], bool]:
    usable, conservative = [], False
    for changeset in changesets:
        if changeset.outcome == "refused" or changeset.dependency_disposition == "refused":
            raise LineageError("REFUSED_CHANGESET", changeset.changeset_id)
        if changeset.outcome == "noop":
            continue
        if changeset.dependency_disposition == "conservative_full":
            conservative = True
        usable.append(changeset)
    return usable, conservative


def _children(nodes: Sequence[StateNode]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {node.state_id: [] for node in nodes}
    for node in nodes:
        for parent in node.lineage.upstream:
            children[parent].append(node.state_id)
    return children


def propagate_corrections(
    nodes: Sequence[StateNode], changesets: Sequence[ChangeSet],
) -> InvalidationReport:
    """Every state a set of 3B changesets invalidates, by declared dependency.

    Direct hits come from :func:`_change_hits`; everything downstream of a
    hit (through ``Lineage.upstream``) is invalid too, whatever its own data
    dependencies say. Deterministic: the report is sorted and depends only
    on the inputs, never on their order.
    """
    by_id = _check_graph(nodes)
    usable, conservative = _usable(changesets)
    first: dict[str, Invalidation] = {}
    for state_id in sorted(by_id):
        for changeset in sorted(usable, key=lambda item: item.changeset_id):
            hits = _direct_hits(by_id[state_id], changeset)
            if hits:
                first.setdefault(state_id, min(hits))
    children = _children(nodes)
    queue = deque(sorted(first))
    while queue:
        parent = queue.popleft()
        for child in sorted(children[parent]):
            if child not in first:
                first[child] = Invalidation(child, "upstream", parent)
                queue.append(child)
    invalid = tuple(sorted(first.values()))
    return InvalidationReport(
        changeset_ids=tuple(sorted(item.changeset_id for item in usable)),
        invalid=invalid,
        valid=tuple(sorted(set(by_id) - set(first))),
        conservative=conservative,
    )


def rebuild_order(
    nodes: Sequence[StateNode],
    only: Iterable[str] | Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Upstream-first order to rebuild ``only`` (default: every state).

    Kahn's algorithm with sorted tie-breaking, so the plan is deterministic.
    Raises ``LineageError("DEPENDENCY_CYCLE")`` when the graph has a cycle.
    """
    wanted = {node.state_id for node in nodes} if only is None else set(only)
    indegree = {node.state_id: 0 for node in nodes}
    children = {node.state_id: [] for node in nodes}
    for node in nodes:
        for parent in node.lineage.upstream:
            if parent in children:
                children[parent].append(node.state_id)
                indegree[node.state_id] += 1
    ready = sorted(state for state, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        state = ready.pop(0)
        order.append(state)
        for child in sorted(children[state]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()
    if len(order) != len(indegree):
        raise LineageError("DEPENDENCY_CYCLE", ",".join(sorted(set(indegree) - set(order))))
    return tuple(state for state in order if state in wanted)
