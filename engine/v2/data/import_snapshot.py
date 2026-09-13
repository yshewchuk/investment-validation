"""Pure §7.1 import planning: enumerate the pinned legacy store, build the
declared read set, and freeze one ``SnapshotImportRequest``.

Phase-2 guide §7.1 points 2-3 ("Resolve the accepted legacy SNAPSHOT, enumerate
every declared table and compatibility object from known paths, and reject
globs outside those roots"; "Create SnapshotImportRequest containing the
expected legacy snapshot hash, exact relative file list, expected contract
refs, calendar and source-priority versions, knowledge modes, and expected
current head"), task brief decision 2 (P2-7/Task7b).

:func:`plan_import` is pure apart from reading file stats/hashes under an
explicit ``source_root`` — it never reads a clock, never touches the catalog,
and never submits anything; the ops layer (``engine.v2.ops.snapshot_import``)
publishes its two documents and submits them through the Phase 1 supervisor.
It enumerates *exactly* the known paths the facts section names:

* ``data/curated/{table}/year=YYYY/part-NNNN.parquet`` for the six Tier-2
  tables — one part per year, except ``daily_market`` (several sorted,
  non-overlapping parts);
* ``features/panel.parquet``, ``features/tier4_forecasts.parquet`` (one file
  each, no ``year=`` partitioning — task brief's "one declared logical
  partition", key ``"all"``, mirroring ``objects.py``'s own partition-key
  convention);
* ``features/SNAPSHOT`` (JSON compatibility object, never inspected as Parquet).

Refusals (task brief decision 2), everywhere named ``CONTRACT_MISMATCH`` for a
declared-shape violation and ``INPUT_CHANGED`` for a missing or indirect path:

* an undeclared file inside a table directory (anything other than
  ``part-NNNN.parquet`` under ``year=YYYY``) — ``CONTRACT_MISMATCH``;
* a symlink at any enumerated path (table dir, year dir, part file, feature
  file, ``SNAPSHOT``) — ``INPUT_CHANGED``;
* a missing required file or table directory — ``INPUT_CHANGED``;
* the legacy ``SNAPSHOT`` JSON missing one of its reviewed
  ``expected_top_level_keys`` — ``CONTRACT_MISMATCH``.

A year's parts out of *name* order is impossible by construction (this module
sorts by filename before building the declared list); out of *key* order is
checked later, by streaming inspection (``objects.inspect_staged_file`` /
``objects.partition_logical_hash``), not here.

``SnapshotImportRequest.calendar_version`` cannot be known at plan time: it is
the ``earnings_events`` dataset's own logical content hash, which only exists
after that table has been inspected. The contract does not make a two-phase
bind impossible (this module's own check, not a stop-and-report): the request
carries :data:`PENDING_CALENDAR_VERSION` as an explicit, greppable placeholder,
and the coordinator binds the real value into the *built* ``SnapshotRef`` it
submits to ``catalog.commit_snapshot`` — a separate contract object, so the
placeholder never reaches anything that treats it as fact. See
``engine/v2/ops/snapshot_import.py``'s module docstring for the bind site.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts``, ``engine.v2.foundation``, and this package's own
``errors``/``legacy_adapter`` (for ``build_legacy_mapping``,
``SOURCE_PRIORITY_VERSION`` and the three reviewed ``*_RELATIVE_PATH``
literals) — never ``engine.v2.ops`` or legacy ``engine.*`` directly (that one
import stays confined to ``legacy_adapter.py``, §4.2's "one adapter module per
package").
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path

from engine.v2.contracts import (
    LegacyFileRef,
    LegacyInputManifest,
    SnapshotImportRequest,
    TableContractRef,
)
from engine.v2.data import errors, legacy_adapter
from engine.v2.foundation import CONTENT_HASH_PREFIX, content_hash, to_document

__all__ = [
    "ImportPlan",
    "PENDING_CALENDAR_VERSION",
    "plan_import",
    "request_hash",
]

#: Explicit, greppable placeholder — see the module docstring's calendar note.
PENDING_CALENDAR_VERSION = "legacy_calendar:pending"

_PART_RE = re.compile(r"^part-(\d{4})\.parquet$")
_YEAR_RE = re.compile(r"^year=(\d{4})$")
_CHUNK = 1 << 20


@dataclasses.dataclass(frozen=True, kw_only=True)
class ImportPlan:
    """Everything §7.1 needs to submit and later re-derive one import.

    ``partition_layout`` is ``{table: ((partition_key, (relpath, ...)), ...)}``,
    in the exact order fragments must be inspected/committed in (task brief
    decision 1 -- membership order is ``(partition_key, primary_key_min)``,
    and within one partition this module's own filename sort already matches
    ascending key order for every table the legacy store actually writes).
    """

    legacy_input_manifest: LegacyInputManifest
    snapshot_import_request: SnapshotImportRequest
    partition_layout: dict[str, tuple[tuple[str, tuple[str, ...]], ...]]


def request_hash(request: SnapshotImportRequest) -> str:
    """The identity every ``FragmentRecord.import_request_hash`` carries.

    Excludes ``expected_head_snapshot_id``/``expected_head_generation``: they
    are the CAS/fencing envelope a caller expects to hold at commit time, not
    part of what was imported (the same judgement ``DataQuery.deadline``
    already makes: "execution metadata ... does not enter... identity"). A
    later, otherwise-identical import submitted against a since-advanced head
    must still be able to reuse every already-published row by this same
    hash — excluding these two fields is what makes that possible.
    """
    payload = to_document(request)
    payload.pop("expected_head_snapshot_id", None)
    payload.pop("expected_head_generation", None)
    return content_hash(payload)


def plan_import(source_root, *, scope: str, expected_head_snapshot_id: str | None,
                expected_head_generation: int, mapping: dict | None = None) -> ImportPlan:
    """Enumerate ``source_root`` and freeze one :class:`ImportPlan`.

    ``mapping`` defaults to :func:`legacy_adapter.build_legacy_mapping`; tests
    pass a modified copy the same way ``legacy_adapter``'s own tests do.
    """
    root = Path(source_root).resolve()
    mapping = mapping if mapping is not None else legacy_adapter.build_legacy_mapping()
    tables = mapping["tables"]

    table_sources: dict[str, tuple[LegacyFileRef, ...]] = {}
    table_contract_refs: dict[str, TableContractRef] = {}
    partition_layout: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {}
    all_refs: list[LegacyFileRef] = []

    for name in legacy_adapter.TIER2_DATASETS:
        refs, layout = _enumerate_curated_table(root, name)
        table_sources[name] = refs
        partition_layout[name] = layout
        table_contract_refs[name] = _contract_ref(tables[name])
        all_refs.extend(refs)

    for name, relative in ((
            "feature_panel", legacy_adapter.PANEL_RELATIVE_PATH),
            ("tier4_forecasts", legacy_adapter.TIER4_RELATIVE_PATH)):
        ref = _single_file_ref(root, relative)
        table_sources[name] = (ref,)
        partition_layout[name] = (("all", (relative,)),)
        table_contract_refs[name] = _contract_ref(tables[name])
        all_refs.append(ref)

    snapshot_ref = _single_file_ref(root, legacy_adapter.SNAPSHOT_RELATIVE_PATH)
    _check_snapshot_shape(root, snapshot_ref, mapping["legacy_snapshot_metadata"])
    all_refs.append(snapshot_ref)

    knowledge_mode = dict(mapping["knowledge_mode_by_table"])
    manifest = _build_manifest(scope, all_refs, table_contract_refs, knowledge_mode)
    request = SnapshotImportRequest(
        scope=scope, source_manifest_ref=manifest.manifest_id,
        source_manifest_hash=content_hash(to_document(manifest)),
        table_sources=table_sources, table_contract_refs=table_contract_refs,
        legacy_snapshot_source_ref=snapshot_ref, calendar_version=PENDING_CALENDAR_VERSION,
        source_priority_version=legacy_adapter.SOURCE_PRIORITY_VERSION,
        finality_receipt_refs=(), knowledge_mode_by_table=knowledge_mode,
        expected_head_snapshot_id=expected_head_snapshot_id,
        expected_head_generation=expected_head_generation)
    return ImportPlan(legacy_input_manifest=manifest, snapshot_import_request=request,
                      partition_layout=partition_layout)


def _contract_ref(doc: dict) -> TableContractRef:
    return TableContractRef(contract_id=doc["contract_id"], definition_hash=doc["definition_hash"])


def _build_manifest(scope: str, refs: list[LegacyFileRef], contract_refs: dict[str, TableContractRef],
                    knowledge_mode: dict[str, str]) -> LegacyInputManifest:
    fields = dict(
        file_refs=tuple(refs),
        table_contract_refs=tuple(f"{name}@{ref.contract_id}@{ref.definition_hash}"
                                  for name, ref in sorted(contract_refs.items())),
        registry_and_model_refs=(), calendar_ref=None, selected_session="",
        finality_receipt_refs=(), knowledge_mode_by_table=knowledge_mode,
        availability_evidence_refs=(), read_set_complete=True,
        capture_implementation_ref="snapshot_import_plan.v1")
    placeholder = LegacyInputManifest(manifest_id="pending", **fields)
    digest = content_hash(to_document(placeholder)).removeprefix(CONTENT_HASH_PREFIX)[:32]
    return LegacyInputManifest(manifest_id=f"snap_import_{scope}_{digest}", **fields)


# --------------------------------------------------------------------------
# filesystem enumeration
# --------------------------------------------------------------------------


def _hash_file(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return CONTENT_HASH_PREFIX + digest.hexdigest(), size


def _file_ref(root: Path, relative: str) -> LegacyFileRef:
    path = root / relative
    if path.is_symlink():
        raise errors.fail("INPUT_CHANGED", "declared legacy file is a symlink",
                  details={"path": relative})
    if not path.is_file():
        raise errors.fail("INPUT_CHANGED", "declared legacy file is missing",
                  details={"path": relative})
    digest, size = _hash_file(path)
    return LegacyFileRef(path=relative, content_hash=digest, byte_size=size)


def _single_file_ref(root: Path, relative: str) -> LegacyFileRef:
    return _file_ref(root, relative)


def _enumerate_curated_table(root: Path, table: str) -> tuple[tuple[LegacyFileRef, ...],
                                                               tuple[tuple[str, tuple[str, ...]], ...]]:
    table_dir = root / "data" / "curated" / table
    if table_dir.is_symlink():
        raise errors.fail("INPUT_CHANGED", "curated table directory is a symlink",
                  details={"table": table})
    if not table_dir.is_dir():
        raise errors.fail("INPUT_CHANGED", "curated table directory is missing",
                  details={"table": table})
    refs: list[LegacyFileRef] = []
    layout: list[tuple[str, tuple[str, ...]]] = []
    for year_entry in sorted(table_dir.iterdir(), key=lambda p: p.name):
        if year_entry.is_symlink():
            raise errors.fail("INPUT_CHANGED", "curated table year directory is a symlink",
                      details={"table": table, "path": year_entry.name})
        match = _YEAR_RE.match(year_entry.name)
        if not match or not year_entry.is_dir():
            raise errors.fail("CONTRACT_MISMATCH", "undeclared entry inside a curated table directory",
                      details={"table": table, "path": year_entry.name})
        year = match.group(1)
        year_refs: list[str] = []
        for part_entry in sorted(year_entry.iterdir(), key=lambda p: p.name):
            if part_entry.is_symlink():
                raise errors.fail("INPUT_CHANGED", "curated table partition file is a symlink",
                          details={"table": table, "path": f"year={year}/{part_entry.name}"})
            if not _PART_RE.match(part_entry.name) or not part_entry.is_file():
                raise errors.fail("CONTRACT_MISMATCH", "undeclared file inside a curated table year directory",
                          details={"table": table, "path": f"year={year}/{part_entry.name}"})
            relative = f"data/curated/{table}/year={year}/{part_entry.name}"
            refs.append(_file_ref(root, relative))
            year_refs.append(relative)
        layout.append((year, tuple(year_refs)))
    return tuple(refs), tuple(layout)


def _check_snapshot_shape(root: Path, ref: LegacyFileRef, metadata: dict) -> None:
    import json

    payload = json.loads((root / ref.path).read_text())
    if not isinstance(payload, dict):
        raise errors.fail("CONTRACT_MISMATCH", "legacy SNAPSHOT is not a JSON object")
    expected = set(metadata["expected_top_level_keys"])
    if set(payload) != expected:
        raise errors.fail("CONTRACT_MISMATCH", "legacy SNAPSHOT top-level keys do not match the reviewed shape",
                  details={"expected": sorted(expected), "actual": sorted(payload)})
