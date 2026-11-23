"""EXP-188 real-store Sources adapter, slices 1 and 2 (design section 13).

Slice 1 opens the catalog read-only, the artifact store, the sealed plan, the selection record, the admission and the chain index, and derives the roster inputs from the plan. Slice 2 adds the table-backed providers (`fetch`, `facts`, `plan_frame`, `table_hash`) from `size_pnl_tables`.

The sealed plan is consumed, not re-derived: it is loaded from the store and verified by content hash and by run id. No installed-code re-derivation runs here.
"""
import functools
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactError, ArtifactStore, DocumentError, content_hash, from_document, to_document
from engine.v2.ops.errors import fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments import size_pnl_driver, size_pnl_gprod, size_pnl_inputs, size_pnl_ladder, size_pnl_readers, size_pnl_tables
from experiments.magnitude_search_plan import PLAN_V2, PLAN_V3
from experiments.magnitude_search_records import read_record
from experiments.native_registration import read_native_registration

_KEYS = ("catalog", "store_root", "spec", "plan_ref", "snapshot_id", "crush_snapshot_id", "production_hash")
_GATE_KEYS = ("gate_fold", "gate_inputs")
_LIMIT = 1 << 20
MAX_OBJECT_BYTES = 1 << 28
MAX_PLAN_BYTES = 64 * 1024 * 1024
_PLAN_SCHEMAS = (PLAN_V2, PLAN_V3)
# Measured on the real v14 snapshot (11 tables): 256,680 fragments (computed_moves 249,177), about 120 MB of data_fragments rows
# and 59 MB of data_objects rows. These caps leave about 4x headroom; scans still stop at cap + 1.
MAX_SNAPSHOT_TABLES = 256
MAX_SNAPSHOT_FRAGMENTS = 1_048_576
MAX_SNAPSHOT_EVIDENCE_BYTES = 512 * 1024 * 1024
# OCTET_LENGTH reads the length from the record header without loading the value (SQLite 3.43+). Measured on SQLite 3.46.1
# with a 60 MiB TEXT value: octet_length 0.0001 s vs cast+length 0.03 s. The CAST(... AS BLOB) fallback only applies on SQLite older than 3.43.
_FAST_OCTETS = sqlite3.sqlite_version_info >= (3, 43, 0)
_BLEND_NAMES = {"M4": "size_b1_v1", "M5": "size_b2_v1"}
_INVALID, _CONFLICT = "INVALID_EXPERIMENT_SPEC", "EXPERIMENT_IDENTITY_CONFLICT"


def _refuse(reason, message):
    return fail(_INVALID, message, details={"reason": reason})


def _read_json(path, reason):
    """Read a regular, non-symlink file of at most _LIMIT bytes and parse it as JSON. The path never reaches a message."""
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > _LIMIT:
                raise ValueError("not a regular file of at most 1 MiB")
            data = handle.read(_LIMIT + 1)
        if len(data) > _LIMIT:
            raise ValueError("oversize")
        return json.loads(data.decode("utf-8"))
    except (OSError, ValueError, RecursionError):
        raise _refuse(reason, "input is unreadable or not valid JSON") from None


def _config(path):
    if path is None:
        raise _refuse("sources_config_invalid", "--sources-config is required")
    doc = _read_json(path, "sources_config_invalid")
    if not (isinstance(doc, dict) and set(_KEYS) <= set(doc) <= set(_KEYS) | set(_GATE_KEYS)
            and all(isinstance(doc[k], str) and doc[k].strip() for k in _KEYS)):
        raise _refuse("sources_config_invalid", "sources config must map the required keys (and optionally gate_fold and gate_inputs) to non-blank strings")
    return doc


def derive_roster_inputs(plan):
    """Blend ids from the plan's named blends, and the unique extra-refit candidate per pair role. Ambiguity omits the role."""
    blends = plan.get("blends", [])
    blend_ids = {}
    for role, name in _BLEND_NAMES.items():
        matches = [b["blend_id"] for b in blends if b.get("name") == name]
        if len(matches) == 1:
            blend_ids[role] = matches[0]
    selection = plan.get("selection")
    extra = set(selection.get("extra_refits") or ()) if isinstance(selection, dict) else set()
    pool = [c for c in plan["candidates"] if c["candidate_id"] in extra]
    pair_candidates = {}
    for role in size_pnl_inputs.PAIR_ROLES:
        required = size_pnl_inputs.PAIR_TRANSFORM.get(role)
        matches = [c["candidate_id"] for c in pool if size_pnl_inputs._recipe_matches(c["recipe"], role)
                   and (required is None or size_pnl_inputs._transform(c) == required)]
        if len(matches) == 1:
            pair_candidates[role] = matches[0]
    return pair_candidates, blend_ids


def derive_panel_id(plan):
    """The panel id is the sealed plan's candidate source binding (the size_panel registration binding). Refuses unless every candidate carries the same non-blank string."""
    ids = {c.get("source_binding_id") for c in plan.get("candidates", []) if isinstance(c, dict)}
    value = next(iter(ids)) if len(ids) == 1 else None
    if not (isinstance(value, str) and value.strip()):
        raise fail(_CONFLICT, "the sealed plan has no single panel source binding", details={"reason": "panel_binding_ambiguous"})
    return value


def load_plan(store, registration, ref):
    """Load the sealed plan from the store. The bytes are verified by content hash (read_verified) and the anchor by run id; nothing is re-derived."""
    if ref.schema_ref not in _PLAN_SCHEMAS:
        raise fail(_CONFLICT, "the sealed plan has an unknown schema", details={"reason": "plan_schema_unknown"})
    if ref.byte_size > MAX_PLAN_BYTES:
        raise fail(_CONFLICT, "the sealed plan is too large", details={"reason": "plan_too_large"})
    try:
        data = store.read_verified(ref)
    except (ArtifactError, OSError) as exc:
        if getattr(exc, "code", None) == "INTEGRITY_FAILED":
            raise fail(_CONFLICT, "the sealed plan failed content verification", details={"reason": "plan_hash_mismatch"}) from None
        raise fail("RESOURCE_UNAVAILABLE", "the sealed plan could not be read", details={"reason": "plan_unreadable"}) from None
    try:
        plan = json.loads(data)
    except (ValueError, RecursionError, UnicodeDecodeError):
        raise fail(_CONFLICT, "the sealed plan is not valid JSON", details={"reason": "plan_malformed"}) from None
    if not isinstance(plan, dict) or plan.get("schema_version") != ref.schema_ref:
        raise fail(_CONFLICT, "the sealed plan does not match its reference schema", details={"reason": "plan_malformed"})
    anchor = plan.get("anchor")
    if not (isinstance(anchor, dict) and anchor.get("run_id") == registration.run_id):
        raise fail(_CONFLICT, "the sealed plan is anchored to another run", details={"reason": "plan_anchor_mismatch"})
    if anchor.get("variant_id") != registration.variant_id:
        raise fail(_CONFLICT, "the sealed plan is anchored to another variant", details={"reason": "plan_anchor_mismatch"})
    implementation = plan.get("implementation")
    expected_implementation = {"source_closure": registration.document.get("source_closure"),
                               "environment": registration.document.get("environment")}
    if not (isinstance(implementation, dict) and implementation == expected_implementation):
        raise fail(_CONFLICT, "the sealed plan implementation is not the registration's", details={"reason": "plan_registration_mismatch"})
    plan_id = plan.get("plan_id")
    if not (isinstance(plan_id, str) and plan_id == content_hash({k: v for k, v in plan.items() if k != "plan_id"})):
        raise fail(_CONFLICT, "the sealed plan body does not match its plan_id", details={"reason": "plan_id_mismatch"})
    if not (isinstance(plan.get("candidates"), list) and isinstance(plan.get("selection"), dict)):
        raise fail(_CONFLICT, "the sealed plan lacks a candidate list or a selection", details={"reason": "plan_malformed"})
    return plan


def _open(cfg):
    if not Path(cfg["store_root"]).is_dir():
        raise fail("RESOURCE_UNAVAILABLE", "existing artifact store required", details={"reason": "store_missing"})
    try:
        conn = sqlite3.connect(Path(cfg["catalog"]).resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
    except sqlite3.Error:
        raise fail("RESOURCE_UNAVAILABLE", "catalog unavailable", details={"reason": "catalog_unavailable"}) from None
    conn.row_factory = sqlite3.Row
    try:
        return conn, ArtifactStore(cfg["store_root"])
    except BaseException:
        conn.close()
        raise


def _columns(conn, table):
    """Column names of one of the fixed table literals used below (empty when the table is absent)."""
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _scan_bytes(conn, table, alias, join_where, params, budget):
    """UTF-8 bytes of every column of `table` (read as `alias`) over the rows `join_where` selects. NULL counts as 0.

    Streams one row at a time and stops at the first row that takes the total past `budget`, closing the cursor; the returned total is then over budget.
    """
    template = 'COALESCE(OCTET_LENGTH(%s."%s"), 0)' if _FAST_OCTETS else 'COALESCE(LENGTH(CAST(%s."%s" AS BLOB)), 0)'
    terms = [template % (alias, c.replace('"', '""')) for c in _columns(conn, table)]
    expr = " + ".join(terms) or "0"
    cursor = conn.execute("SELECT %s FROM %s" % (expr, join_where), params)
    total = 0
    for (row_bytes,) in cursor:
        total += row_bytes
        if total > budget:
            cursor.close()
            return total
    return total


def _bound_snapshot(conn, snapshot_id):
    """Refuse a pinned snapshot whose catalog metadata exceeds the table, fragment or evidence caps. Read-only counts only."""
    try:
        # Each count stops at cap + 1 rows, so the scan work is bounded by the caps.
        tables = conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM data_snapshot_tables WHERE snapshot_id = ? LIMIT ?)",
            (snapshot_id, MAX_SNAPSHOT_TABLES + 1)).fetchone()[0]
        fragments = conn.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM data_version_fragments vf JOIN data_snapshot_tables st "
            "ON st.dataset_version_id = vf.dataset_version_id WHERE st.snapshot_id = ? LIMIT ?)",
            (snapshot_id, MAX_SNAPSHOT_FRAGMENTS + 1)).fetchone()[0]
        if tables > MAX_SNAPSHOT_TABLES or fragments > MAX_SNAPSHOT_FRAGMENTS:
            raise fail(_CONFLICT, "snapshot catalog metadata is too large", details={"reason": "snapshot_metadata_too_large"})
        params = (snapshot_id,)
        evidence = 0

        def add(table, alias, join_where):
            # The running budget makes each scan stop at the first row that crosses the cap; the check then refuses.
            nonlocal evidence
            evidence += _scan_bytes(conn, table, alias, join_where, params, MAX_SNAPSHOT_EVIDENCE_BYTES - evidence)
            if evidence > MAX_SNAPSHOT_EVIDENCE_BYTES:
                raise fail(_CONFLICT, "snapshot catalog metadata is too large", details={"reason": "snapshot_metadata_too_large"})

        add("data_snapshots", "s", "data_snapshots s WHERE s.snapshot_id = ?")
        add("data_snapshot_tables", "st", "data_snapshot_tables st WHERE st.snapshot_id = ?")
        add(
            "data_dataset_versions", "dv",
            "data_dataset_versions dv JOIN data_snapshot_tables st ON st.dataset_version_id = dv.dataset_version_id WHERE st.snapshot_id = ?")
        add(
            "data_fragments", "f",
            "data_fragments f JOIN data_version_fragments vf ON vf.fragment_id = f.fragment_id "
            "JOIN data_snapshot_tables st ON st.dataset_version_id = vf.dataset_version_id WHERE st.snapshot_id = ?")
        add(
            "data_objects", "o",
            "data_objects o JOIN data_fragments f ON f.object_id = o.object_id "
            "JOIN data_version_fragments vf ON vf.fragment_id = f.fragment_id "
            "JOIN data_snapshot_tables st ON st.dataset_version_id = vf.dataset_version_id WHERE st.snapshot_id = ?")
        add(
            "data_contracts", "c",
            "data_contracts c JOIN data_dataset_versions dv ON dv.contract_id = c.contract_id "
            "JOIN data_snapshot_tables st ON st.dataset_version_id = dv.dataset_version_id WHERE st.snapshot_id = ?")
        add(
            "data_version_fragments", "vf",
            "data_version_fragments vf JOIN data_snapshot_tables st ON st.dataset_version_id = vf.dataset_version_id WHERE st.snapshot_id = ?")
        add(
            "data_contracts", "c",
            "data_contracts c WHERE c.contract_id IN (SELECT f.contract_id FROM data_fragments f "
            "JOIN data_version_fragments vf ON vf.fragment_id = f.fragment_id "
            "JOIN data_snapshot_tables st ON st.dataset_version_id = vf.dataset_version_id WHERE st.snapshot_id = ?)")
    except sqlite3.Error:
        raise fail("RESOURCE_UNAVAILABLE", "catalog unavailable", details={"reason": "catalog_unavailable"}) from None


def _snapshot(repository, snapshot_id):
    try:
        return repository.resolve(snapshot_id)
    except DataError as exc:
        if exc.code != "SNAPSHOT_NOT_FOUND":
            raise
        raise fail(_CONFLICT, "pinned snapshot cannot be resolved", details={"reason": "snapshot_missing"}) from None


def _snapshot_mismatch():
    return fail(_CONFLICT, "the configured snapshot is not the registration's pinned snapshot", details={"reason": "snapshot_mismatch"})


def _derived(derive, plan):
    try:
        return derive(plan)
    except (KeyError, TypeError, AttributeError, ValueError):
        raise fail(_CONFLICT, "the sealed plan has malformed candidates or selection", details={"reason": "plan_malformed"}) from None


def assemble(*, plan, read, selection, expected, panel_id, admit, admitted, snapshot_id, load_index, tables, production_hash,
             crush_table_hash, gate=None):
    pair_candidates, blend_ids = _derived(derive_roster_inputs, plan)
    return size_pnl_driver.Sources(
        plan=plan, read=read, selection=selection, pair_candidates=pair_candidates, blend_ids=blend_ids,
        expected=expected, panel_id=panel_id, admit=admit, admitted=admitted, fetch=tables.fetch,
        production_hash=production_hash, production_table_hash=tables.table_hash, plan_frame=tables.plan_frame,
        load_index=load_index, facts=tables.facts, snapshot_id=snapshot_id, crush_table_hash=crush_table_hash,
        exclusions=tables.exclusions, gate=gate)


def build_sources(args):
    """Public factory for `run.py --sources`. The connection stays open for the run's `read` closure; it closes on any refusal."""
    cfg = _config(getattr(args, "sources_config", None))
    expected = getattr(args, "expected_identities", None)
    if not (isinstance(expected, dict) and all(isinstance(expected.get(k), str) and expected[k].strip()
                                               for k in ("plan_id", "registration_id", "seal_id", "panel_id"))):
        raise _refuse("expected_invalid", "expected identities must hold plan_id, registration_id, seal_id and panel_id")
    try:
        spec = experiment_spec_from_document(_read_json(cfg["spec"], "input_unreadable"))
        ref = from_document(ArtifactRef, _read_json(cfg["plan_ref"], "input_unreadable"))
    except (DocumentError, TypeError):
        raise _refuse("input_unreadable", "a SIZE input document is invalid") from None
    gate = size_pnl_gprod.load_gate({k: cfg[k] for k in _GATE_KEYS if k in cfg})  # before the catalog opens: a refused gate leaves no connection
    conn, store = _open(cfg)
    try:
        registration = read_native_registration(conn, store, spec)
        pinned = registration.document.get("snapshot")
        if not (isinstance(pinned, dict) and pinned.get("snapshot_id") == cfg["snapshot_id"]):
            raise _snapshot_mismatch()
        repository = Repository(conn, store, pin_snapshots=True)
        _bound_snapshot(conn, cfg["snapshot_id"])
        snapshot = _snapshot(repository, cfg["snapshot_id"])
        if to_document(snapshot) != pinned:
            raise _snapshot_mismatch()
        _bound_snapshot(conn, cfg["crush_snapshot_id"])
        crush_snapshot = _snapshot(repository, cfg["crush_snapshot_id"])
        plan = load_plan(store, registration, ref)
        panel_id = _derived(derive_panel_id, plan)
        read = functools.partial(read_record, conn, store, registration.run_id)
        selection = read("selection")
        if not (isinstance(selection, dict) and isinstance(selection.get("selections"), list)):
            raise fail(_CONFLICT, "the selection record is missing", details={"reason": "selection_missing"})
        month = registration.document["holdouts"]["holdout_as_of_month"]
        admit = size_pnl_inputs.population_admit(repository, snapshot, month, max_object_bytes=MAX_OBJECT_BYTES)
        admitted = size_pnl_readers.admit_events(repository, snapshot, month, max_object_bytes=MAX_OBJECT_BYTES)
        tables = size_pnl_tables.tables(repository, snapshot, admitted, crush_snapshot=crush_snapshot,
                                        max_object_bytes=MAX_OBJECT_BYTES, expected_hash=cfg["production_hash"])
        load_index = size_pnl_ladder.default_load_index(repository, snapshot)
        sources = assemble(plan=plan, read=read, selection=selection, expected=expected, panel_id=panel_id,
                           admit=admit, admitted=admitted, snapshot_id=cfg["snapshot_id"], load_index=load_index,
                           tables=tables, production_hash=cfg["production_hash"],
                           crush_table_hash=tables.crush_table_hash, gate=gate)
    except sqlite3.Error:
        conn.close()
        raise fail("RESOURCE_UNAVAILABLE", "catalog unavailable", details={"reason": "catalog_unavailable"}) from None
    except BaseException:
        conn.close()
        raise
    print(f"[pnl_sources] opened snapshot={cfg['snapshot_id']} crush_snapshot={cfg['crush_snapshot_id']} admitted={len(admitted['events'])} gate={'G_PROD' if gate else 'none'}",
          file=sys.stderr)
    return sources
