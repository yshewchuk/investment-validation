"""EXP-188 real-store Sources adapter, slice 1: synthetic fixtures only (no store, no network, no sqlite file outside tmp_path)."""
import json
import os
import sqlite3
import types

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.ops.errors import OpsError
from experiments import size_pnl_driver as driver
from experiments import size_pnl_inputs as inputs
from experiments import size_pnl_ladder as ladder
from experiments import size_pnl_readers as readers
from experiments import size_pnl_sources as src

EXPECTED = {"plan_id": "p1", "registration_id": "r1", "seal_id": "s1", "panel_id": "P"}
SELECTION = {"selections": []}
PINNED = {"snapshot_id": "snap-1"}


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _write(path, doc):
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return str(path)


def _cfg(tmp_path):
    return {"catalog": str(tmp_path / "catalog.sqlite"), "store_root": str(tmp_path), "spec": str(tmp_path / "spec.json"),
            "request": str(tmp_path / "request.json"), "plan_ref": str(tmp_path / "plan_ref.json"),
            "snapshot_id": "snap-1", "production_hash": "h"}


def _hgb(loss):
    return {"schema_version": "magnitude_ridge_hgb_recipe.v2", "estimator": {"class": "HistGradientBoostingRegressor", "params": {
        "loss": loss, "learning_rate": 0.03, "l2_regularization": 10, "max_iter": 3000, "max_leaf_nodes": 15, "min_samples_leaf": 300}}}


def _en(schema):
    return {"schema_version": schema, "estimator": {"class": "ElasticNet", "params": {"alpha": 0.03, "l1_ratio": 0.75}},
            "preprocessing": {"spline": {"n_knots": 6, "degree": 3}}}


def _mlp(schema):
    return {"schema_version": schema, "profile": "chrono15_relu64x32_l2_1", "estimator": {"class": "MLPRegressor", "params": {"hidden_layer_sizes": [64, 32]}}}


PAIR_CANDIDATES = [
    {"candidate_id": "c-hsq", "recipe": _hgb("squared_error")},
    {"candidate_id": "c-habs", "recipe": _hgb("absolute_error")},
    {"candidate_id": "c-eraw", "recipe": _en("magnitude_elasticnet_spline_recipe.v1")},
    {"candidate_id": "c-elog", "recipe": _en("magnitude_elasticnet_spline_recipe.v2")},
    {"candidate_id": "c-nraw", "recipe": _mlp("magnitude_mlp_recipe.v1")},
    {"candidate_id": "c-nlog", "recipe": _mlp("magnitude_mlp_recipe.v2")},
]
FULL_PAIR_MAP = {"H-sq": "c-hsq", "H-abs": "c-habs", "E-raw": "c-eraw", "E-log": "c-elog", "N-raw": "c-nraw", "N-log": "c-nlog"}


def _plan(candidates=PAIR_CANDIDATES, blends=(), extra=None):
    ids = [c["candidate_id"] for c in candidates]
    return {"plan_id": "p1", "anchor": {"run_id": "r1"}, "horizons": [5, 3, 1], "candidates": list(candidates),
            "blends": list(blends), "selection": {"extra_refits": ids if extra is None else extra}}


# --- (1) _config -------------------------------------------------------------

def test_config_accepts_valid_file(tmp_path):
    cfg = _cfg(tmp_path)
    assert src._config(_write(tmp_path / "c.json", cfg)) == cfg


def test_config_refuses_missing_argument():
    with pytest.raises(OpsError) as exc:
        src._config(None)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "sources_config_invalid")


@pytest.mark.parametrize("case", ["unknown_key", "missing_key", "blank_value", "non_str_value", "not_dict"])
def test_config_refuses_bad_documents(tmp_path, case):
    cfg = _cfg(tmp_path)
    doc = {
        "unknown_key": {**cfg, "extra": "x"},
        "missing_key": {k: v for k, v in cfg.items() if k != "spec"},
        "blank_value": {**cfg, "spec": "   "},
        "non_str_value": {**cfg, "snapshot_id": 3},
        "not_dict": [cfg],
    }[case]
    with pytest.raises(OpsError) as exc:
        src._config(_write(tmp_path / "c.json", doc))
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "sources_config_invalid")


def test_config_refuses_oversized_file(tmp_path):
    with pytest.raises(OpsError) as exc:
        src._config(_write(tmp_path / "c.json", "x" * ((1 << 20) + 1)))
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "sources_config_invalid")


def test_config_refuses_symlink(tmp_path):
    target = _write(tmp_path / "c.json", _cfg(tmp_path))
    os.symlink(target, tmp_path / "link.json")
    with pytest.raises(OpsError) as exc:
        src._config(str(tmp_path / "link.json"))
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "sources_config_invalid")


def test_config_refuses_directory(tmp_path):
    (tmp_path / "d").mkdir()
    with pytest.raises(OpsError) as exc:
        src._config(str(tmp_path / "d"))
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "sources_config_invalid")


# --- (2) derive_roster_inputs ------------------------------------------------

def test_blend_names_map_to_roles():
    blends = [{"blend_id": "b1", "name": "size_b1_v1"}, {"blend_id": "b2", "name": "size_b2_v1"}]
    assert src.derive_roster_inputs(_plan(blends=blends))[1] == {"M4": "b1", "M5": "b2"}


def test_unknown_and_duplicated_blend_names_omitted():
    blends = [{"blend_id": "x", "name": "other"}, {"blend_id": "b1", "name": "size_b1_v1"}, {"blend_id": "b1b", "name": "size_b1_v1"}]
    assert src.derive_roster_inputs(_plan(blends=blends))[1] == {}


def test_each_pair_role_maps_to_its_unique_candidate():
    pair_candidates, _ = src.derive_roster_inputs(_plan())
    assert pair_candidates == FULL_PAIR_MAP


def test_zero_matches_omitted():
    extra = [c["candidate_id"] for c in PAIR_CANDIDATES if c["candidate_id"] != "c-hsq"]
    pair_candidates, _ = src.derive_roster_inputs(_plan(extra=extra))
    assert "H-sq" not in pair_candidates and pair_candidates["H-abs"] == "c-habs"


def test_two_matches_omitted():
    candidates = PAIR_CANDIDATES + [{"candidate_id": "c-hsq2", "recipe": _hgb("squared_error")}]
    pair_candidates, _ = src.derive_roster_inputs(_plan(candidates=candidates))
    assert "H-sq" not in pair_candidates and pair_candidates["H-abs"] == "c-habs"


def test_candidate_outside_extra_refits_is_ignored():
    candidates = PAIR_CANDIDATES + [{"candidate_id": "c-hsq-off", "recipe": _hgb("squared_error")}]
    extra = [c["candidate_id"] for c in PAIR_CANDIDATES]
    pair_candidates, _ = src.derive_roster_inputs(_plan(candidates=candidates, extra=extra))
    assert pair_candidates["H-sq"] == "c-hsq"


def test_transform_mismatch_omitted():
    # A log1p (v2) elastic-net cannot fill the raw (E-raw) role; it fills E-log.
    candidates = [{"candidate_id": "c-eraw", "recipe": _en("magnitude_elasticnet_spline_recipe.v2")}]
    pair_candidates, _ = src.derive_roster_inputs(_plan(candidates=candidates))
    assert pair_candidates == {"E-log": "c-eraw"}


def test_selection_without_extra_refits_omits_all_pairs():
    plan = _plan(blends=[{"blend_id": "b1", "name": "size_b1_v1"}])
    plan["selection"] = {}
    pair_candidates, blend_ids = src.derive_roster_inputs(plan)
    assert pair_candidates == {}
    assert blend_ids == {"M4": "b1"}


def test_plan_without_selection_omits_all_pairs():
    plan = _plan(blends=[{"blend_id": "b1", "name": "size_b1_v1"}])
    del plan["selection"]
    pair_candidates, blend_ids = src.derive_roster_inputs(plan)
    assert pair_candidates == {}
    assert blend_ids == {"M4": "b1"}


# --- (3) _open ---------------------------------------------------------------

def test_open_refuses_missing_store(tmp_path):
    with pytest.raises(OpsError) as exc:
        src._open({**_cfg(tmp_path), "store_root": str(tmp_path / "absent")})
    assert _refusal(exc) == ("RESOURCE_UNAVAILABLE", "store_missing")


def test_catalog_connection_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "ArtifactStore", lambda root: ("store", root))
    db = tmp_path / "catalog.sqlite"
    seed = sqlite3.connect(str(db))
    seed.execute("create table t (x)")
    seed.commit()
    seed.close()
    conn, store = src._open({**_cfg(tmp_path), "catalog": str(db)})
    try:
        assert store == ("store", str(tmp_path))
        assert conn.execute("select count(*) from t").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("insert into t values (1)")
    finally:
        conn.close()


def test_missing_catalog_refused(tmp_path):
    with pytest.raises(OpsError) as exc:
        src._open({**_cfg(tmp_path), "catalog": str(tmp_path / "missing.sqlite")})
    assert _refusal(exc) == ("RESOURCE_UNAVAILABLE", "catalog_unavailable")


# --- (4) _snapshot -----------------------------------------------------------

class _Repo:
    def resolve(self, snapshot_id):
        raise data_fail("SNAPSHOT_NOT_FOUND", "gone")


class _CorruptRepo:
    def resolve(self, snapshot_id):
        raise data_fail("MANIFEST_CORRUPT", "bad")


class _ValueErrorRepo:
    def resolve(self, snapshot_id):
        raise ValueError("malformed")


class _KeyErrorRepo:
    def resolve(self, snapshot_id):
        raise KeyError("malformed")


def test_snapshot_resolve_failure_refused():
    with pytest.raises(OpsError) as exc:
        src._snapshot(_Repo(), "snap-1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_missing")


def test_snapshot_other_data_error_reraised():
    with pytest.raises(DataError) as exc:
        src._snapshot(_CorruptRepo(), "snap-1")
    assert exc.value.code == "MANIFEST_CORRUPT"


def test_snapshot_value_error_propagates():
    with pytest.raises(ValueError):
        src._snapshot(_ValueErrorRepo(), "snap-1")


def test_snapshot_key_error_propagates():
    with pytest.raises(KeyError):
        src._snapshot(_KeyErrorRepo(), "snap-1")


# --- (4b) _bound_snapshot ----------------------------------------------------

def _catalog(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "meta.sqlite"))
    conn.executescript("""
        create table data_snapshots (snapshot_id, finality_receipt_refs_json, knowledge_mode_by_table_json);
        create table data_snapshot_tables (snapshot_id, dataset_version_id, table_name);
        create table data_dataset_versions (dataset_version_id, evidence_json, partition_logical_hashes_json, contract_id);
        create table data_fragments (fragment_id, partition_key, key_bounds_json, time_bounds_json, input_receipt_refs_json, object_id);
        create table data_version_fragments (dataset_version_id, fragment_id);
        create table data_objects (object_id, storage_key);
        create table data_contracts (contract_id, definition_json);
    """)
    return conn


def _seed(conn, tables=1, fragments=0, evidence="ab", hashes="cd"):
    # With the defaults each seeded table adds 7 bytes: its snapshot_tables row ('s1', 'dvN', 'tN') and its
    # dataset_versions row ('dvN', evidence, hashes); 'ab' and 'cd' are 2 bytes each.
    for i in range(tables):
        conn.execute(
            "insert into data_dataset_versions (dataset_version_id, evidence_json, partition_logical_hashes_json) values (?, ?, ?)",
            (f"dv{i}", evidence, hashes))
        conn.execute("insert into data_snapshot_tables values ('s1', ?, ?)", (f"dv{i}", f"t{i}"))
    for i in range(fragments):
        conn.execute("insert into data_version_fragments values ('dv0', ?)", (f"f{i}",))
    conn.commit()


def test_bound_snapshot_within_caps_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_TABLES", 2)
    monkeypatch.setattr(src, "MAX_SNAPSHOT_FRAGMENTS", 3)
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 100)
    conn = _catalog(tmp_path)
    _seed(conn, tables=2, fragments=3)
    assert src._bound_snapshot(conn, "s1") is None
    conn.close()


def test_bound_snapshot_too_many_tables_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_TABLES", 2)
    conn = _catalog(tmp_path)
    _seed(conn, tables=3)
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_too_many_fragments_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_FRAGMENTS", 2)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1, fragments=3)
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_evidence_over_cap_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 3)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1, evidence="ab", hashes="cd")
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_fragment_json_over_evidence_cap_refused(tmp_path, monkeypatch):
    # Tables alone total 14 bytes (under 20); the fragment row adds 'f0' + 8 bytes of key_bounds_json = 24 (over 20).
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 20)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1, fragments=1, evidence="ab", hashes="cd")
    conn.execute("insert into data_fragments values ('f0', NULL, ?, NULL, NULL, NULL)", ("k" * 8,))
    conn.commit()
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_counts_utf8_bytes_not_characters(tmp_path, monkeypatch):
    # evidence is 10 characters but 20 UTF-8 bytes: 7 (snapshot_tables) + 3 (dv id) + 20 = 30 bytes, but only 20 characters.
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 25)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1, evidence="é" * 10, hashes="")
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_finality_receipt_refs_over_evidence_cap_refused(tmp_path, monkeypatch):
    # Without the snapshot row the total is 10 bytes (cap 10, passes); the finality refs add 's1' + 24 bytes.
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 10)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1, evidence="", hashes="")
    conn.execute("insert into data_snapshots values ('s1', ?, NULL)", ('["' + "r" * 20 + '"]',))
    conn.commit()
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_knowledge_mode_json_over_cap_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 50)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1)
    conn.execute("insert into data_snapshots values ('s1', NULL, ?)", ("k" * 100,))
    conn.commit()
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_partition_key_over_cap_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 50)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1, fragments=1)
    conn.execute("insert into data_fragments values ('f0', ?, NULL, NULL, NULL, NULL)", ("p" * 100,))
    conn.commit()
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_object_storage_key_over_cap_refused(tmp_path, monkeypatch):
    # Without data_objects the total is 18 bytes (under 50); the object row, linked to fragment f0 by object_id, adds 'o0' + 100 bytes.
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 50)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1, fragments=1)
    conn.execute("insert into data_fragments values ('f0', NULL, NULL, NULL, NULL, 'o0')")
    conn.execute("insert into data_objects values ('o0', ?)", ("o" * 100,))
    conn.commit()
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_contract_definition_over_cap_refused(tmp_path, monkeypatch):
    # Without data_contracts the total is 16 bytes (under 50: 7 + 7 + the 'c0' contract_id on dv0); the contract row linked through dv0 adds 'c0' + 100 bytes of definition_json.
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 50)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1)
    conn.execute("update data_dataset_versions set contract_id = 'c0' where dataset_version_id = 'dv0'")
    conn.execute("insert into data_contracts values ('c0', ?)", ("k" * 100,))
    conn.commit()
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_object_not_linked_to_snapshot_not_counted(tmp_path, monkeypatch):
    # Snapshot total is 21 bytes (under 50). Object o1 is oversized but its fragment f1 is not in the snapshot's version fragments, so it is not counted.
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 50)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1, fragments=1)
    conn.execute("insert into data_fragments values ('f0', NULL, NULL, NULL, NULL, 'o0')")
    conn.execute("insert into data_objects values ('o0', 's')")
    conn.execute("insert into data_fragments values ('f1', NULL, NULL, NULL, NULL, 'o1')")
    conn.execute("insert into data_objects values ('o1', ?)", ("o" * 100,))
    conn.commit()
    assert src._bound_snapshot(conn, "s1") is None
    conn.close()


@pytest.mark.parametrize("table", ["data_snapshots", "data_fragments", "data_dataset_versions"])
def test_bound_snapshot_missing_table_refused(tmp_path, table):
    conn = _catalog(tmp_path)
    _seed(conn, tables=1, fragments=1)
    conn.execute(f"drop table {table}")
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("RESOURCE_UNAVAILABLE", "catalog_unavailable")
    conn.close()


def test_bound_snapshot_unknown_snapshot_passes_with_zero_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_TABLES", 0)
    monkeypatch.setattr(src, "MAX_SNAPSHOT_FRAGMENTS", 0)
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 0)
    conn = _catalog(tmp_path)
    _seed(conn, tables=2, fragments=2)
    assert src._bound_snapshot(conn, "nope") is None
    conn.close()


def _no_byte_scan(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("byte sum ran before the row-count refusal")
    monkeypatch.setattr(src, "_row_bytes", fail_if_called)
    monkeypatch.setattr(src, "_columns", fail_if_called)


def test_bound_snapshot_table_count_over_cap_refuses_before_byte_sums(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_TABLES", 2)
    monkeypatch.setattr(src, "MAX_SNAPSHOT_FRAGMENTS", 100)
    conn = _catalog(tmp_path)
    _seed(conn, tables=3)
    _no_byte_scan(monkeypatch)
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_fragment_count_over_cap_refuses_before_byte_sums(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_TABLES", 100)
    monkeypatch.setattr(src, "MAX_SNAPSHOT_FRAGMENTS", 2)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1, fragments=3)
    _no_byte_scan(monkeypatch)
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    conn.close()


def test_bound_snapshot_at_exact_caps_reaches_byte_sums(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_TABLES", 2)
    monkeypatch.setattr(src, "MAX_SNAPSHOT_FRAGMENTS", 3)
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 10**9)
    conn = _catalog(tmp_path)
    _seed(conn, tables=2, fragments=3)
    calls = []
    real_row_bytes = src._row_bytes

    def counting_row_bytes(*args, **kwargs):
        calls.append(args[1])
        return real_row_bytes(*args, **kwargs)

    monkeypatch.setattr(src, "_row_bytes", counting_row_bytes)
    assert src._bound_snapshot(conn, "s1") is None
    assert calls, "expected the byte sums to run at exactly the caps"
    conn.close()


def test_bound_snapshot_refuses_after_first_table_without_scanning_later_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(src, "MAX_SNAPSHOT_EVIDENCE_BYTES", 100)
    conn = _catalog(tmp_path)
    _seed(conn, tables=1)
    calls = []

    def big_first_row_bytes(conn_, table, alias, join_where, params):
        calls.append(table)
        return 10**6

    monkeypatch.setattr(src, "_row_bytes", big_first_row_bytes)
    with pytest.raises(OpsError) as exc:
        src._bound_snapshot(conn, "s1")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    assert len(calls) == 1
    conn.close()


# --- (5) assemble ------------------------------------------------------------

def test_assemble_returns_sources_accepted_by_check_sources():
    admitted = readers.Admitted({"events": pd.DataFrame({"event_id": ["e1"], "event_date": pd.to_datetime(["2025-01-02"])}), "excluded": {}})
    sources = src.assemble(plan=_plan(), read=lambda slot: None, selection=SELECTION, expected=EXPECTED, panel_id="P",
                           admit=lambda ids: {"admitted": sorted(ids), "dropped": []}, admitted=admitted,
                           snapshot_id="snap-1", load_index=lambda keys: None)
    assert driver.check_sources(sources) is sources
    assert sources.pair_candidates == FULL_PAIR_MAP and sources.production_hash == "" and sources.production_table_hash == ""
    for call in (lambda: sources.fetch(["event_id"], ["e1"]), lambda: sources.facts(["e1"])):
        with pytest.raises(OpsError) as exc:
            call()
        assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


def test_assemble_plan_frame_refuses_source_table_missing():
    admitted = readers.Admitted({"events": pd.DataFrame({"event_id": ["e1"], "event_date": pd.to_datetime(["2025-01-02"])}), "excluded": {}})
    sources = src.assemble(plan=_plan(), read=lambda slot: None, selection=SELECTION, expected=EXPECTED, panel_id="P",
                           admit=lambda ids: {"admitted": sorted(ids), "dropped": []}, admitted=admitted,
                           snapshot_id="snap-1", load_index=lambda keys: None)
    with pytest.raises(OpsError) as exc:
        sources.plan_frame.columns
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")
    with pytest.raises(OpsError) as exc:
        ladder.check_plan(admitted, sources.plan_frame)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


# --- (6) build_sources end to end (every store touch monkeypatched) -----------

class _Conn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def env(tmp_path, monkeypatch):
    conn, slots = _Conn(), {"selection": SELECTION}
    for name in ("spec.json", "request.json", "plan_ref.json"):
        _write(tmp_path / name, {"doc": name})
    config = _write(tmp_path / "config.json", _cfg(tmp_path))
    monkeypatch.setattr(src, "experiment_spec_from_document", lambda doc: ("spec", doc))
    monkeypatch.setattr(src, "from_document", lambda cls, doc: ("ref", doc))
    monkeypatch.setattr(src, "_open", lambda cfg: (conn, "store"))
    monkeypatch.setattr(src, "read_native_registration", lambda conn, store, spec: types.SimpleNamespace(
        run_id="r1", document={"holdouts": {"holdout_as_of_month": "2025-01-01"}, "snapshot": PINNED}))
    monkeypatch.setattr(src, "to_document", lambda snapshot: snapshot)
    monkeypatch.setattr(src, "Repository", lambda conn, store, pin_snapshots: "repo")
    monkeypatch.setattr(src, "_bound_snapshot", lambda conn, snapshot_id: None)
    monkeypatch.setattr(src, "_snapshot", lambda repository, snapshot_id: dict(PINNED))
    monkeypatch.setattr(src, "read_search_plan", lambda conn, store, registration, request, ref: _plan(blends=[{"blend_id": "b1", "name": "size_b1_v1"}]))
    monkeypatch.setattr(src, "read_record", lambda conn, store, run_id, slot: slots.get(slot))
    monkeypatch.setattr(inputs, "population_admit", lambda repository, snapshot, as_of_month, *, max_object_bytes: "admit")
    monkeypatch.setattr(readers, "admit_events", lambda repository, snapshot, as_of_month, *, max_object_bytes: readers.Admitted(
        {"events": pd.DataFrame({"event_id": ["e1"], "event_date": pd.to_datetime(["2025-01-02"])}), "excluded": {}}))
    monkeypatch.setattr(ladder, "default_load_index", lambda repository, snapshot: "index")
    args = types.SimpleNamespace(sources_config=config, expected_identities=EXPECTED)
    return types.SimpleNamespace(conn=conn, slots=slots, args=args)


def test_build_sources_success_leaves_connection_open(env, capsys):
    sources = src.build_sources(env.args)
    assert sources.plan["blends"] == [{"blend_id": "b1", "name": "size_b1_v1"}] and sources.blend_ids == {"M4": "b1"}
    assert sources.expected == EXPECTED and sources.panel_id == "P" and sources.snapshot_id == "snap-1"
    assert sources.selection == SELECTION and sources.load_index == "index" and sources.admit == "admit"
    assert env.conn.closed is False
    assert "[pnl_sources] opened snapshot=snap-1 admitted=1" in capsys.readouterr().err


def test_build_sources_selection_missing_refuses_and_closes(env):
    env.slots.pop("selection")
    with pytest.raises(OpsError) as exc:
        src.build_sources(env.args)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "selection_missing")
    assert env.conn.closed is True


@pytest.mark.parametrize("record", [{"other": []}, {"selections": {"a": 1}}])
def test_build_sources_selection_without_list_refuses_and_closes(env, record):
    env.slots["selection"] = record
    with pytest.raises(OpsError) as exc:
        src.build_sources(env.args)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "selection_missing")
    assert env.conn.closed is True


def test_build_sources_snapshot_too_large_refuses_and_closes(env, monkeypatch):
    calls = []

    def refuse(conn, snapshot_id):
        raise src.fail("EXPERIMENT_IDENTITY_CONFLICT", "snapshot catalog metadata is too large",
                       details={"reason": "snapshot_metadata_too_large"})

    monkeypatch.setattr(src, "_bound_snapshot", refuse)
    monkeypatch.setattr(src, "_snapshot", lambda repository, snapshot_id: calls.append(snapshot_id))
    with pytest.raises(OpsError) as exc:
        src.build_sources(env.args)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_metadata_too_large")
    assert env.conn.closed is True
    assert calls == []


def test_build_sources_configured_snapshot_not_pinned_refuses_and_closes(env, monkeypatch, tmp_path):
    calls, resolved = [], []
    _write(tmp_path / "config.json", {**_cfg(tmp_path), "snapshot_id": "snap-2"})
    monkeypatch.setattr(src, "_bound_snapshot", lambda conn, snapshot_id: calls.append(snapshot_id))
    monkeypatch.setattr(src, "_snapshot", lambda repository, snapshot_id: resolved.append(snapshot_id))
    with pytest.raises(OpsError) as exc:
        src.build_sources(env.args)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_mismatch")
    assert env.conn.closed is True
    assert calls == [] and resolved == []


def test_build_sources_resolved_snapshot_differing_from_pin_refuses_and_closes(env, monkeypatch):
    monkeypatch.setattr(src, "_snapshot", lambda repository, snapshot_id: {"snapshot_id": "snap-1", "extra": True})
    with pytest.raises(OpsError) as exc:
        src.build_sources(env.args)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "snapshot_mismatch")
    assert env.conn.closed is True


def test_build_sources_without_config_refuses(env):
    with pytest.raises(OpsError) as exc:
        src.build_sources(types.SimpleNamespace(expected_identities=EXPECTED))
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "sources_config_invalid")


@pytest.mark.parametrize("expected", [{k: v for k, v in EXPECTED.items() if k != "seal_id"}, "not-a-dict"])
def test_build_sources_expected_invalid_refuses_before_open(env, monkeypatch, expected):
    flags = []
    monkeypatch.setattr(src, "_open", lambda cfg: flags.append(cfg))
    env.args.expected_identities = expected
    with pytest.raises(OpsError) as exc:
        src.build_sources(env.args)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "expected_invalid")
    assert flags == []


def test_read_json_deeply_nested_refuses(tmp_path):
    path = tmp_path / "deep.json"
    path.write_text("[" * 100000 + "]" * 100000)
    with pytest.raises(OpsError) as exc:
        src._read_json(str(path), "input_unreadable")
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "input_unreadable")


def test_build_sources_sqlite_error_refuses_catalog_unavailable_and_closes(env, monkeypatch):
    def boom(conn, store, spec):
        raise sqlite3.OperationalError("boom")

    monkeypatch.setattr(src, "read_native_registration", boom)
    with pytest.raises(OpsError) as exc:
        src.build_sources(env.args)
    assert _refusal(exc) == ("RESOURCE_UNAVAILABLE", "catalog_unavailable")
    assert env.conn.closed is True
