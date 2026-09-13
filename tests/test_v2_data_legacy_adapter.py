"""D02: every source column has a complete, reviewed legacy mapping.

Phase-2 guide §12 (D02): "All six Tier-2 schemas plus panel, Tier-4, and
legacy snapshot metadata have complete reviewed mappings. Removing or adding
one source column fails." Tier 0: seconds, frozen fixtures, no panel load, no
network (``engine/v2/data/README.md`` "Testing").

``engine.v2.data.legacy_adapter`` is the only module this test drives besides
the legacy symbols it wraps; the private-schema test at the bottom is the
sole exception that touches real data, and it is read-only and skipped by
default.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data import legacy_adapter as la  # noqa: E402
from engine.v2.data.legacy_adapter import (  # noqa: E402
    LegacyMappingError,
    build_legacy_mapping,
)
from engine.v2.data.manifests import table_contract_hash  # noqa: E402
from engine.v2.data.documents import decode_document  # noqa: E402
from engine.v2.data.objects import normalize_physical_type  # noqa: E402
from engine.v2.contracts.data import TableContract  # noqa: E402
from engine.v2.foundation import canonical_json  # noqa: E402

from engine.data.schemas import SCHEMAS  # noqa: E402
from engine.data.features.panel import PANEL_COLUMNS  # noqa: E402
from engine.data.features.tier4 import (  # noqa: E402
    COLUMNS as TIER4_COLUMNS,
    KEY_COLUMNS as TIER4_KEY_COLUMNS,
)
from engine import paths as legacy_paths  # noqa: E402

EXPECTED_ORDER = (
    "securities", "earnings_events", "daily_market", "option_chains", "option_daily",
    "trades", "feature_panel", "tier4_forecasts",
)


def _annotations() -> dict:
    """A fresh, independently mutable copy of the reviewed annotations file."""
    return json.loads(la.ANNOTATIONS_PATH.read_text())


def _column(doc: dict, table: str, name: str) -> dict:
    return next(c for c in doc["tables"][table]["columns"] if c["name"] == name)


# --------------------------------------------------------------------------
# structure: all eight datasets, in order, columns exact
# --------------------------------------------------------------------------


def test_all_eight_datasets_present_in_order():
    doc = build_legacy_mapping()
    assert tuple(doc["tables"].keys()) == EXPECTED_ORDER
    assert la.DATASET_ORDER == EXPECTED_ORDER
    assert doc["schema_version"] == "legacy_table_mapping.v1.0"


def test_tier2_columns_match_legacy_source_exactly():
    doc = build_legacy_mapping()
    for name in la.TIER2_DATASETS:
        expected = [c.name for c in SCHEMAS[name].columns]
        got = [c["name"] for c in doc["tables"][name]["columns"]]
        assert got == expected, name


def test_panel_and_tier4_columns_match_legacy_source_exactly():
    doc = build_legacy_mapping()
    assert [c["name"] for c in doc["tables"]["feature_panel"]["columns"]] == list(PANEL_COLUMNS)
    assert [c["name"] for c in doc["tables"]["tier4_forecasts"]["columns"]] == list(TIER4_COLUMNS)
    assert tuple(doc["tables"]["tier4_forecasts"]["primary_key"]) == tuple(TIER4_KEY_COLUMNS)


def test_tier2_physical_types_and_nullability_are_derived_from_schemas():
    doc = build_legacy_mapping()
    for name in la.TIER2_DATASETS:
        schema = SCHEMAS[name]
        for col in schema.columns:
            contract_col = _column(doc, name, col.name)
            assert contract_col["physical_type"] == la.LEGACY_DTYPE_MAP[col.dtype], (name, col.name)
            assert contract_col["nullable"] == col.nullable, (name, col.name)


def test_tier2_partition_columns_follow_the_declared_legacy_partition():
    doc = build_legacy_mapping()
    for name in la.TIER2_DATASETS:
        schema = SCHEMAS[name]
        expected = (schema.partition_by,) if schema.partition_by else ()
        assert tuple(doc["tables"][name]["partition_columns"]) == expected, name


def test_panel_and_tier4_declare_one_logical_partition():
    doc = build_legacy_mapping()
    assert doc["tables"]["feature_panel"]["partition_columns"] == []
    assert doc["tables"]["tier4_forecasts"]["partition_columns"] == []


def test_snapshot_metadata_present_not_queryable_and_not_a_table():
    doc = build_legacy_mapping()
    meta = doc["legacy_snapshot_metadata"]
    assert meta["queryable"] is False
    assert "SNAPSHOT" not in doc["tables"]
    assert set(doc["tables"]) == set(EXPECTED_ORDER)
    assert meta["legacy_path"] == "features/SNAPSHOT"
    assert set(meta["expected_top_level_keys"]) == {
        "snapshot", "generated_at", "format", "tables", "panel_sha256", "tier4_sha256",
    }


def test_hardcoded_relative_paths_match_engine_paths():
    """The three path constants la.py carries instead of a 4th legacy import
    (judgement call 4 of the task report) must not silently drift from
    ``engine/paths.py``. The private-schema test only notices a *missing*
    file under an opted-in ``PHASE2_PRIVATE_ROOT`` and is skipped by
    default, so this tier-0 check is the one that always runs.
    """
    assert la.PANEL_RELATIVE_PATH == (
        legacy_paths.PANEL.relative_to(legacy_paths.DATA).as_posix()
    )
    assert la.TIER4_RELATIVE_PATH == (
        legacy_paths.TIER4.relative_to(legacy_paths.DATA).as_posix()
    )
    assert la.SNAPSHOT_RELATIVE_PATH == (
        legacy_paths.SNAPSHOT_FILE.relative_to(legacy_paths.DATA).as_posix()
    )


def test_knowledge_mode_is_reconstructed_for_all_eight():
    doc = build_legacy_mapping()
    assert doc["knowledge_mode_by_table"] == {name: "reconstructed" for name in EXPECTED_ORDER}


def test_every_build_output_passes_decode_document():
    doc = build_legacy_mapping()
    for name in EXPECTED_ORDER:
        decode_document(TableContract, doc["tables"][name])


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------


def test_extra_source_column_fails_naming_dataset_and_column(monkeypatch):
    schema = SCHEMAS["securities"]
    extra = dataclasses.replace(schema.columns[0], name="totally_new_column")
    patched = dataclasses.replace(schema, columns=schema.columns + (extra,))
    patched_schemas = dict(SCHEMAS)
    patched_schemas["securities"] = patched
    monkeypatch.setattr(la, "SCHEMAS", patched_schemas)

    with pytest.raises(LegacyMappingError) as exc:
        build_legacy_mapping()
    assert exc.value.code == "MISSING_ANNOTATION"
    assert exc.value.dataset == "securities"
    assert "totally_new_column" in str(exc.value)


def test_removed_annotation_fails_naming_dataset_and_column():
    ann = _annotations()
    del ann["tables"]["securities"]["columns"]["ticker"]

    with pytest.raises(LegacyMappingError) as exc:
        build_legacy_mapping(ann)
    assert exc.value.code == "MISSING_ANNOTATION"
    assert exc.value.dataset == "securities"
    assert "ticker" in str(exc.value)


def test_annotation_for_nonexistent_column_fails():
    ann = _annotations()
    ann["tables"]["securities"]["columns"]["not_a_real_column"] = copy.deepcopy(
        ann["tables"]["securities"]["columns"]["ticker"]
    )

    with pytest.raises(LegacyMappingError) as exc:
        build_legacy_mapping(ann)
    assert exc.value.code == "UNKNOWN_ANNOTATED_COLUMN"
    assert exc.value.dataset == "securities"
    assert "not_a_real_column" in str(exc.value)


def test_unmapped_tier2_dtype_fails(monkeypatch):
    schema = SCHEMAS["securities"]
    bad = dataclasses.replace(schema.columns[0], dtype="decimal128")
    patched = dataclasses.replace(schema, columns=(bad,) + schema.columns[1:])
    patched_schemas = dict(SCHEMAS)
    patched_schemas["securities"] = patched
    monkeypatch.setattr(la, "SCHEMAS", patched_schemas)

    with pytest.raises(LegacyMappingError) as exc:
        build_legacy_mapping()
    assert exc.value.code == "UNMAPPED_DTYPE"
    assert exc.value.dataset == "securities"


def test_unmapped_panel_physical_type_fails():
    ann = _annotations()
    ann["tables"]["feature_panel"]["columns"]["ticker"]["physical_type"] = "decimal128"

    with pytest.raises(LegacyMappingError) as exc:
        build_legacy_mapping(ann)
    assert exc.value.code == "UNMAPPED_DTYPE"
    assert exc.value.dataset == "feature_panel"


def test_undeclared_filterable_column_fails():
    ann = _annotations()
    ann["tables"]["securities"]["filterable_columns"].append("not_a_column")

    with pytest.raises(LegacyMappingError) as exc:
        build_legacy_mapping(ann)
    assert exc.value.code == "UNDECLARED_COLUMN"
    assert exc.value.dataset == "securities"


# --------------------------------------------------------------------------
# definition_hash: recomputes, sensitive to order and content, deterministic
# --------------------------------------------------------------------------


def test_definition_hash_recomputes_from_the_contract():
    doc = build_legacy_mapping()
    for name in EXPECTED_ORDER:
        contract = decode_document(TableContract, doc["tables"][name])
        assert table_contract_hash(contract) == contract.definition_hash, name


def test_reordering_two_columns_changes_definition_hash(monkeypatch):
    original = build_legacy_mapping()["tables"]["securities"]["definition_hash"]

    schema = SCHEMAS["securities"]
    cols = list(schema.columns)
    cols[0], cols[1] = cols[1], cols[0]
    patched = dataclasses.replace(schema, columns=tuple(cols))
    patched_schemas = dict(SCHEMAS)
    patched_schemas["securities"] = patched
    monkeypatch.setattr(la, "SCHEMAS", patched_schemas)

    reordered = build_legacy_mapping()["tables"]["securities"]["definition_hash"]
    assert reordered != original


def test_changing_one_column_unit_changes_definition_hash():
    original = build_legacy_mapping()["tables"]["securities"]["definition_hash"]

    ann = _annotations()
    ann["tables"]["securities"]["columns"]["mcap_usd"]["unit"] = "eur"
    changed = build_legacy_mapping(ann)["tables"]["securities"]["definition_hash"]

    assert changed != original


def test_building_twice_is_byte_identical_canonical_json():
    doc1 = build_legacy_mapping()
    doc2 = build_legacy_mapping()
    assert canonical_json(doc1) == canonical_json(doc2)


# --------------------------------------------------------------------------
# sentinel policy: every column whose legacy doc mentions a sentinel is tagged
# --------------------------------------------------------------------------


def test_implied_move_zero_sentinel_is_documented():
    doc = build_legacy_mapping()
    ann = _column(doc, "daily_market", "implied_move")
    assert ann["sentinel_policy"]
    assert "zero" in ann["sentinel_policy"].lower()
    assert "flt_max" in ann["sentinel_policy"].lower()


#: Every ORATS-vended numeric field this review found documented against
#: CONVENTIONS["orats_flt_max_sentinel"] ("ORATS uses FLT_MAX ... in numeric
#: fields"). Recorded explicitly here (rather than derived structurally) is
#: this review's own judgement call — see the task report.
FLT_MAX_COLUMNS = (
    ("daily_market", "spot"), ("daily_market", "iv10"), ("daily_market", "iv30"),
    ("daily_market", "exern_iv10"), ("daily_market", "exern_iv30"), ("daily_market", "rvol30"),
    ("daily_market", "skew"), ("daily_market", "contango"), ("daily_market", "fwd90_30"),
    ("daily_market", "fexern90_30"), ("daily_market", "iee"),
    ("option_chains", "bid"), ("option_chains", "ask"), ("option_chains", "mid"),
    ("option_chains", "iv"), ("option_chains", "delta"), ("option_chains", "spot"),
)


def test_every_orats_numeric_field_documents_the_flt_max_sentinel():
    doc = build_legacy_mapping()
    for table, name in FLT_MAX_COLUMNS:
        ann = _column(doc, table, name)
        assert ann["sentinel_policy"], f"{table}.{name} should document the FLT_MAX sentinel"
        assert "flt_max" in ann["sentinel_policy"].lower()


# --------------------------------------------------------------------------
# private-schema check — read-only Parquet *schema* only, never rows
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("PHASE2_PRIVATE_ROOT"),
    reason="PHASE2_PRIVATE_ROOT not set; private-schema check is opt-in",
)
def test_private_parquet_schema_matches_contract():
    """Physical types in real curated Parquet match this build's contracts.

    Reads ONLY ``pyarrow.parquet.read_schema`` — never a row — under
    ``$PHASE2_PRIVATE_ROOT``. Uses ``engine.v2.data.objects.normalize_physical_type``,
    the same normalization ``inspect_fragment`` applies, so ``large_string``/``double``
    (pyarrow's own spellings for ``string``/``float64``) are treated as equal; any
    timestamp *unit* mismatch (``ns`` vs ``us``) is reported explicitly.
    """
    import pyarrow.parquet as pq

    root = Path(os.environ["PHASE2_PRIVATE_ROOT"])
    doc = build_legacy_mapping()
    mismatches: list[str] = []

    for name in la.TIER2_DATASETS:
        table_dir = root / "curated" / name
        parts = sorted(table_dir.glob("year=*/*.parquet"))
        if not parts:
            mismatches.append(f"{name}: no Parquet files found under {table_dir}")
            continue
        schema = pq.read_schema(parts[0])
        by_name = {f.name: str(f.type) for f in schema}
        for col in doc["tables"][name]["columns"]:
            mismatches.extend(_check_physical(name, col, by_name))

    for table, rel in (("feature_panel", "features/panel.parquet"),
                        ("tier4_forecasts", "features/tier4_forecasts.parquet")):
        path = root / rel
        if not path.exists():
            mismatches.append(f"{table}: no Parquet file at {path}")
            continue
        schema = pq.read_schema(path)
        by_name = {f.name: str(f.type) for f in schema}
        for col in doc["tables"][table]["columns"]:
            mismatches.extend(_check_physical(table, col, by_name))

    assert not mismatches, "private-schema mismatches:\n" + "\n".join(mismatches)


def _check_physical(table: str, col: dict, by_name: dict[str, str]) -> list[str]:
    name = col["name"]
    if name not in by_name:
        return [f"{table}.{name}: not present in the private Parquet schema"]
    actual = by_name[name]
    normalized = normalize_physical_type(actual)
    expected = col["physical_type"]
    if normalized == expected:
        return []
    if actual.startswith("timestamp[") and expected.startswith("timestamp["):
        return [f"{table}.{name}: timestamp unit mismatch — contract {expected!r}, Parquet {actual!r}"]
    return [f"{table}.{name}: contract {expected!r}, Parquet {actual!r}"]
