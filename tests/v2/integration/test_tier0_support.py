"""Tier-0 support preserves behavior without pulling in its process runner.

# packages: engine.v2.diagnosis, engine.v2.data, engine.v2.contracts, engine.v2.foundation
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from checks import rearchitecture_phase2_evidence as evidence
from checks import tier0_corpus as corpus
from checks import tier0_support as support
from tools import mutation_pilot as selector

ROOT = Path(__file__).resolve().parents[3]


def test_compatibility_exports_are_the_same_objects():
    assert corpus.DEFAULT_CORPUS is support.DEFAULT_CORPUS
    assert evidence.DEFAULT_CORPUS is support.DEFAULT_CORPUS
    assert corpus.round_params is support.round_params
    assert corpus.finding_dicts is support.finding_dicts


def test_default_corpus_retains_repository_root_independently_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert support.DEFAULT_CORPUS == ROOT / "fixtures" / "tier0"
    assert support.DEFAULT_CORPUS == corpus.ROOT / "fixtures" / "tier0"


def test_round_params_preserves_bytes_order_and_copy_boundaries():
    nested = {"x": [1.123456789]}
    record = {"structure_params": {
        "float": 1.123456789, "int": 2, "bool": True,
        "nested": nested, "negative_zero": -0.0,
    }, "outside": nested, "unrounded": 1.123456789}
    original = json.dumps(record)
    result = support.round_params(record)
    assert json.dumps(result) == (
        '{"structure_params": {"float": 1.123457, "int": 2, "bool": true, '
        '"nested": {"x": [1.123456789]}, "negative_zero": -0.0}, '
        '"outside": {"x": [1.123456789]}, "unrounded": 1.123456789}'
    )
    assert json.dumps(record) == original
    assert result is not record
    assert result["structure_params"] is not record["structure_params"]
    assert result["outside"] is result["structure_params"]["nested"]
    result["outside"]["x"].append(3)
    assert nested == {"x": [1.123456789]}


@pytest.mark.parametrize("params", [None, [], [1.123456789], "raw", 7])
def test_non_dictionary_params_are_copied_without_rounding(params):
    result = support.round_params({"structure_params": params})
    assert result == {"structure_params": params}
    if isinstance(params, list):
        assert result["structure_params"] is not params


def test_missing_params_and_nonfinite_floats_keep_existing_behavior():
    assert support.round_params({}) == {}
    values = {"nan": float("nan"), "pos": float("inf"), "neg": -float("inf")}
    result = support.round_params({"structure_params": values})["structure_params"]
    assert math.isnan(result["nan"])
    assert result["pos"] == values["pos"] and result["neg"] == values["neg"]


@pytest.mark.parametrize("record", [None, [], 1])
def test_invalid_records_still_raise_attribute_error(record):
    with pytest.raises(AttributeError):
        support.round_params(record)


def test_deepcopy_failure_is_not_swallowed():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy")

    with pytest.raises(RuntimeError, match="cannot copy"):
        support.round_params({"nested": Uncopyable()})


def test_finding_projection_preserves_order_duplicates_and_value_identity():
    value = ["retained"]
    first = SimpleNamespace(first_differing_stage="forecast", field_path="x", kind=value)
    second = SimpleNamespace(first_differing_stage="analogs", field_path="y", kind="changed")
    receipt = SimpleNamespace(findings=[second, first, first])
    result = support.finding_dicts(receipt)
    assert result == [
        {"first_differing_stage": "analogs", "field_path": "y", "kind": "changed"},
        {"first_differing_stage": "forecast", "field_path": "x", "kind": value},
        {"first_differing_stage": "forecast", "field_path": "x", "kind": value},
    ]
    assert list(result[0]) == ["first_differing_stage", "field_path", "kind"]
    assert result[1] is not result[2]
    assert result[1]["kind"] is value
    result[1]["field_path"] = "changed"
    assert first.field_path == "x" and result[2]["field_path"] == "x"
    assert support.finding_dicts(SimpleNamespace(findings=[])) == []


@pytest.mark.parametrize("receipt,error", [
    (None, AttributeError),
    (SimpleNamespace(findings=None), TypeError),
    (SimpleNamespace(findings=[SimpleNamespace(first_differing_stage="x")]), AttributeError),
])
def test_invalid_receipts_keep_their_errors(receipt, error):
    with pytest.raises(error):
        support.finding_dicts(receipt)


def test_pure_consumers_do_not_inherit_corpus_process_taint():
    # Mutation work copies contain these sources but have no Git metadata.
    tracked = sorted(str(path.relative_to(ROOT))
                     for source in ("checks", "engine", "tests", "tools")
                     for path in (ROOT / source).rglob("*.py"))
    graph = selector.build_import_graph(tracked)
    unresolved = selector.unresolved_import_files(tracked)
    runner = "checks/tier0_corpus.py"
    assert runner in unresolved  # The genuine process runner still fails safe.
    for path in (
        "checks/tier0_support.py",
        "checks/rearchitecture_phase2_evidence.py",
        "tests/v2/diagnosis/test_phase0_negative_controls.py",
    ):
        roots = {path} | selector._conftest_ancestors(path, set(tracked))
        closure, tainted = selector._closure_from_roots(
            roots, graph, unresolved, taint_exempt=set())
        assert runner not in closure, path
        assert not tainted, path
