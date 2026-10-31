"""Public regression preparation admits caller shapes before dependent metadata."""
from dataclasses import replace
from pathlib import Path

import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.ops.errors import OpsError, fail
from experiments import iv_crush_search
from experiments import native_registration as native
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_iv_crush_registration import prepared  # noqa: F401
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_native_registration import artifact_object_bytes


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("missing", ["spec", "folds", "feature_snapshot", "target_snapshot"])
def test_malformed_public_prepare_preserves_native_problem_and_no_mutation(prepared, monkeypatch, generic, missing):  # noqa: F811
    data = prepared
    spec, options = data.spec, dict(data.options)
    if missing == "spec":
        spec = None
        expected = fail("INVALID_EXPERIMENT_SPEC", "regression registration requires an experiment specification")
    elif missing == "folds":
        spec = replace(spec, folds=None)
        expected = fail("INVALID_EXPERIMENT_SPEC", "experiment plan field must be a tuple of strings",
            details={"field": "folds", "type": "NoneType"})
    else:
        options[missing] = None
        expected = data_fail("SNAPSHOT_UNRESOLVED", "an exact native snapshot reference is required")
    before, saved = catalog_rows(data.conn), artifact_object_bytes(data.store)
    resolved = []
    original = data.repository.resolve
    def resolve(snapshot_id):
        resolved.append(snapshot_id)
        return original(snapshot_id)
    def forbidden(*a, **k):
        pytest.fail("malformed preparation reached dependent table/payload reads or publication")
    monkeypatch.setattr(data.repository, "resolve", resolve)
    for name in ("table_contract", "scan", "resolve_pinned"):
        monkeypatch.setattr(data.repository, name, forbidden)
    monkeypatch.setattr(data.store, "publish_bytes", forbidden)
    with pytest.raises((OpsError, DataError)) as caught:
        if generic:
            native.prepare_regression_registration(data.repository, spec,
                code_root=Path(native.__file__).resolve().parents[1],
                target=data.registration.document["regression_plan"]["target_recipe"], **options)
        else:
            iv_crush_search.prepare(data.repository, spec, **options)
    assert type(caught.value) is type(expected) and caught.value.problem == expected.problem
    assert caught.value.problem.retryable is False
    # Generic admission retains its ordered source/feature/target identity checks;
    # the IV wrapper checks both reference shapes before either table-contract read.
    expected_resolves = []
    if generic and missing in ("feature_snapshot", "target_snapshot"):
        expected_resolves = [data.options["source"].snapshot_id]
        if missing == "target_snapshot":
            expected_resolves.append(data.options["feature_snapshot"].snapshot_id)
    assert resolved == expected_resolves
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == saved
