"""Metadata-only IV plans cannot reach execution effects before native activation."""
import json
from dataclasses import asdict

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import SystemClock, canonical_json, from_document, to_document
from engine.v2.ops.errors import OpsError
from engine.v2.ops.migrations import migrate
from experiments import magnitude_search_outer as outer
from experiments import magnitude_search_plan as plans
from experiments import magnitude_search_trials as trials
from experiments import native_registration as native
from experiments import v2_magnitude_search_run as driver
from experiments.magnitude_search_records import MIGRATIONS, OWNER
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_iv_crush_registration import prepared  # noqa: F401
from tests.v2.research.test_iv_crush_search_plan import request
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_native_registration import artifact_object_bytes


@pytest.mark.parametrize("entry", ["trials", "driver", "outer", "run", "collect", "cli-run", "cli-collect"])
def test_iv_metadata_plan_refuses_execution_before_effects(prepared, monkeypatch, tmp_path, capsys, entry):  # noqa: F811
    data = prepared
    registered = native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
    document = request(data)
    sealed = plans.seal_search_plan(data.conn, data.store, registered, document)
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    assert plans.read_search_plan(data.conn, data.store, registered, document, ref) == sealed["plan"]
    migrate(data.conn, OWNER, MIGRATIONS, clock=SystemClock())
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    def forbidden(*a, **k):
        pytest.fail("metadata-only IV reached an execution effect")
    for module in (trials, outer, driver):
        monkeypatch.setattr(module, "append_record", forbidden)
    for module in (trials, outer):
        monkeypatch.setattr(module, "admit_fold_inputs", forbidden)
        monkeypatch.setattr(module, "run_fitted_trial", forbidden)
        monkeypatch.setattr(module, "load_magnitude_targets", forbidden)
    args = (data.conn, data.store, data.repository, registered, document, ref)
    if entry.startswith("cli-"):
        argv = [entry.removeprefix("cli-"), "--catalog", data.conn.execute("PRAGMA database_list").fetchone()[2],
                "--store-root", str(data.store.root)]
        for name, value in (("spec", asdict(data.spec)), ("request", document), ("plan-ref", sealed["plan_ref"])):
            path = tmp_path / (name + ".json")
            path.write_text(canonical_json(value), encoding="utf-8")
            argv += ["--" + name, str(path)]
        if entry == "cli-run":
            for name, value in document["limits"]["child"].items():
                argv += ["--" + name.replace("_", "-"), str(value)]
        assert driver.main(argv) == 2
        problem = json.loads(capsys.readouterr().out)["refused"]
    else:
        with pytest.raises(OpsError) as caught:
            if entry == "trials":
                trials.SearchTrials(*args)
            elif entry == "driver":
                driver.seal_search_run(*args, clock=SystemClock())
            elif entry == "run":
                driver.run_search(*args, limits=document["limits"]["child"])
            elif entry == "collect":
                outer.collect_outer_results(*args)
            else:
                outer.execute_outer_fold(*args, "2024", limits=document["limits"]["child"])
        problem = to_document(caught.value.problem)
    assert problem["code"] == "INVALID_EXPERIMENT_SPEC"
    assert problem["message"] == "IV trial and outer orchestration is not enabled"
    assert problem["retryable"] is False
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects
