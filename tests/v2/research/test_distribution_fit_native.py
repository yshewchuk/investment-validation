"""C consumes B's real synthetic native admission, not another source policy."""
from copy import deepcopy

import pytest

from engine.v2.foundation import SystemClock, content_hash
from engine.v2.ops.errors import OpsError
from experiments import distribution_plan, distribution_trials
from experiments.distribution_fitted import DistributionFits
from tests.v2.research.test_distribution_native_admission import (
    case as source_case,
    own as source_own,
    trials,
)

pytestmark = pytest.mark.xdist_group("distribution_fit_native")


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    yield from source_case.__wrapped__(tmp_path_factory)


@pytest.fixture
def own(case, request):
    return source_own.__wrapped__(case, request)


def test_fit_context_is_exact_detached_projection_without_fresh_outcomes(case, own, monkeypatch):
    adapter = trials(case, own)
    plan = adapter.seal(clock=SystemClock())
    receipt = adapter.start("inner", "2024", "2023")

    def forbidden(*args):
        pytest.fail("stored fit context must not reopen native outcomes")

    monkeypatch.setattr(distribution_trials, "fold_inputs", forbidden)
    context = adapter.read_fit_context("inner", "2024", "2023")
    assert context["plan"] == plan
    assert context["admission"] == receipt
    assert context["fold"] == case.plan["folds"][0]["inner"][0]
    assert context["inputs"]["training"]
    original = deepcopy(context)
    context["inputs"]["training"].clear()
    context["plan"]["recipe"]["horizons"].clear()
    assert adapter.read_fit_context("inner", "2024", "2023") == original
    assert adapter.read_admission("inner", "2024", "2023") == receipt


@pytest.mark.parametrize("horizon", [5, 3, 1])
def test_real_native_degenerate_training_claims_once_and_never_fakes_completion(case, own, horizon):
    adapter = trials(case, own)
    adapter.seal(clock=SystemClock())
    admitted = adapter.start("inner", "2024", "2023")
    # This deliberately tiny native panel has one admitted training event.
    assert len(adapter.read_fit_context("inner", "2024", "2023")["inputs"]["training"]) == 1
    fitter = DistributionFits(adapter)
    with pytest.raises(OpsError) as failed:
        fitter.fit("inner", "2024", horizon, "2023")
    assert failed.value.code == "EXPERIMENT_VARIANT_FAILED"
    for action in (fitter.fit, fitter.read):
        with pytest.raises(OpsError) as replay:
            action("inner", "2024", horizon, "2023")
        assert replay.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert adapter.read_admission("inner", "2024", "2023") == admitted


def test_new_c_implementation_is_bound_and_changed_closure_refuses_old_seal(case, own, monkeypatch):
    adapter = trials(case, own)
    plan = adapter.seal(clock=SystemClock())
    sources = plan["implementation"]["source_closure"]
    assert {"experiments/distribution_fitted.py", "experiments/distribution_oof.py"} <= set(sources)
    adapter.start("inner", "2024", "2023")
    original = distribution_plan.source_closure

    def changed(*args):
        return {**original(*args), "experiments/distribution_oof.py": content_hash("changed C")}

    monkeypatch.setattr(distribution_plan, "source_closure", changed)
    with pytest.raises(OpsError) as caught:
        adapter.read_fit_context("inner", "2024", "2023")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
