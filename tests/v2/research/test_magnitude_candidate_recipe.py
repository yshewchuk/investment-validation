"""Closed family dispatch and source-free recipe assembly, not admission."""
import subprocess
import sys

import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_candidate import magnitude_candidate_family
from engine.v2.models.magnitude_elasticnet_recipe import elasticnet_recipe
from engine.v2.models.magnitude_ridge_hgb_recipe import ridge_hgb_recipe
from experiments import magnitude_candidate_recipe as assembly

HASH = "sha256:" + "a" * 64
CASES = [("ridge", {"alpha": a}) for a in (0.1, 1, 10)] + [
    ("elasticnet", {"alpha": a, "l1_ratio": r}) for a in (0.01, 0.1, 1)
    for r in (0.25, 0.75)] + [("hgb", {"max_leaf_nodes": n}) for n in (7, 15)]
HGB_V2 = {"learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 20,
          "l2_regularization": 0.0}


def test_size_policy_is_not_advertised_by_generic_recipe_or_numeric_dispatch():
    with pytest.raises(ValueError, match="^candidate requires a supported family and exact parameter keys$"):
        assembly.candidate_recipe("size_v1_4", {}, columns=object(), feature_dictionary_hash=object())
    for schema in ("size_causal_comparator.v1", "size_history_mean_candidate.v1"):
        with pytest.raises(ValueError, match="^candidate requires a supported recipe schema/model pair$"):
            magnitude_candidate_family({"schema_version": schema})


@pytest.mark.parametrize("family,parameters", CASES)
@pytest.mark.parametrize("width", [1, 3, 11, 28, 42, 43])
def test_all_choices_reuse_complete_family_recipe_and_detach(family, parameters, width):
    columns = [f"x{i}" for i in range(width)]
    before_parameters = parameters.copy()
    kwargs = {"columns": columns, "feature_dictionary_hash": HASH, **parameters}
    expected = (elasticnet_recipe(**kwargs) if family == "elasticnet" else
                ridge_hgb_recipe(model="history_ridge" if family == "ridge" else "history_hgb", **kwargs))
    result = assembly.candidate_recipe(family, parameters, columns=columns, feature_dictionary_hash=HASH)
    assert result == expected
    columns.clear()
    assert result == expected
    result["estimator"]["params"].clear()
    assert parameters == before_parameters


def test_hgb_four_axis_parameters_return_v2_recipe():
    result = assembly.candidate_recipe("hgb", HGB_V2, columns=["x"], feature_dictionary_hash=HASH)
    expected = ridge_hgb_recipe(model="history_hgb", columns=["x"],
                                feature_dictionary_hash=HASH, **HGB_V2)
    assert result == expected
    assert result["schema_version"] == "magnitude_ridge_hgb_recipe.v2"
    assert result["estimator"]["params"]["early_stopping"] is True


def test_hgb_legacy_leaf_only_parameters_still_return_v1_recipe():
    result = assembly.candidate_recipe("hgb", {"max_leaf_nodes": 7}, columns=["x"],
                                       feature_dictionary_hash=HASH)
    assert result["schema_version"] == "magnitude_ridge_hgb_recipe.v1"
    assert result["estimator"]["params"]["early_stopping"] is False


@pytest.mark.parametrize("parameters", [
    {"max_leaf_nodes": 7, "learning_rate": 0.05},
    {"max_leaf_nodes": 7, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {**HGB_V2, "max_depth": 2},
    {"max_leaf_nodes": 7, "extra": 1},
])
def test_hgb_mixed_key_sets_refuse(parameters):
    with pytest.raises(ValueError, match="^candidate requires a supported family and exact parameter keys$"):
        assembly.candidate_recipe("hgb", parameters, columns=["x"], feature_dictionary_hash=HASH)


def test_order_dictionary_and_every_parameter_choice_bind_identity():
    hashes = {content_hash(assembly.candidate_recipe(f, p, columns=["first", "last"],
              feature_dictionary_hash=HASH)) for f, p in CASES}
    assert len(hashes) == 11
    for columns, digest in ((["last", "first"], HASH), (["first"], HASH),
                            (["first", "last"], "sha256:" + "b" * 64)):
        changed = assembly.candidate_recipe("ridge", {"alpha": 1}, columns=columns,
                                            feature_dictionary_hash=digest)
        assert content_hash(changed) not in hashes


@pytest.mark.parametrize("family,parameters", [
    (None, {}), ([], {}), ("unknown", {}), ("history_ridge", {"alpha": 1}),
    ("ridge", None), ("ridge", []), ("ridge", {}), ("ridge", {"alpha": 1, "other": 2}),
    ("elasticnet", {"alpha": 1}), ("hgb", {"alpha": 1}),
])
def test_invalid_dispatch_envelope_refuses_before_constructor(family, parameters, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("constructor reached for invalid dispatch envelope")
    monkeypatch.setattr(assembly, "elasticnet_recipe", forbidden)
    monkeypatch.setattr(assembly, "ridge_hgb_recipe", forbidden)
    with pytest.raises(ValueError, match="^candidate requires a supported family and exact parameter keys$"):
        assembly.candidate_recipe(family, parameters, columns=object(), feature_dictionary_hash=object())


@pytest.mark.parametrize("family,parameters,message", [
    ("ridge", {"alpha": True}, "Ridge requires declared alpha and no HGB parameter"),
    ("ridge", {"alpha": 10000.0001}, "Ridge requires declared alpha and no HGB parameter"),
    ("elasticnet", {"alpha": 1, "l1_ratio": 1.0001},
     "ElasticNet parameters must be finite numeric values within declared bounds"),
    ("hgb", {"max_leaf_nodes": 7.0}, "HGB requires declared integer leaf count and no Ridge parameter"),
])
def test_family_owns_parameter_value_refusal(family, parameters, message):
    with pytest.raises(ValueError, match="^" + message + "$"):
        assembly.candidate_recipe(family, parameters, columns=["x"], feature_dictionary_hash=HASH)


def test_ridge_alpha_at_ceiling_is_permitted():
    result = assembly.candidate_recipe("ridge", {"alpha": 10000.0}, columns=["x"],
                                       feature_dictionary_hash=HASH)
    assert result["estimator"]["params"]["alpha"] == 10000.0


@pytest.mark.parametrize("family,parameters", [
    ("ridge", {"alpha": 100.0}),
    ("elasticnet", {"alpha": 0.003, "l1_ratio": 0.25}),
    ("elasticnet", {"alpha": 0.03, "l1_ratio": 0.75}),
])
def test_v3_request_grid_values_reach_the_closed_family_recipe(family, parameters):
    result = assembly.candidate_recipe(family, parameters, columns=["x"], feature_dictionary_hash=HASH)
    expected = (elasticnet_recipe(columns=["x"], feature_dictionary_hash=HASH, **parameters)
                if family == "elasticnet" else
                ridge_hgb_recipe(model="history_ridge", columns=["x"],
                                 feature_dictionary_hash=HASH, **parameters))
    assert result == expected


def test_import_and_invalid_fit_dispatch_never_load_training_or_sources():
    probe = '''
import importlib.abc
import sys
class RejectSources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(("engine.v2.models.training", "sklearn",
            "engine.v2.research", "engine.v2.ops", "engine.v2.data", "engine.data")), fullname
sys.meta_path.insert(0, RejectSources())
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_candidate_fit import fit_candidate
from engine.v2.models.no_fit import no_fit_guard, RuntimeFitForbidden
for family, knobs in (("ridge", {"alpha": 10}), ("hgb", {"max_leaf_nodes": 15}),
                      ("elasticnet", {"alpha": 1, "l1_ratio": 0.75})):
    candidate_recipe(family, knobs, columns=["arbitrary"], feature_dictionary_hash="sha256:" + "a"*64)
try:
    fit_candidate({}, object(), object(), object(), columns=object())
except ValueError:
    pass
else:
    raise AssertionError("unsupported envelope accepted")
with no_fit_guard():
    try:
        fit_candidate(object(), object(), object(), object(), columns=object())
    except RuntimeFitForbidden:
        pass
    else:
        raise AssertionError("guard ignored")
'''
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("family,parameters", [("ridge", {"alpha": 1}),
    ("elasticnet", {"alpha": 0.1, "l1_ratio": 0.75}), ("hgb", {"max_leaf_nodes": 7})])
def test_successful_dispatch_imports_only_selected_family(family, parameters):
    probe = '''
import importlib.abc
import sys
family, parameters = FAMILY, PARAMETERS
chosen = "magnitude_elasticnet" if family == "elasticnet" else "magnitude_ridge_hgb"
other = "magnitude_ridge_hgb" if family == "elasticnet" else "magnitude_elasticnet"
prefix = "engine.v2.models.training."
class RejectOtherFamily(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname != prefix + other, fullname
sys.meta_path.insert(0, RejectOtherFamily())
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_candidate_fit import fit_candidate
assert "engine.v2.models.training" not in sys.modules
policy = candidate_recipe(family, parameters, columns=["x"], feature_dictionary_hash="sha256:" + "a"*64)
result = fit_candidate(policy, [[0], [1], [2]], [2, 3, 4], [[1]], columns=["x"])
assert result["predictions"] == [3.0]
assert prefix + chosen in sys.modules
assert prefix + other not in sys.modules
'''.replace("FAMILY", repr(family)).replace("PARAMETERS", repr(parameters))
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
