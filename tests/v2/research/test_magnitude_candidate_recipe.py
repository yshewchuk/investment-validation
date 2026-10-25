"""Closed family dispatch and source-free recipe assembly, not admission."""
import subprocess
import sys

import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_elasticnet_recipe import elasticnet_recipe
from engine.v2.models.magnitude_ridge_hgb_recipe import ridge_hgb_recipe
from experiments import magnitude_candidate_recipe as assembly

HASH = "sha256:" + "a" * 64
CASES = [("ridge", {"alpha": a}) for a in (0.1, 1, 10)] + [
    ("elasticnet", {"alpha": a, "l1_ratio": r}) for a in (0.01, 0.1, 1)
    for r in (0.25, 0.75)] + [("hgb", {"max_leaf_nodes": n}) for n in (7, 15)]


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
    ("elasticnet", {"alpha": 1, "l1_ratio": 0.5}, "ElasticNet parameters must belong to the declared six combinations"),
    ("hgb", {"max_leaf_nodes": 7.0}, "HGB requires declared integer leaf count and no Ridge parameter"),
])
def test_family_owns_parameter_value_refusal(family, parameters, message):
    with pytest.raises(ValueError, match="^" + message + "$"):
        assembly.candidate_recipe(family, parameters, columns=["x"], feature_dictionary_hash=HASH)


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
