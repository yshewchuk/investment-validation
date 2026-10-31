"""Additive dictionary dispatch for source-free stage-2 declaration previews."""
import json

from engine.v2.foundation import content_hash
from engine.v2.research.magnitude_panel_qualified_table import recipe as v3_recipe
from engine.v2.research.magnitude_panel_table import recipe as v1_recipe
from engine.v2.research.magnitude_panel_v2_table import recipe as v2_recipe
from experiments.magnitude_search_request import compile_request as compile_v1_request
from experiments.magnitude_search_request_leads import compile_request as compile_v2_leads_request
from experiments.magnitude_search_request_v3 import compile_request as compile_v3_request

__all__ = ["compile_request"]

_V2_HASH = v2_recipe("HMO")["dictionary_hash"]
_V3_HASH = v3_recipe("HMO")["dictionary_hash"]


def compile_request(request, *, horizons=(5, 3, 1)):
    """Dispatch bounded v3 grids while preserving existing dictionary replay."""
    version = request.get("schema_version") if type(request) is dict else None
    if version == "regression_search_request.v1":
        return compile_v1_request(request)
    compiler = (compile_v3_request if version == "magnitude_search_request.v3"
                else compile_v2_leads_request if version == "magnitude_search_request.v2"
                else compile_v1_request)
    compile_kwargs = {"horizons": horizons} if version in ("magnitude_search_request.v2", "magnitude_search_request.v3") else {}
    dictionary = request.get("feature_dictionary_hash") if type(request) is dict else None
    if type(dictionary) is not str or dictionary not in (_V2_HASH, _V3_HASH):
        return compiler(request, **compile_kwargs)

    selected = v3_recipe if dictionary == _V3_HASH else v2_recipe
    surrogate = dict(request)
    surrogate["feature_dictionary_hash"] = v1_recipe("HMO")["dictionary_hash"]
    result = compiler(surrogate, **compile_kwargs)
    result["request"]["feature_dictionary_hash"] = dictionary
    result["request_id"] = content_hash(result["request"])
    recipes = {subset: selected(subset) for subset in result["request"]["feature_sets"]}
    for spec in result["candidate_specs"]:
        spec["feature_recipe"] = recipes[spec["feature_recipe"]["subset"]]
        spec["candidate_spec_id"] = content_hash({
            "request_id": result["request_id"], "family": spec["family"],
            "feature_recipe": spec["feature_recipe"], "parameters": spec["parameters"]})
    result["candidate_specs"].sort(key=lambda spec: spec["candidate_spec_id"])
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
