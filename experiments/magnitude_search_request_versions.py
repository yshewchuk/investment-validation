"""Additive dictionary dispatch for source-free stage-2 declaration previews."""
import json

from engine.v2.foundation import content_hash
from engine.v2.research.magnitude_panel_qualified_table import recipe as v3_recipe
from engine.v2.research.magnitude_panel_table import recipe as v1_recipe
from engine.v2.research.magnitude_panel_v2_table import recipe as v2_recipe
from experiments.magnitude_search_request import compile_request as compile_v1_request

__all__ = ["compile_request"]

_V2_HASH = v2_recipe("HMO")["dictionary_hash"]
_V3_HASH = v3_recipe("HMO")["dictionary_hash"]


def compile_request(request):
    """Preserve v1 replay; adapt only exact full-HMO v2/v3 declaration hashes.

    The surrogate is private validation state, never a source or fit identity.
    Public v1 validation checks caps before expansion. No admission, I/O or fit
    occurs; fixed metadata work is not an input-byte or resource-feasibility cap.
    """
    dictionary = request.get("feature_dictionary_hash") if type(request) is dict else None
    selected = v3_recipe if dictionary == _V3_HASH else v2_recipe
    if (type(request) is not dict
            or type(request.get("feature_dictionary_hash")) is not str
            or request["feature_dictionary_hash"] not in (_V2_HASH, _V3_HASH)):
        return compile_v1_request(request)

    surrogate = dict(request)
    surrogate["feature_dictionary_hash"] = v1_recipe("HMO")["dictionary_hash"]
    result = compile_v1_request(surrogate)
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
