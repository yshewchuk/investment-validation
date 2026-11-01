"""IV-owned wire policy without importing search orchestration."""
from engine.v2.data.iv_crush_table import IV_CRUSH_TABLE_NAME
from engine.v2.ops.errors import fail
from engine.v2.ops.native_registration import validate_regression_registration
from experiments.iv_crush_plan import target


def validate_registration(document):
    """Require IV policy after shared wire validation, without reading source payloads."""
    validate_regression_registration(document)
    declared = document["execution_plan"]
    if (document["regression_plan"]["target_recipe"] != target()
            or declared.get("runner") != "experiments/iv_crush_search.py"
            or declared.get("price_source") != IV_CRUSH_TABLE_NAME):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "registration differs from signed IV research policy")
