"""Separate experiment-qualified realized history; live contracts are not aliases."""
from dataclasses import replace

from engine.v2.data.errors import fail
from engine.v2.data.manifests import table_contract_hash
from engine.v2.foundation import to_document
from engine.v2.research.magnitude_history_table import CONTRACT as _V1
from engine.v2.research.magnitude_history_table import FEATURES
from engine.v2.research.magnitude_history_table import METADATA as METADATA

TABLE = "research_eligible_history"
_base = replace(_V1, table_name=TABLE, contract_id=TABLE + ".v1", semantic_version="1.0.0",
    provenance_semantics="exact source anchors, grouped eligibility and contributor identity",
    coverage_semantics="exact requested event/horizon set; policy-qualified reconstructed history")
CONTRACT = replace(_base, definition_hash=table_contract_hash(_base))


def recipe():
    """Closed provenance, not a caller-controlled research-safe flag."""
    return {"schema_version": "eligible_history_features.v1", "usage": "experiment",
        "contract": to_document(CONTRACT), "fields": list(FEATURES),
        "arithmetic": "panel_math.history_features", "units": "percentage_points",
        "missing": "native_null_and_warmup", "implied_quote_history": "unsupported"}


def admit_fields(contract, context, *, policy_id, columns):
    """Refuse live/unknown provenance and unsupported aliases or derivations."""
    if (contract != CONTRACT or context.get("feature_recipe") != recipe()
            or context.get("eligibility", {}).get("policy", {}).get("policy_id") != policy_id
            or type(columns) not in (list, tuple) or not columns
            or any(type(name) is not str or name not in FEATURES for name in columns)
            or len(set(columns)) != len(columns)):
        raise fail("HOLDOUT_ACCESS_DENIED", "experiment fields require exact eligible-history provenance")
