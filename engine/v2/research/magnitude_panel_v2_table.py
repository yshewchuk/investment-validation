"""Additive v2 panel definitions; no context, source access or row admission."""
import json
from dataclasses import replace

from engine.v2.contracts.data import TableContract
from engine.v2.data.errors import fail
from engine.v2.data.manifests import table_contract_hash
from engine.v2.foundation import content_hash
from engine.v2.research import magnitude_panel_table as v1
from engine.v2.research.magnitude_panel_v2_policies import policy_definitions

__all__ = ["CONTRACT", "feature_dictionary", "recipe"]


def feature_dictionary(subset):
    """Return ordered, detached ten-field descriptors for one explicit subset."""
    if type(subset) is not str or subset not in ("H", "HM", "HMO"):
        raise fail("CONTRACT_MISMATCH", "unknown v2 panel feature subset")
    fields = v1.feature_dictionary(subset)
    if subset != "H":
        index = next(i for i, field in enumerate(fields) if field["name"] == "dist_ema")
        fields.insert(index + 1, {**fields[index], "name": "abs_dist_ema",
            "source_alias": "abs_dist_ema", "native_output_alias": "abs_dist_ema"})
    if subset == "HMO":
        implied = next(field for field in fields if field["name"] == "implied_move")
        fields.append({**implied, "name": "mean_prior_or_implied", "block": "implied_history",
            "source_alias": "mean_prior_or_implied", "native_output_alias": "mean_prior_or_implied"})
    policies = policy_definitions()
    roles = {"history": ("stored_history",), "runup": ("runup_prices",),
        "spy": ("market_context",), "daily_market": ("daily_market",),
        "implied_history": ("daily_market", "event_metadata")}
    for field in fields:
        name = field["name"]
        if field["block"] == "prior_events":
            dependencies = ("prior_outcomes",) if name == "signed_streak" else ("stored_history",)
        else:
            dependencies = roles[field["block"]]
        field["source_dependencies"] = [{"role": role, "policy": policies["source_policies"][role]}
            for role in dependencies]
        field["derivation_policy"] = policies["derivation_policies"].get(name)
    return json.loads(json.dumps(fields))


def recipe(subset):
    """Static raw projection, not a fitted family recipe or resolved source binding."""
    dictionary = feature_dictionary(subset)
    policies = policy_definitions()
    quote = policies["quote_indicator"] if subset == "HMO" else None
    return {"schema_version": "research_magnitude_panel.v2", "subset": subset,
        "dictionary": dictionary, "dictionary_hash": content_hash(dictionary), "imputation": "none",
        "missing_indicators": policies["missing_indicators"], "quote_indicator": quote,
        "model_columns": [field["name"] for field in dictionary] + ([quote["name"]] if quote else [])}


def _contract():
    columns = []
    for column in v1.CONTRACT.columns:
        columns.append(column)
        if column.name == "dist_ema":
            columns.append(replace(column, name="abs_dist_ema"))
    implied = next(column for column in v1.CONTRACT.columns if column.name == "implied_move")
    columns.append(replace(implied, name="mean_prior_or_implied"))
    contract = replace(v1.CONTRACT, contract_id="research_magnitude_panel.v2", semantic_version="2.0.0",
        columns=tuple(columns), definition_hash="sha256:" + "0" * 64)
    return replace(contract, definition_hash=table_contract_hash(contract))


CONTRACT: TableContract = _contract()
