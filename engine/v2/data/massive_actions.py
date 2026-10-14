"""Provider-reported actions, never inferred price or total-return adjustments."""
import math
from dataclasses import replace
from datetime import date

from engine.v2.contracts import ColumnContract
from engine.v2.data import errors, manifests, massive_daily
from engine.v2.foundation import format_timestamp, parse_timestamp

NORMALIZER = "massive_actions_normalizer.v1"
_FIELDS = {
    "splits": {"execution_date": "string", "split_from": "float64", "split_to": "float64",
               "adjustment_type": "string", "historical_adjustment_factor": "float64"},
    "dividends": {"ex_dividend_date": "string", "cash_amount": "float64", "currency": "string",
        "declaration_date": "string", "record_date": "string", "pay_date": "string",
        "frequency": "int64", "distribution_type": "string", "split_adjusted_cash_amount": "float64",
        "historical_adjustment_factor": "float64"},
}
_REQUIRED = {"splits": ("execution_date", "split_from", "split_to"),
             "dividends": ("ex_dividend_date", "cash_amount", "currency")}


def _contract(dataset):
    columns = tuple(ColumnContract(name=n, physical_type="string", nullable=False)
                    for n in (*massive_daily._TEXT, "id"))
    columns += tuple(ColumnContract(name=n, physical_type=t, nullable=n not in _REQUIRED[dataset],
        unit="provider_reported", adjustment_basis="provider-reported value; no conversion")
        for n, t in _FIELDS[dataset].items())
    contract = replace(massive_daily.CONTRACT, contract_id=f"massive_{dataset}.v1",
        table_name=f"massive_{dataset}", columns=columns, primary_key=("ticker", "id"),
        orderable_columns=("ticker", "id"), duplicate_policy="duplicate provider event IDs refuse")
    return replace(contract, definition_hash=manifests.table_contract_hash(contract))


CONTRACTS = {dataset: _contract(dataset) for dataset in _FIELDS}


def _number(name, kind, value):
    if (type(value) not in (int, float) or not math.isfinite(value) or float(value) != value
            or value < 0 or (kind == "int64" and (type(value) is not int or value > 2**63 - 1))):
        raise ValueError
    if name in ("split_from", "split_to", "historical_adjustment_factor") and value == 0:
        raise ValueError


def _value(name, kind, value):
    if value is None:
        return None
    if kind == "string":
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ValueError
        if name.endswith("_date") and date.fromisoformat(value).isoformat() != value:
            raise ValueError
    else:
        _number(name, kind, value)
    return value


def normalize_rows(rows, request, binding, evidence):
    dataset, result = request["dataset"], []
    try:
        if request["adjustment_basis"] != "corporate_action":
            raise ValueError
        if any(format_timestamp(parse_timestamp(evidence[k])) != evidence[k]
               for k in ("first_received_at", "observed_at")):
            raise ValueError
        for row in rows:
            day = _value(_REQUIRED[dataset][0], "string", row[_REQUIRED[dataset][0]])
            if not binding["start"] <= day <= binding["end"]:
                continue
            values = {k: _value(k, t, row.get(k)) for k, t in _FIELDS[dataset].items()}
            if (any(values[k] is None for k in _REQUIRED[dataset])
                    or _value("ticker", "string", row["ticker"]).upper() != request["ticker"]):
                raise ValueError
            identity = _value("id", "string", row["id"])
            if identity is None:
                raise ValueError
            result.append({"ticker": request["ticker"], "id": identity, "date": day,
                "adjustment_basis": "corporate_action", "occurrence_id": binding["occurrence_id"],
                **values, **{k: evidence[k] for k in
                            ("raw_receipt_id", "raw_hash", "first_received_at", "observed_at")}})
    except (ValueError, TypeError, KeyError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "Massive action row is invalid") from None
    return result
