"""EXP-189 A1: the spec selects model F via its one amendment carrying ``fill_model``, id "F" (A1.7 R4/R5, A1.9)."""
import collections
from pathlib import Path

from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as fvr
from experiments import size_width_f_fills, size_width_f_quotes, size_width_pricing

MODE = "f7_orats_borrow_discrete_dividend"


def fill_model(spec):
    """The dict of the one amendment carrying ``fill_model`` (id "F"), or None when no amendment carries one."""
    carriers = [a["fill_model"] for a in spec.get("amendments", []) if isinstance(a, dict) and "fill_model" in a]
    if len(carriers) > 1:
        raise fvi.FairValueRefusal("fill_model_ambiguous", f"{len(carriers)} amendments carry fill_model")
    if not carriers:
        return None
    if not isinstance(carriers[0], dict):
        raise fvi.FairValueRefusal("fill_model_invalid", "fill_model is not a mapping")
    if carriers[0].get("id") != "F":
        raise fvi.FairValueRefusal("fill_model_id", "fill_model id is not F")
    return carriers[0]


FInputs = collections.namedtuple("FInputs", "pins read_quotes snapshot_id")
FRun = collections.namedtuple("FRun", "pricing binding")


def open_run(fill, inputs, plan, expected_snapshot_id, *, root=Path(__file__).resolve().parents[1]):
    """The F run that ``fill`` selects (None: current marks). Every mismatch refuses before a quote is read."""
    if fill is None:
        if inputs is not None:
            raise fvi.FairValueRefusal("f_inputs_unexpected", "F inputs were given but the spec does not select F")
        return None
    if inputs is None:
        raise fvi.FairValueRefusal("f_inputs_missing", "the spec selects F but no F inputs were given")
    try:
        mode, fvf, rules = fill["carry_dividend_mode"], fill["fvf_id"], fill["rate_rules"]
        vol, rate, reingest = fill["vol_scale"], fill["rate_scale"], fill["reingest"]
        path, sha, units = reingest["path"], reingest["manifest_sha256"], reingest["manifest_units"]
    except (KeyError, TypeError) as exc:
        raise fvi.FairValueRefusal("fill_model_invalid", "fill_model lacks a required field") from exc
    if not (isinstance(rules, dict) and set(rules) == set(fvi.RATE_RULE_VALUES)
            and all(rules[k] in fvi.RATE_RULE_VALUES[k] for k in rules)
            and isinstance(path, str) and bool(path) and not Path(path).is_absolute() and ".." not in Path(path).parts
            and isinstance(sha, str) and bool(sha) and isinstance(fvf, str) and bool(fvf) and isinstance(units, dict)):
        raise fvi.FairValueRefusal("fill_model_invalid", "fill_model holds a malformed rate rule, path, hash, id or units")
    if mode != MODE:
        raise fvi.FairValueRefusal("carry_dividend_mode", f"carry_dividend_mode is not {MODE}")
    fvr.require_fvf_id(fvf)
    pins = inputs.pins
    if not (isinstance(pins, dict) and isinstance(pins.get("manifest_sha256"), str) and isinstance(pins.get("files"), dict)
            and pins["files"] and all(isinstance(k, str) and isinstance(v, str) for k, v in pins["files"].items())):
        raise fvi.FairValueRefusal("pins_invalid", "pins need a manifest_sha256 string and a non-empty files dict of strings")
    if pins["manifest_sha256"] != sha:
        raise fvi.FairValueRefusal("manifest_hash", "pins' manifest_sha256 differs from the spec's reingest hash")
    scales = {"vol_scale": fvi._unit_scale(vol, "vol_scale"), "rate_scale": fvi._unit_scale(rate, "rate_scale")}
    quotes = size_width_f_quotes.FQuotes(plan, inputs.read_quotes, tables_root=Path(root) / path, pins=pins,
                                         snapshot_id=inputs.snapshot_id, expected_snapshot_id=expected_snapshot_id,
                                         rate_rules=rules, **scales)
    binding = {"fill_model_id": "F", "fvf_id": fvf, "reingest_manifest_sha256": sha,
               "every_table_file_sha256": dict(sorted(pins["files"].items())), "quote_snapshot_id": expected_snapshot_id,
               "units": {**scales, "manifest_units": units}, "rate_out_of_range": rules["rate_out_of_range"],
               "rate_empty_row": rules["rate_empty_row"], "carry_dividend_mode": mode}
    return FRun(size_width_pricing.load(fills=size_width_f_fills.FFills(quotes)), binding)
