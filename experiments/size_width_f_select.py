"""EXP-189 A1: the spec selects model F via its one amendment carrying ``fill_model``, id "F" (A1.7 R5, A1.9). ``open_run`` builds the F run (pricing and receipt binding); ``check_report`` refuses a report whose inputs do not bind what this registration selects."""
import hashlib
import json
from collections import namedtuple
from pathlib import Path

from engine import paths
from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as fvr
from experiments import size_pnl_receipts as receipts
from experiments import size_width_f_fills, size_width_f_quotes, size_width_pricing

MODE = "f7_orats_borrow_discrete_dividend"
FInputs = namedtuple("FInputs", "pins read_quotes snapshot_id")
FRun = namedtuple("FRun", "pricing binding")


def _validate(fill):
    """The one shared check of a selected fill_model: a mapping with id "F" and a matching fvf_id. Returns it."""
    if not isinstance(fill, dict):
        raise fvi.FairValueRefusal("fill_model_invalid", "fill_model is not a mapping")
    if fill.get("id") != "F":
        raise fvi.FairValueRefusal("fill_model_id", "fill_model id is not F")
    fvf = fill.get("fvf_id")
    if not isinstance(fvf, str) or not fvf:
        raise fvi.FairValueRefusal("fill_model_invalid", "fill_model has no fvf_id string")
    fvr.require_fvf_id(fvf)
    return fill


def fill_model(spec):
    """The validated dict of the one amendment carrying ``fill_model`` (id "F"), or None when no amendment carries one."""
    carriers = [a["fill_model"] for a in spec.get("amendments", []) if isinstance(a, dict) and "fill_model" in a]
    if len(carriers) > 1:
        raise fvi.FairValueRefusal("fill_model_ambiguous", f"{len(carriers)} amendments carry fill_model")
    if not carriers:
        return None
    return _validate(carriers[0])


def open_run(fill, inputs=None, plan=None, expected_snapshot_id=None, *, root=None):
    """The run boundary: None (current marks) with no selection; else the F run's pricing and binding, or a refusal before any quote is read."""
    if fill is None:
        if inputs is not None: raise fvi.FairValueRefusal("f_inputs_unexpected", "F inputs were given, but the spec selects no F")
        return None
    _validate(fill)
    if inputs is None: raise fvi.FairValueRefusal("f_inputs_missing", "the spec selects F: pass its pinned inputs (--f-pins and --f-quotes)")
    if not isinstance(inputs, FInputs): raise fvi.FairValueRefusal("f_inputs_invalid", "f_inputs is not a FInputs (pins, read_quotes, snapshot_id)")
    size_width_f_quotes.admit_inputs(inputs.pins, inputs.read_quotes)
    r = fill["reingest"] if isinstance(fill.get("reingest"), dict) else {}
    rules, pins = fill.get("rate_rules"), {**inputs.pins, "files": dict(inputs.pins["files"])}  # a private copy: a later change to the caller's mapping reaches neither the receipt nor the loads
    if not (all(k in fill for k in ("carry_dividend_mode", "rate_rules", "vol_scale", "rate_scale", "reingest")) and isinstance(rules, dict) and set(rules) == set(fvi.RATE_RULE_VALUES)
            and all(rules[k] in v for k, v in fvi.RATE_RULE_VALUES.items()) and isinstance(r.get("path"), str) and r.get("path")
            and not Path(r["path"]).is_absolute() and ".." not in Path(r["path"]).parts and isinstance(r.get("manifest_sha256"), str)
            and isinstance(r.get("manifest_units"), dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in r["manifest_units"].items())):
        raise fvi.FairValueRefusal("fill_model_invalid", "fill_model lacks a field, or its rate rules, reingest path, hash or units are malformed")
    if fill["carry_dividend_mode"] != MODE: raise fvi.FairValueRefusal("carry_dividend_mode", "the carry and dividend mode is not the registered one")
    tables = Path(paths.ROOT if root is None else root) / r["path"]
    vol, rate = fvi._unit_scale(fill["vol_scale"], "vol_scale"), fvi._unit_scale(fill["rate_scale"], "rate_scale")
    blob = fvi._read_capped(tables / "manifest.json", fvi.MAX_MANIFEST_BYTES, "manifest.json") if (tables / "manifest.json").is_file() else b""
    try:  # an absent or non-object manifest never matches the spec's units
        units = json.loads(blob).get("units")
    except (ValueError, AttributeError):
        units = None
    if not (hashlib.sha256(blob).hexdigest() == pins["manifest_sha256"] == r["manifest_sha256"] and units == r["manifest_units"]):
        raise fvi.FairValueRefusal("manifest_binding", "the manifest, the pins, or the reingest hash or units differ")
    quotes = size_width_f_quotes.FQuotes(plan, inputs.read_quotes, tables_root=tables, pins=pins, snapshot_id=inputs.snapshot_id,
                                         expected_snapshot_id=expected_snapshot_id, vol_scale=vol, rate_scale=rate, rate_rules=rules)
    binding = {"fill_model_id": "F", "fvf_id": fill["fvf_id"], "reingest_manifest_sha256": r["manifest_sha256"],
               "every_table_file_sha256": dict(sorted(pins["files"].items())), "quote_snapshot_id": inputs.snapshot_id,
               "units": {"vol_scale": vol, "rate_scale": rate, "manifest_units": r["manifest_units"]},
               "rate_out_of_range": rules["rate_out_of_range"], "rate_empty_row": rules["rate_empty_row"],
               "carry_dividend_mode": fill["carry_dividend_mode"]}
    return FRun(size_width_pricing.load(fills=size_width_f_fills.FFills(quotes)), binding)


def check_report(receipt, out_dir):
    """The report's out-dir inputs must bind what this registration selects: F under this spec_hash, or no fill model (A1.7 R5)."""
    fill, path = receipt.get("fill_model"), Path(out_dir) / "receipts" / "inputs_inner.json"
    if fill is None and not path.exists(): return  # nothing bound; the report's own load refuses a missing receipt
    if fill is not None: _validate(fill)
    inputs = receipts.load_receipt(path)
    bound = inputs.get("fill_model")
    ok = "fill_model" not in inputs if fill is None else (isinstance(bound, dict) and bound.get("fill_model_id") == "F" and inputs.get("spec_hash") == receipt.get("spec_hash"))
    if not ok:
        raise fvi.FairValueRefusal("f_report_unbound", "the out-dir's inputs do not bind what this registration selects")
