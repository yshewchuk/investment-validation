"""EXP-189 A1: the spec selects model F via its one amendment carrying ``fill_model``, id "F" (A1.7 R5, A1.9); F runs are refused until the F wiring slice (W7d2) lands."""
from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as fvr


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


def open_run(fill):
    """The public run boundary: None (current marks) for no selection. Every F selection refuses until W7d2 wires F."""
    if fill is None:
        return None
    if not isinstance(fill, dict):
        raise fvi.FairValueRefusal("fill_model_invalid", "fill_model is not a mapping")
    if fill.get("id") != "F":
        raise fvi.FairValueRefusal("fill_model_id", "fill_model id is not F")
    fvf = fill.get("fvf_id")
    if not isinstance(fvf, str) or not fvf:
        raise fvi.FairValueRefusal("fill_model_invalid", "fill_model has no fvf_id string")
    fvr.require_fvf_id(fvf)
    raise fvi.FairValueRefusal("f_run_not_wired", "the spec selects F, but F runs are not wired yet; an F spec never runs on current marks")
