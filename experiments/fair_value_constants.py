"""Fill model F, slice F6b: fitted constants document, tick rule and floored price-space prediction (design section 9).
Not in fvf_id; F's fill code does not use it. Applying the constants and the floor to leg fills is a later slice.
"""
import hashlib
import json

import numpy as np

from experiments import fair_value_fills as fvf

SPREAD_SENSITIVITIES = {"spread025": 0.25, "spread050": 0.50}
TICK_FLOOR = {"penny": 0.01, "nickel": 0.05, "dime": 0.10, "multiple": 0.05, "mid_break": 3.00}
FITTED = ("C_BASE", "C_SPREAD", "C_THIN_SIZE", "C_THIN_OI")


def tick_size(bid, ask):
    m = TICK_FLOOR["multiple"]
    grid = (np.abs(bid / m - np.round(bid / m)) < 1e-6) & (np.abs(ask / m - np.round(ask / m)) < 1e-6)
    mid = (bid + ask) / 2.0
    return np.where(~grid, TICK_FLOOR["penny"], np.where(mid < TICK_FLOOR["mid_break"] - 1e-9, TICK_FLOOR["nickel"], TICK_FLOOR["dime"]))


def concession(base, spread, size_w, oi_w, spread_term, thin_size, thin_oi):
    return np.minimum(1.0, base + spread * spread_term + size_w * thin_size + oi_w * thin_oi)


def predict_x(v, c, floor_x):
    return np.minimum(1.0, np.maximum(v + c * (1.0 - v), floor_x))


def primary_constants(c_base, c_spread, c_thin_size, c_thin_oi):
    fitted = {name: float(value) for name, value in zip(FITTED, (c_base, c_spread, c_thin_size, c_thin_oi))}
    return {**fitted, "SPREAD_SCALE": fvf.SPREAD_SCALE, "THIN_SIZE_BELOW": fvf.THIN_SIZE_BELOW, "THIN_OI_BELOW": fvf.THIN_OI_BELOW,
            "UNKNOWN_THIN": fvf.UNKNOWN_THIN, "GUARD_FILL_FRACTION": fvf.GUARD_FILL_FRACTION, "TICK_FLOOR": dict(TICK_FLOOR)}


def constants_id(constants):
    return hashlib.sha256(json.dumps(constants, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def constants_document(constants):
    return {"constants_id": constants_id(constants), "constants": constants,
            "sensitivity_variants": {name: {"C_SPREAD": v} for name, v in SPREAD_SENSITIVITIES.items()}}
