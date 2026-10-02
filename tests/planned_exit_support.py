"""Right-aware reference for the planned-exit simulation parity tests.

``pnl_sim.expected_pnl`` values every exit leg with the put formula and has no
notion of a leg's right. This mirrors its draw, seed and aggregation step for
step (same pool, same seed material, same put kernel) and only adds the call
case, valued as ``put + (spot - strike)`` (put-call parity, zero rates and
dividends). On put-only legs it equals ``pnl_sim.expected_pnl`` bit for bit.
"""
import hashlib

import numpy as np

from engine import pnl_sim


def legacy_expected_pnl(
    *, exit_legs, spot, entry_cost, pre_iv30, pred_abs_move, pred_iv_crush,
    dte_exit, event_date, pool, key="", draws=pnl_sim.DRAWS,
):
    """Each exit leg is ``{"strike", "qty", "side", "right"}``; right defaults to put."""
    seed = int.from_bytes(
        hashlib.sha256(f"{key}|{event_date}".encode()).digest()[:8], "big",
    )
    rng = np.random.default_rng(seed)
    err_move, err_crush = pool.draw(event_date, pred_abs_move, draws, rng)
    move = np.maximum(pred_abs_move + err_move, 0.0)
    crush = pred_iv_crush + err_crush
    sign = rng.choice((-1.0, 1.0), size=move.size)
    spot_exit = spot * np.maximum(
        1.0 + sign * move / 100.0, pnl_sim.MIN_SPOT_FRACTION,
    )
    vol_exit = (pre_iv30 / 100.0) * (1.0 + crush / 100.0)
    value = np.zeros(move.size)
    for leg in exit_legs:
        strike, qty = float(leg["strike"]), float(leg["qty"])
        side = 1.0 if str(leg["side"]).lower() == "sell" else -1.0
        leg_value = pnl_sim.black_scholes_put(
            spot_exit, strike, dte_exit / 365.0, vol_exit,
        )
        if str(leg.get("right", "P")).upper() in {"C", "CALL"}:
            leg_value = leg_value + (spot_exit - strike)
        value += side * qty * leg_value
    returns = (value - entry_cost) / entry_cost
    return {
        "exp_pnl_sim": float(np.mean(returns)),
        "win_sim": float(np.mean(returns > 0)),
        "sim_p10": float(np.quantile(returns, 0.10)),
        "sim_p90": float(np.quantile(returns, 0.90)),
        "pool_n": int(pool.before(event_date)),
    }
