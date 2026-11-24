"""EXP-189 W5: nearest-feasible-width fallback arm (design 12.3), its recovery statistics and T9."""
import sys
import time

import pandas as pd

from engine.structures import BUY
from experiments import size_pnl_shortvol as sv
from experiments.size_width_band import PHIS, band_range, frozen_bands, rho_ladder, tier
from experiments.size_width_sweep import TAKEN_COLS, ALPHA, E_ARM, INNER, Sweep, admit_k, in_window, scaled
from experiments.size_width_tuning import evaluate


def floor_ok(legs):
    """True when the net put position of ``legs`` (dicts with strike, side, qty) never pays below zero at expiry."""
    def payoff(spot):
        return sum((1.0 if leg["side"] == BUY else -1.0) * leg["qty"] * max(leg["strike"] - spot, 0.0) for leg in legs)
    return min(payoff(s) for s in [0.0, *(leg["strike"] for leg in legs)]) >= -1e-9


def recover(chunk, events, sims, name, frame, k, ladder):
    """(out_frame, found): each ladder_too_coarse row takes the FIRST rho of ``ladder`` that scores and passes floor_ok; ``k`` is effective."""
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    ids = events["event_id"].astype(str)
    recs = [{**row, "k": float(k)} for row in frame.to_dict("records")]
    where = {row["event_id"]: i for i, row in enumerate(recs)}
    pending = [row["event_id"] for row in recs if row["reason"] == "ladder_too_coarse"]
    found = dict.fromkeys(pending)
    for rho in ladder:
        if not pending:
            break
        moved = scaled(events[ids.isin(pending)], k * rho)
        trial = sv.score_events(chunk, moved, sims, name, alpha=ALPHA)
        pred = dict(zip(moved["event_id"].astype(str), moved["pred_move"]))
        for row in trial.to_dict("records"):
            eid = str(row["event_id"])
            if row["status"] != "scored" or not floor_ok(sv.resolve(chunk, pos[eid], name, pred[eid])[0]):
                continue
            recs[where[eid]] = {**row, "k": float(k * rho)}
            found[eid] = rho
            pending.remove(eid)
    return pd.DataFrame(recs, columns=[*sv.SCORE_COLS, "k"]), found


def _mcap(sweep, arm_of):
    """event_id -> mcap_usd over every scored row of ``sweep`` at ``arm_of(name)`` for the seven structures."""
    return {str(e): m for name in sv.NAMES for f in sweep.scored.get((name, arm_of(name)), []) for e, m in zip(f["event_id"], f["mcap_usd"])}


def _mean(values):
    return float(values.astype(float).mean()) if len(values) else None


class FallbackSweep(Sweep):
    """Arm E is the only arm: ladder_too_coarse rows recover at the nearest feasible rho when the structure has a band; ``k`` is effective."""
    def __init__(self, factors, out, inputs_hash, phi):
        """Bands come only from the frozen tuning_band receipt under ``out``: ``frozen_bands`` refuses a missing, changed or malformed receipt."""
        super().__init__()
        body = frozen_bands(out, inputs_hash)
        if phi not in PHIS:
            raise ValueError(f"phi must be one of {PHIS}")
        if any(n not in factors for n in sv.NAMES):
            raise ValueError(f"no factor for {[n for n in sv.NAMES if n not in factors]}")
        self.phi = phi
        self.factors, self.bands, self.trials = {n: admit_k(factors[n]) for n in sv.NAMES}, {n: band_range(body, n, phi) for n in sv.NAMES}, []

    def _frames(self, chunk, events, sims, name):
        k = self.factors[name]
        frame = sv.score_events(chunk, scaled(events, k), sims, name, alpha=ALPHA)
        if self.bands[name] is None:
            return [(E_ARM, frame.assign(k=k))]
        t0 = time.time()
        ladder = rho_ladder(*self.bands[name], k)
        out, found = recover(chunk, events, sims, name, frame, k, ladder)
        print(f"[width_fallback] {name} k={k} declined={len(found)} recovered={sum(r is not None for r in found.values())} rungs={len(ladder)} elapsed={time.time() - t0:.1f}s", file=sys.stderr)
        dates = dict(zip(frame["event_id"], frame["event_date"]))
        self.trials += [{"structure": name, "event_id": eid, "event_date": dates[eid], "rho": rho} for eid, rho in found.items()]
        return [(E_ARM, out)]


def _trades(sweep, arm_of, window):
    """Taken rows of the seven structures at ``arm_of(name)`` in ``window`` with ``structure``, ``ret`` and ``tier`` columns."""
    mcap, parts = _mcap(sweep, arm_of), []
    for name in sv.NAMES:
        rows = sweep.rows([(name, arm_of(name))], window)
        if not rows.empty:
            parts.append(rows.assign(structure=name, ret=rows["exit_value_mid"] / rows["entry_cost"] - 1.0,
                                     tier=[tier(mcap.get(str(e))) for e in rows["event_id"]]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*TAKEN_COLS, "ret", "tier"])


def split_trades(fb, window):
    """``_trades`` of the fallback arm in ``window`` plus a ``recovered`` flag."""
    recovered = {(t["structure"], str(t["event_id"])) for t in fb.trials if t["rho"] is not None}
    trades = _trades(fb, lambda name: E_ARM, window)
    return trades.assign(recovered=[(s, str(e)) in recovered for s, e in zip(trades["structure"], trades["event_id"])])


def recovery(fb, base, window=INNER):
    """Recovered taken trades of ``fb`` (count, mean ret); ``taken_ret`` is the mean ret of ALL no-fallback ``base`` trades at fb.factors in ``window``."""
    if base is None:
        raise ValueError("recovery needs the no-fallback base sweep")
    trades = split_trades(fb, window)
    flag = trades["recovered"].astype(bool)
    return {"recovered": int(flag.sum()), "recovered_ret": _mean(trades.loc[flag, "ret"]),
            "taken_ret": _mean(_trades(base, lambda name: fb.factors[name], window)["ret"])}


def _stats(rows, tag):
    return {f"{tag}_trades": len(rows), f"{tag}_ret": _mean(rows["ret"]), f"{tag}_pnl": _mean(rows["pnl_dollars_mid"])}


def t9(fb, base, phi, window):
    """One row per (structure, tier) with a trial in ``window``; taken stats come from ``base`` at fb.factors on the same events (design 12.3)."""
    if base is None:
        raise ValueError("t9 needs the no-fallback base sweep")
    if phi != fb.phi:
        raise ValueError(f"t9 phi {phi} does not match the sweep phi {fb.phi}")
    mcap = _mcap(fb, lambda name: E_ARM)
    trials = in_window(pd.DataFrame(fb.trials, columns=["structure", "event_id", "event_date", "rho"]), window).assign(tier=lambda d: [tier(mcap.get(str(e))) for e in d["event_id"]])
    trades, taken, rows = split_trades(fb, window), _trades(base, lambda name: fb.factors[name], window), []
    for (name, t), grp in trials.groupby(["structure", "tier"]):
        got, band = grp["rho"].notna(), fb.bands.get(name)
        cell = trades[(trades["structure"] == name) & (trades["tier"] == t)]
        flag = cell["recovered"].astype(bool)
        base_cell = taken[(taken["structure"] == name) & (taken["tier"] == t)]
        rows.append({"phi": phi, "structure": name, "tier": t, "lo": band and band[0], "hi": band and band[1],
                     "declined": len(grp), "recovered": int(got.sum()), "fallback_infeasible": int((~got).sum()),
                     "share": float(got.sum() / len(grp)),
                     "rho": {str(float(r)): int(c) for r, c in grp["rho"].dropna().value_counts().items()},
                     **_stats(cell[flag], "recovered"), **_stats(base_cell, "taken")})
    return rows


def accounts(fb, base, windows):
    """{label: {"fallback", "base"}}: the seven-structure total return with every structure at E_ARM, and the plain Sweep at the same factors."""
    return {label: {"fallback": evaluate(fb, {n: E_ARM for n in sv.NAMES}, w)["account"]["total_return"],
                    "base": evaluate(base, fb.factors, w)["account"]["total_return"]} for label, w in windows.items()}
