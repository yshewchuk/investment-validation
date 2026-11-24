"""EXP-189 W5: nearest-feasible-width fallback arm (design 12.3), its recovery statistics and T9."""
import pandas as pd

from engine.structures import BUY
from experiments import size_pnl_shortvol as sv
from experiments.size_width_band import rho_ladder, tier
from experiments.size_width_sweep import _TAKEN, ALPHA, E_ARM, INNER, Sweep, admit_k, in_window, scaled
from experiments.size_width_tuning import evaluate


def floor_ok(legs):
    """True when the net put position of ``legs`` (dicts with strike, side, qty) never pays below zero at expiry."""
    def payoff(spot):
        return sum((1.0 if leg["side"] == BUY else -1.0) * leg["qty"] * max(leg["strike"] - spot, 0.0) for leg in legs)
    return min(payoff(s) for s in [0.0, *(leg["strike"] for leg in legs)]) >= -1e-9


def recover(chunk, events, sims, name, frame, k, ladder):
    """(out_frame, found): each ladder_too_coarse row takes the FIRST rho of ``ladder`` whose trial scores and passes floor_ok.
    Every row's ``k`` is its effective multiplier; rho never reads exp_pnl_sim or labels."""
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


def _mcap(fb):
    """event_id -> mcap_usd over every scored row of the fallback arm."""
    return {str(e): m for (_n, arm), frames in fb.scored.items() if arm == E_ARM
            for f in frames for e, m in zip(f["event_id"], f["mcap_usd"])}


def _mean(values):
    return float(values.astype(float).mean()) if len(values) else None


class FallbackSweep(Sweep):
    """Arm E is the only arm: ladder_too_coarse rows are recovered at the nearest feasible rho when the structure has a band.
    It reuses Sweep's per-event-k path, so the k column is the effective multiplier."""
    def __init__(self, factors, bands):
        super().__init__()
        if any(n not in factors for n in sv.NAMES):
            raise ValueError(f"no factor for {[n for n in sv.NAMES if n not in factors]}")
        self.factors, self.bands, self.trials = {n: admit_k(factors[n]) for n in sv.NAMES}, {n: bands.get(n) for n in sv.NAMES}, []

    def _frames(self, chunk, events, sims, name):
        k = self.factors[name]
        frame = sv.score_events(chunk, scaled(events, k), sims, name, alpha=ALPHA)
        if self.bands[name] is None:
            return [(E_ARM, frame.assign(k=k))]
        out, found = recover(chunk, events, sims, name, frame, k, rho_ladder(*self.bands[name], k))
        dates = dict(zip(frame["event_id"], frame["event_date"]))
        self.trials += [{"structure": name, "event_id": eid, "event_date": dates[eid], "rho": rho} for eid, rho in found.items()]
        return [(E_ARM, out)]


def split_trades(fb, window):
    """Taken rows of the seven structures in ``window`` with ``structure``, ``ret``, ``recovered`` and ``tier`` columns."""
    recovered, mcap, parts = {(t["structure"], str(t["event_id"])) for t in fb.trials if t["rho"] is not None}, _mcap(fb), []
    for name in sv.NAMES:
        rows = fb.rows([(name, E_ARM)], window)
        ids = rows["event_id"].astype(str)
        if not rows.empty:
            parts.append(rows.assign(structure=name, ret=rows["exit_value_mid"] / rows["entry_cost"] - 1.0,
                                     recovered=[(name, e) in recovered for e in ids], tier=[tier(mcap.get(e)) for e in ids]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*_TAKEN, "ret", "recovered", "tier"])


def recovery(fb, window=INNER):
    """Recovered taken trades (count, mean ret) and the mean ret of the other taken trades, pooled over structures."""
    trades = split_trades(fb, window)
    flag = trades["recovered"].astype(bool)
    return {"recovered": int(flag.sum()), "recovered_ret": _mean(trades.loc[flag, "ret"]),
            "taken_ret": _mean(trades.loc[~flag, "ret"])}


def _stats(rows, tag):
    return {f"{tag}_trades": len(rows), f"{tag}_ret": _mean(rows["ret"]), f"{tag}_pnl": _mean(rows["pnl_dollars_mid"])}


def t9(fb, base, phi, window):
    """One row per (structure, tier) with a trial in ``window``. ``base`` is accepted for the call shape of ``accounts``, unused here."""
    mcap = _mcap(fb)
    trials = in_window(pd.DataFrame(fb.trials, columns=["structure", "event_id", "event_date", "rho"]), window)
    trials = trials.assign(tier=[tier(mcap.get(str(e))) for e in trials["event_id"]])
    trades, rows = split_trades(fb, window), []
    for (name, t), grp in trials.groupby(["structure", "tier"]):
        got, band = grp["rho"].notna(), fb.bands.get(name)
        cell = trades[(trades["structure"] == name) & (trades["tier"] == t)]
        flag = cell["recovered"].astype(bool)
        rows.append({"phi": phi, "structure": name, "tier": t, "lo": band and band[0], "hi": band and band[1],
                     "declined": len(grp), "recovered": int(got.sum()), "fallback_infeasible": int((~got).sum()),
                     "share": float(got.sum() / len(grp)),
                     "rho": {str(float(r)): int(c) for r, c in grp["rho"].dropna().value_counts().items()},
                     **_stats(cell[flag], "recovered"), **_stats(cell[~flag], "taken")})
    return rows


def accounts(fb, base, windows):
    """{label: {"fallback", "base"}}: the seven-structure total return with every structure at E_ARM, and the plain Sweep at the same factors."""
    arm = {n: E_ARM for n in sv.NAMES}
    return {label: {"fallback": evaluate(fb, arm, w)["account"]["total_return"],
                    "base": evaluate(base, fb.factors, w)["account"]["total_return"]} for label, w in windows.items()}
