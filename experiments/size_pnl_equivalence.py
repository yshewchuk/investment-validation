"""EXP-188 S9c, design guides/size_decision_pnl_design.md section 4: the fast-path equivalence preflight.

``build_sample`` draws a fixed sample of simulable events over the real ladder chunk, one structure per event,
spread across ``NAMES``. ``check_chunk`` holds the linear fast path to ``expected_pnl`` on that sample and
refuses above 1e-9 (``check_equivalence``). Pure functions: no I/O, no ledger.
"""
import math

from engine.fills import FillModel
from experiments import size_pnl_shortvol as shortvol
from experiments.size_pnl_pool import check_equivalence

SAMPLE = 8


def build_sample(chunk, events, sims, *, alpha, size=SAMPLE):
    """Up to ``size`` check items, one per event, in ``str(event_id)`` order so the sample ignores row order."""
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    fill, sample = FillModel(alpha), []
    for ev in sorted(events.to_dict("records"), key=lambda r: str(r["event_id"])):
        if len(sample) >= size:
            break
        eid = str(ev["event_id"])
        sim = sims.get(eid)
        if sim is None or eid not in pos:
            continue
        i = pos[eid]
        for k in range(len(shortvol.NAMES)):  # the structure offset spreads the sample over the names
            name = shortvol.NAMES[(len(sample) + k) % len(shortvol.NAMES)]
            priced, legs, _ = shortvol.resolve_priced(chunk, i, name, ev["pred_move"], fill)
            entered = None if legs is None else shortvol.entry(priced)
            if entered is None or not (math.isfinite(entered[0]) and entered[0] >= shortvol.MIN_DEBIT):
                continue
            exits = [{"strike": leg["strike"], "side": shortvol._CLOSE[leg["side"]], "qty": leg["qty"]}
                     for leg in legs if leg["qty"] > 0]
            sample.append({"key": eid, "event_date": ev["event_date"], "spot": float(chunk["spot_entry"][i]),
                           "pre_iv30": shortvol._num(ev["pre_iv30"]), "pred_move": shortvol._num(ev["pred_move"]),
                           "pred_crush": shortvol._num(ev["pred_crush"]), "dte_exit": shortvol._num(ev["dte_exit"]),
                           "strikes": sim[0], "legs": exits, "entry_cost": entered[0]})
            break
    return sample


def check_chunk(chunk, events, sims, pool, *, alpha):
    """None when the sample is empty; otherwise the ``check_equivalence`` summary (raises on a mismatch)."""
    sample = build_sample(chunk, events, sims, alpha=alpha)
    if not sample:
        return None
    return check_equivalence(pool, sample)
