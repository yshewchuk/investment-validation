"""EXP-189 pricing seam: a private copy of size_pnl_shortvol whose strike resolution runs through engine.chain_index.price_structure_indexed.

``load()`` re-executes the sv source into a fresh module (the real module is untouched); ``chain_snapshot`` caches one snapshot per event of the current chunk. The entry and exit marks of the copy come from ``fills`` (default ``size_width_fills.CurrentFills``, today's marks). The fills get the Trade (event and resolved legs); skip reasons reach score_events rows; exit flags and an unquoted exit reach score rows."""
import importlib.util

import numpy as np
import pandas as pd

from engine.chain_index import price_structure_indexed
from experiments import size_pnl_shortvol as sv
from experiments.size_pnl_straddle import EXIT_UNQUOTED
from experiments.size_width_fills import CurrentFills, Exit, Skip, Trade


def _cached(build):
    """``build(chunk, i)`` with one snapshot per event of the CURRENT chunk: a new chunk object drops the cache and is held in its place."""
    held, snaps = None, {}

    def cached(chunk, i):
        nonlocal held, snaps
        if chunk is not held:
            held, snaps = chunk, {}
        if i not in snaps:
            snaps[i] = build(chunk, i)
        return snaps[i]
    return cached


def _wire(module, fills):
    """Route ``module``'s entry, exit marks and row labels through ``fills``. The last resolved trade, the skip reasons and the
    recorded exits live in this closure, so they belong to this one copy."""
    state = {"trade": None, "skipped": {}, "exits": {}}
    orig_resolve, orig_events, orig_score = module.resolve_priced, module.score_events, module.score

    def resolve_priced(chunk, i, name, pred, fill):
        out = orig_resolve(chunk, i, name, pred, fill)
        if out[1] is not None:
            state["trade"] = Trade(chunk, i, str(chunk["event_id"][i]), name, tuple(out[1]))
        return out

    def entry(priced):
        trade = state["trade"]
        got = fills.entry(trade, priced)
        if isinstance(got, Skip):
            state["skipped"][trade.event_id] = got.reason
            return None
        return got

    def _exit_value(chunk, traded, alpha, mid=False):
        trade = state["trade"]
        got = fills.exit_value(trade, traded, alpha, mid=mid)
        if not isinstance(got, Exit):
            return got
        if not set(got.flags) <= set(fills.exit_flags):
            raise ValueError(f"{trade.event_id!r}: exit flags {sorted(got.flags)} are not declared in exit_flags")
        if got.unquoted and got.value != 0.0:
            raise ValueError(f"{trade.event_id!r}: an unquoted exit must have value 0.0, not {got.value!r}")
        state["exits"][trade.event_id] = got
        return got.value

    def score_events(chunk, events, sims, name, *, alpha=0.5):
        state["skipped"] = {}
        rows = orig_events(chunk, events, sims, name, alpha=alpha)
        for eid, reason in state["skipped"].items():
            rows.loc[(rows["event_id"] == eid) & (rows["reason"] == "entry_quote_unusable"), "reason"] = reason
        return rows

    def score(chunk, events, decisions, name, alpha, *, drop_unquoted=False, mid_marks=False, repriced=False):
        state["exits"] = {}
        rows = orig_score(chunk, events, decisions, name, alpha, drop_unquoted=drop_unquoted, mid_marks=mid_marks,
                          repriced=repriced)
        if not state["exits"] and not fills.exit_flags:
            return rows
        got = [state["exits"].get(eid) for eid in rows["event_id"]]
        unq = np.array([g is not None and g.unquoted for g in got], dtype=bool)
        rows["rung"] = rows["rung"].mask(unq, EXIT_UNQUOTED)
        if drop_unquoted:
            rows = rows.loc[~unq].reset_index(drop=True)
            got = [g for g, u in zip(got, unq) if not u]
        for flag in fills.exit_flags:
            rows[flag] = np.array([g is not None and flag in g.flags for g in got], dtype=bool)
        return rows

    module.resolve_priced, module.entry, module._exit_value = resolve_priced, entry, _exit_value
    module.score_events, module.score = score_events, score


def load(fills=None):
    """A private instance of size_pnl_shortvol on the chain-index pricer; the real module is untouched."""
    fills = CurrentFills() if fills is None else fills
    spec = importlib.util.find_spec(sv.__name__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.Refusal = sv.Refusal  # callers catch sv.Refusal; the copy raises the same class
    module.price_structure = price_structure_indexed
    module.chain_snapshot = _cached(module.chain_snapshot)
    module.fills = fills
    _wire(module, fills)  # after the patches above: resolve_priced and score look them up as globals
    return module


PRICING = load()
