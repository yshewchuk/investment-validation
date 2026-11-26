"""EXP-189 A1 W7c3 part b: the F report inputs of one run stage (design A1.10; #3010: no section 8 statistic at run time).
``legs_NNNN``: one row per positive-quantity leg (event_id, structure, k, point, right, strike, side; unlisted exit: nulls); entry rows are the
grid-k candidates F priced, plus taken legs. ``taken_NNNN``: the taken rows plus ``arm``. No legs or taken rows of the anchor or fake draws are stored, only their A1.4 counts: inner table
``anchor_c_nontrades`` (columns arm, draws, structure, k, year, metric, reason, count; ``draws`` = sweeps summed; metric ``candidates`` is the
denominator, metric ``nontrade`` is per decline reason, all reasons, by event year over the inner window). No C rows when C is omitted (the inner
summary's c_status says why). Only run stages open a Store."""
import sys
import time

import pandas as pd

from experiments import size_width_f_counts as fcounts, size_width_f_report_inputs as ri, size_width_f_rows
from experiments.size_width_sweep import WINDOWS
from experiments.size_pnl_ladder import _rss_mib
from experiments.size_width_fills import Trade

_KEYS = ["point", "event_id", "structure", "k", "side", "qty", "ticker", "obs_date", "expiry", "right", "strike"]
_TABLE = ["bid", "ask", "eff_bid", "eff_ask", "V", "status", "source", "vol", "smvVol", "rate", "c_buy", "c_sell", "fill_buy",
          "fill_sell", "repaired", "open_interest", "bid_size", "ask_size", "chain_kind", "assetType", "spot"]
_LEG = ["point", "event_id", "structure", "k", "right", "strike", "side", "qty", "ticker", "obs_date", "expiry", "year", "mid", *_TABLE]
_NONTRADE = ["arm", "structure", "k", "year", "metric", "reason", "count"]
_NONTRADE_OUT = ["arm", "draws", "structure", "k", "year", "metric", "reason", "count"]


def open_store(out, stage, pricing):
    """A Store for an F run stage; None for current pricing, which records nothing and calls nothing new."""
    return Store(out, stage, pricing.fills.quotes) if size_width_f_rows.is_f(pricing) else None


class Store:
    def __init__(self, out, stage, quotes):
        self.out, self.stage, self.quotes, self.keys, self.taken_buf, self.parts, self.flags, self.n = out, stage, quotes, [], [], {}, [], 0
        self.arm_counts = {}  # arm -> one count frame per sweep counted (A1.4)

    def _resolved(self, sweep, chunk, events, triples):
        """(position, event_id, structure, k, resolved legs) for each (event_id, structure, k) triple of the chunk."""
        pos, pred = {str(e): i for i, e in enumerate(chunk["event_id"])}, dict(zip(events["event_id"].astype(str), events["pred_move"]))
        for eid, name, k in triples:
            yield pos[eid], eid, name, k, sweep.pricing.resolve(chunk, pos[eid], name, pred[eid])[0]

    def print_entry_coverage(self, chunks):
        """Builds each distinct consecutive chunk's entry table (cached for the sweep), then prints the merged entry coverage."""
        last, done, started = None, 0, time.monotonic()
        for chunk in chunks:
            if chunk is not last:
                self.quotes.table(chunk)
                done += 1
                print(f"[f_store] entry chunk={done} elapsed={time.monotonic() - started:.1f}s rss={_rss_mib():.0f}MiB", file=sys.stderr)
            last = chunk
        counts = ri.merge_counts(self.quotes.coverage) if self.quotes.coverage else pd.DataFrame(columns=ri.POINT_COLS)
        print("[f_store] entry coverage\n" + counts[counts["point"] == "entry"].to_csv(index=False), file=sys.stderr, end="")

    def candidates(self, sweep, chunk, events, name, k):
        """Entry legs of each grid-k candidate (``events`` already scaled to k) whose entry F priced."""
        for i, eid, _, _, legs in self._resolved(sweep, chunk, events, [(e, name, k) for e in events["event_id"].astype(str)]):
            if legs is not None and sweep.pricing.fills.was_priced(Trade(chunk, i, eid, name, tuple(legs))):
                self._buffer(chunk, i, eid, name, k, legs, ("entry",))

    def taken(self, sweep, chunk, events, rows, arm):
        """Entry and exit legs of each taken row, plus the rows labelled with their arm (fallback: phi and factors)."""
        for i, eid, name, k, legs in self._resolved(sweep, chunk, events, zip(rows["event_id"].astype(str), rows["structure"], rows["k"])):
            self._buffer(chunk, i, eid, name, k, legs, ("entry", "exit"))
        label = str(arm) if getattr(sweep, "phi", None) is None else (
            f"fallback phi={sweep.phi} factors=" + ",".join(f"{n}={v}" for n, v in sorted(sweep.factors.items())))
        self.taken_buf.append(rows.assign(arm=label))

    def count_sweep(self, arm, sweep):
        """A1.4 non-trade counts of one anchor or C sweep: candidates and decline reasons per name and grid k, per window (E is not counted)."""
        rows = []
        for name in sweep.names:
            for k in sweep.ks:
                for window in (WINDOWS["2023H2"], WINDOWS["2024"]):
                    got, group = fcounts.nontrades(sweep, name, k, window), {"arm": arm, "structure": name, "k": float(k), "year": window[0].year}
                    rows.append({**group, "metric": "candidates", "reason": "", "count": got["candidates"]})
                    rows += [{**group, "metric": "nontrade", "reason": reason, "count": n} for reason, n in got["reasons"].items()]
        self.arm_counts.setdefault(arm, []).append(pd.DataFrame(rows, columns=_NONTRADE))

    def _buffer(self, chunk, i, eid, name, k, legs, points):
        """Buffers the positive-quantity legs of one row at each of ``points`` (no legs: nothing buffered)."""
        self.keys += [(p, eid, name, float(k), leg["side"], float(leg["qty"]), *self.quotes.leg_key(chunk, i, leg["idx"], exit=p == "exit"))
                      for p in points for leg in legs or [] if leg["qty"] > 0]

    def flush(self, chunk):
        """Publishes the buffered legs and taken rows as one monthly part each, then clears the buffers."""
        if not self.keys: return
        self.parts[f"legs_{self.n:04d}"] = ri.publish(self.out, self.stage, f"legs_{self.n:04d}", self._legs(chunk, pd.DataFrame(self.keys, columns=_KEYS)))
        if self.taken_buf:
            taken = pd.concat(self.taken_buf, ignore_index=True)
            self.parts[f"taken_{self.n:04d}"] = ri.publish(self.out, self.stage, f"taken_{self.n:04d}", taken)
            self.flags.append(ri.exit_flag_counts(taken))
        self.keys, self.taken_buf, self.n = [], [], self.n + 1

    def _legs(self, chunk, keys):
        """One leg row per buffered key: the quote table's fields at its contract (an unlisted contract: unpriceable, nulls)."""
        parts = []
        for point, part in keys.drop_duplicates(["event_id", "structure", "k", "point", "right", "strike", "side"]).groupby("point"):
            table = self.quotes.table(chunk, exit=point == "exit")
            index = pd.MultiIndex.from_arrays([part[c].to_numpy() for c in ("ticker", "obs_date", "expiry", "right", "strike")])
            got = table.reindex(index)[_TABLE].reset_index(drop=True)
            got["status"] = got["status"].where(table.index.get_indexer(index) >= 0, "unpriceable")
            got = got.assign(repaired=got["repaired"].astype("boolean"),
                             **{c: got[c].astype(object).where(got[c].notna(), None) for c in ("source", "chain_kind", "assetType", "status")})
            parts.append(pd.concat([part.reset_index(drop=True), got], axis=1))
        legs = pd.concat(parts, ignore_index=True)
        return legs.assign(year=pd.to_datetime(legs["obs_date"]).dt.year, mid=(legs["bid"] + legs["ask"]) / 2)[_LEG]

    def close(self, inputs_hash, scope):
        """Writes the stage's receipt with the coverage, exit-flag and (when a sweep was counted) A1.4 non-trade tables; refused while any leg or taken row is unflushed."""
        if self.keys or self.taken_buf: raise ValueError("close needs a flush first: buffered legs or taken rows are unpublished")
        flags = ri.merge_counts(self.flags) if self.flags else pd.DataFrame(columns=["arm", "structure", "year", "metric", "count"])
        coverage = ri.merge_counts(self.quotes.coverage) if self.quotes.coverage else pd.DataFrame(columns=ri.POINT_COLS)
        tables = {"coverage": coverage, "exit_flags": flags}
        if self.arm_counts:  # each arm's sweeps summed; draws = its sweeps (anchor: 1, C: one per draw)
            tables["anchor_c_nontrades"] = pd.concat([ri.merge_counts(self.arm_counts[a]).assign(draws=len(self.arm_counts[a])) for a in sorted(self.arm_counts)],
                                                     ignore_index=True)[_NONTRADE_OUT]
        ri.write(self.out, self.stage, inputs_hash, scope, tables, parts=self.parts)
