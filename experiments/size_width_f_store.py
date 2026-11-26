"""EXP-189 W7c3b: one F stage's raw report inputs: legs_ and taken_ monthly parts (schema: experiments/README.md); only run stages open a Store."""
import sys
import pandas as pd

from experiments import size_width_f_report_inputs as ri, size_width_f_rows
from experiments.size_pnl_ladder import _rss_mib
from experiments.size_width_fills import Trade

_KEYS = ["point", "event_id", "structure", "k", "side", "qty", "ticker", "obs_date", "expiry", "right", "strike"]
_TABLE = ["bid", "ask", "eff_bid", "eff_ask", "V", "status", "source", "vol", "smvVol", "rate", "c_buy", "c_sell", "fill_buy",
          "fill_sell", "repaired", "open_interest", "bid_size", "ask_size", "chain_kind", "assetType", "spot"]


def open_store(out, stage, pricing): return Store(out, stage, pricing.fills.quotes) if size_width_f_rows.is_f(pricing) else None


class Store:
    def __init__(self, out, stage, quotes):
        self.out, self.stage, self.quotes, self.keys, self.taken_buf, self.parts, self.n = out, stage, quotes, [], [], {}, 0
        self.flags = [pd.DataFrame(columns=["arm", "structure", "year", "metric", "count"])]

    def print_entry_coverage(self, chunks):
        for done, chunk in enumerate(chunks, 1): self.quotes.table(chunk); print(f"[f_store] entry month={done} rss={_rss_mib():.0f}MiB", file=sys.stderr)
        counts = ri.merge_counts(self.quotes.coverage or [pd.DataFrame(columns=ri.POINT_COLS)])
        print("[f_store] entry coverage\n" + counts[counts["point"] == "entry"].to_csv(index=False), file=sys.stderr, end="")

    def candidates(self, sweep, chunk, events, name, k):  # entry legs of each grid-k candidate whose entry F priced
        self._record(sweep, chunk, events, [(e, name, k) for e in events["event_id"].astype(str)], ("entry",))

    def taken(self, sweep, chunk, events, rows, arm):  # entry and exit legs of each taken row; rows labelled with their arm
        self._record(sweep, chunk, events, zip(rows["event_id"].astype(str), rows["structure"], rows["k"]), ("entry", "exit"))
        label = str(arm) if getattr(sweep, "phi", None) is None else (
            f"fallback phi={sweep.phi} factors=" + ",".join(f"{n}={v}" for n, v in sorted(sweep.factors.items())))
        self.taken_buf.append(rows.assign(arm=label))

    def _record(self, sweep, chunk, events, triples, points):  # buffer positive-quantity legs per (event, structure, k); entry-only legs must be F-priced
        pos, pred = {str(e): i for i, e in enumerate(chunk["event_id"])}, dict(zip(events["event_id"].astype(str), events["pred_move"]))
        for eid, name, k, i in ((e, n, k, pos[e]) for e, n, k in triples):
            legs = sweep.pricing.resolve(chunk, i, name, pred[eid])[0]
            if legs is None or (points == ("entry",) and not sweep.pricing.fills.was_priced(Trade(chunk, i, eid, name, tuple(legs)))): continue
            self.keys += [(p, eid, name, float(k), leg["side"], float(leg["qty"]), *self.quotes.leg_key(chunk, i, leg["idx"], exit=p == "exit"))
                          for p in points for leg in legs if leg["qty"] > 0]

    def flush(self, chunk):
        if not self.keys: return
        self.parts[f"legs_{self.n:04d}"] = ri.publish(self.out, self.stage, f"legs_{self.n:04d}", self._legs(chunk, pd.DataFrame(self.keys, columns=_KEYS)))
        if self.taken_buf:
            taken = pd.concat(self.taken_buf, ignore_index=True)
            self.parts[f"taken_{self.n:04d}"] = ri.publish(self.out, self.stage, f"taken_{self.n:04d}", taken)
            self.flags.append(ri.exit_flag_counts(taken))
        self.keys, self.taken_buf, self.n = [], [], self.n + 1

    def _legs(self, chunk, keys):
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
        return legs.assign(year=pd.to_datetime(legs["obs_date"]).dt.year, mid=(legs["bid"] + legs["ask"]) / 2)

    def close(self, inputs_hash, scope):
        coverage = ri.merge_counts(self.quotes.coverage or [pd.DataFrame(columns=ri.POINT_COLS)])
        ri.write(self.out, self.stage, inputs_hash, scope, {"coverage": coverage, "exit_flags": ri.merge_counts(self.flags)}, parts=self.parts)
