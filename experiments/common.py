"""Shared implementation pieces for the Sep-2026 validation experiments.

The experiment plan (``guides/experiment_plan_sep2026.md`` §6) names four
pieces that EXP-102/105/107 all need and none of which existed when the plan
was written:

* the registered champion gate as a walk-forward :class:`engine.evaluate.Gate`
  with ``predict_proba`` — the registered gates predict a *return*, while the
  calibration block needs P(win);
* ``tail_shock`` — mandatory for CAL-P's short leg;
* ``repricer`` — the ±1-day slippage and stale-date stresses;
* ``spy_daily`` — regime replays and the IV-regime split.

They are built once here so all three experiments use one implementation — the
same reasoning that keeps ``engine.replay`` the only pricing path. Nothing in
this module prices a trade itself: pricing always goes through
``engine.replay.replay_one`` and the stored Tier-2 rows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.calendar import TradingCalendar, trading_calendar
from engine.evaluate import Gate
from engine.models.registry import load_registry
from engine.models.training import gate as gate_mod
from engine.models.training.common import SEED
from engine.structures import BUY, STRUCTURES, Structure

__all__ = [
    "load_engine_trades",
    "load_spy_daily",
    "gate_dataset",
    "make_registered_gate",
    "make_trained_gate",
    "make_repricer",
    "calp_tail_shock",
    "abs_move_tail_shock",
    "MIN_FIT_ROWS",
]

#: A fold with fewer complete rows than this does not fit a gate model — the
#: same floor the champion's own walk-forward used (``min_train_rows``).
MIN_FIT_ROWS = 500


# --------------------------------------------------------------------------
# trade sets
# --------------------------------------------------------------------------


def load_engine_trades(strategy: str) -> pd.DataFrame:
    """The engine-replayed trade rows for one strategy, every fill alpha.

    Read from the Tier-2 ``trades`` table (not re-priced): the table is the
    artifact Phase 0 produced through ``engine.replay`` over the unselected
    event universe, and reading it keeps the experiments on exactly the priced
    set the plan's numbers quote. The ``session`` column (BMO/AMC) is not in
    the trades schema — the repricer and the chain snapshots need it, so it is
    joined from ``earnings_events`` here, once, at load.
    """
    from engine.data import store

    trades = store.read_table("trades")
    rows = trades[
        (trades["strategy"] == strategy)
        & (trades["provenance"].astype(str) == "engine.replay")
    ].reset_index(drop=True)
    for col in ("event_date", "entry_date", "exit_date"):
        rows[col] = pd.to_datetime(rows[col])

    events = store.read_table(
        "earnings_events", columns=["event_id", "ticker", "event_date", "session"])
    events["event_date"] = pd.to_datetime(events["event_date"])
    if "session" not in rows.columns or rows["session"].isna().all():
        rows = rows.drop(columns=[c for c in ("session",) if c in rows.columns])
        rows = rows.merge(
            events[["event_id", "session"]].drop_duplicates("event_id"),
            on="event_id", how="left")
        n_missing = int(rows["session"].isna().sum())
        if n_missing:
            # Fall back to (ticker, event_date) for event_ids the events table
            # spells differently.
            fallback = rows[rows["session"].isna()].merge(
                events[["ticker", "event_date", "session"]].drop_duplicates(
                    ["ticker", "event_date"]),
                on=["ticker", "event_date"], how="left", suffixes=("", "_fb"))
            rows.loc[rows["session"].isna(), "session"] = fallback["session_fb"].to_numpy()
            n_missing = int(rows["session"].isna().sum())
        print(f"  [common] {strategy}: session joined, {n_missing:,} rows without one",
              flush=True)
    return rows


def load_spy_daily(path: Path | None = None) -> pd.DataFrame:
    """The cached S&P daily series as ``date``/``close``.

    Same parse as ``engine.calendar`` and the panel builder (yfinance
    multi-header: three rows to skip). The stress stages need nothing else.
    """
    p = Path(path or paths.GSPC_DAILY)
    raw = pd.read_csv(p, skiprows=3, header=None)
    raw.columns = ["date", "adj", "close", "high", "low", "open", "volume"][: raw.shape[1]]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    out = raw[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
    out["close"] = out["close"].astype(float)
    return out


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------


def gate_dataset(strategy: str, trades: pd.DataFrame, cache_dir: Path | str) -> pd.DataFrame:
    """Feature frame for the gate at alpha=0.5, cached under ``cache_dir``.

    Delegates to the champion's own ``build_dataset`` so the features are the
    registry's feature list by construction, not a re-derivation. Building it
    walks the panel and the daily store once per strategy (~minutes), so the
    frame is cached as parquet keyed by strategy; delete the cache file to
    force a rebuild after a store change.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"gate_dataset_{strategy.lower().replace('-', '_')}.parquet"
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
        print(f"  [common] gate dataset {strategy}: {len(frame):,} rows from cache", flush=True)
        return frame

    started = time.time()
    frame = gate_mod.build_dataset(trades)
    frame.to_parquet(cache_path, index=False)
    print(
        f"  [common] gate dataset {strategy}: {len(frame):,} rows built in "
        f"{time.time() - started:.0f}s -> {cache_path.name}",
        flush=True,
    )
    return frame


class _RegisteredGateState:
    """Fold state behind :func:`make_registered_gate` and :func:`make_trained_gate`.

    ``fit(train)`` refits the champion model class on the fold's train rows —
    the harness hands it only years strictly before the year being traded —
    and fits the isotonic P(win) map on the same rows. ``select`` applies the
    threshold; ``predict_proba`` returns P(win) from the most recent fit.

    The threshold comes from one of two places and never from anywhere else. A
    strategy with a registered champion passes ``threshold=`` and the registry's
    stored number is applied unchanged — that is the decision rule as promoted.
    A strategy with no champion yet passes ``top_fraction=`` instead, and each
    fold chooses its own threshold as that quantile of its TRAINING-year
    predictions. A threshold picked on the year being gated is a rank the model
    could not have known, and it would make every walk-forward year look like a
    selection the gate actually made. ``engine.evaluate.walk_forward`` calls ``predict_proba`` BEFORE
    the same year's ``fit``, so each year's probabilities come from the
    previous fold's model (and the first gated year has none, hence NaN). That
    is leak-free — the model behind a probability never saw that year — and one
    fold stale, which the experiment report says rather than hiding.

    Rows without complete features cannot be scored — the live scorer has no
    features to feed either — so ``select`` returns False for them and
    ``predict_proba`` NaN; the count lands in ``stats`` for the report.
    """

    def __init__(self, name: str, features: Sequence[str], feat: pd.DataFrame,
                 threshold: float | None = None, seed: int = SEED,
                 top_fraction: float | None = None):
        if (threshold is None) == (top_fraction is None):
            raise ValueError(
                "pass exactly one of threshold= (a registered champion's stored "
                "rule) or top_fraction= (choose it per fold on training rows)")
        self.name = name
        self.features = tuple(features)
        self.feat = feat
        self.threshold = None if threshold is None else float(threshold)
        self.top_fraction = top_fraction
        self.seed = seed
        self.model_ = None
        self.iso_ = None
        self.base_rate_: float | None = None
        self.stats: list[dict] = []

    def _lookup(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ids = rows["event_id"].to_numpy()
        X = self.feat.reindex(ids)[list(self.features)].to_numpy(dtype=float)
        complete = np.isfinite(X).all(axis=1)
        return X, complete

    def fit(self, train: pd.DataFrame) -> None:
        from sklearn.isotonic import IsotonicRegression

        X, complete = self._lookup(train)
        y = train["ret"].to_numpy(dtype=float)
        ok = complete & np.isfinite(y)
        n_complete = int(ok.sum())
        self.stats.append({
            "train_rows": int(len(train)), "complete_rows": n_complete,
            "train_year_max": int(pd.to_datetime(train["event_date"]).dt.year.max()) if len(train) else None,
        })
        if n_complete < MIN_FIT_ROWS:
            self.model_ = None
            self.iso_ = None
            if self.top_fraction is not None:
                self.threshold = None
            return
        self.model_ = gate_mod.fit(X[ok], y[ok], seed=self.seed)
        pred = np.asarray(self.model_.predict(X[ok]), dtype=float)
        if self.top_fraction is not None:
            self.threshold = gate_mod.choose_threshold(pred, self.top_fraction)
            self.stats[-1]["threshold_chosen_on_train"] = self.threshold
        win = (y[ok] > 0).astype(float)
        self.base_rate_ = float(win.mean())
        try:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(pred, win)
            self.iso_ = iso
        except ValueError:
            # Degenerate predictions (a constant fit): no monotone map exists,
            # and the honest probability is the base rate — calibration_block
            # will say the probabilities are degenerate.
            self.iso_ = None

    def select(self, rows: pd.DataFrame) -> pd.Series:
        if self.model_ is None or self.threshold is None:
            return pd.Series(False, index=rows.index)
        X, complete = self._lookup(rows)
        pred = np.full(len(rows), np.nan)
        if complete.any():
            pred[complete] = np.asarray(self.model_.predict(X[complete]), dtype=float)
        mask = np.isfinite(pred) & (pred >= self.threshold)
        self.stats.append({
            "test_rows": int(len(rows)),
            "scored_rows": int(complete.sum()),
            "passed_rows": int(mask.sum()),
            "threshold": self.threshold,
        })
        return pd.Series(mask, index=rows.index)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        out = np.full(len(rows), np.nan)
        if self.model_ is None:
            return out
        X, complete = self._lookup(rows)
        if not complete.any():
            return out
        pred = np.asarray(self.model_.predict(X[complete]), dtype=float)
        if self.iso_ is not None:
            out[complete] = np.asarray(self.iso_.predict(pred), dtype=float)
        elif self.base_rate_ is not None:
            out[complete] = self.base_rate_
        return out


def make_registered_gate(strategy: str, dataset: pd.DataFrame) -> tuple[Gate, _RegisteredGateState]:
    """The registered champion gate for ``strategy`` as a walk-forward Gate.

    ``dataset`` is :func:`gate_dataset` output: one row per event at alpha=0.5
    with the registry's feature columns. The returned state object carries the
    per-fold diagnostics the report quotes (rows scored vs not, threshold).
    """
    registry = load_registry(missing_ok=False)
    entry = registry.champion("gate", strategy)
    if entry.threshold is None:
        raise ValueError(f"{entry.id}: champion gate carries no stored threshold")

    features = [c for c in entry.features if c in dataset.columns]
    missing = [c for c in entry.features if c not in dataset.columns]
    if missing:
        raise ValueError(f"{entry.id}: dataset is missing features {missing}")

    feat = dataset.set_index("event_id")[features]
    feat = feat[~feat.index.duplicated(keep="first")]
    state = _RegisteredGateState(entry.id, features, feat, entry.threshold, seed=entry.seed or SEED)
    gate = Gate(
        fit=state.fit,
        select=state.select,
        predict_proba=state.predict_proba,
        name=f"{entry.id}@{entry.threshold:.5f}",
    )
    return gate, state


def make_trained_gate(
    name: str,
    dataset: pd.DataFrame,
    features: Sequence[str],
    *,
    top_fraction: float = gate_mod.TOP_FRACTION,
    seed: int = SEED,
) -> tuple[Gate, _RegisteredGateState]:
    """A gate for a strategy that has no champion yet.

    Same model class, same features, same fold discipline as
    :func:`make_registered_gate` — the difference is that there is no stored
    threshold to apply, so each fold picks its own on its training predictions.
    Use this to ask whether a gate is worth promoting; use the registered one to
    re-validate a gate that already was.
    """
    missing = [c for c in features if c not in dataset.columns]
    if missing:
        raise ValueError(f"{name}: dataset is missing features {missing}")
    feat = dataset.set_index("event_id")[list(features)]
    feat = feat[~feat.index.duplicated(keep="first")]
    state = _RegisteredGateState(
        name, features, feat, seed=seed, top_fraction=top_fraction)
    gate = Gate(
        fit=state.fit,
        select=state.select,
        predict_proba=state.predict_proba,
        name=f"{name}@top{top_fraction:.0%}",
    )
    return gate, state


# --------------------------------------------------------------------------
# repricer (slippage + stale-date stress)
# --------------------------------------------------------------------------


def make_repricer(
    strategy: str,
    *,
    structure: Structure | None = None,
    calendar: TradingCalendar | None = None,
    alpha: float = 0.5,
) -> Callable[[pd.DataFrame, int], pd.DataFrame]:
    """Shift entry/exit by N trading days and re-price through engine.replay.

    The stress stages hand this the rows they want repriced; the repricer
    loads ONLY the chains the shifted dates need (never the full store — the
    box does not have the memory), re-runs ``replay_one`` per event, and drops
    events whose shifted chains are absent. The re-priced share is returned as
    the frame's ``coverage`` attr. A missing chain is never fabricated.

    Entry and exit shift together: that is both the slippage question ("what
    if we traded a day late") and the stale-date question (an event mis-dated
    by one day resolves entry and exit one trading day off).
    """
    from engine.replay import ChainIndex, load_chain_index, replay_one

    struct = structure or STRUCTURES[strategy]()
    cal = calendar or trading_calendar()

    def repricer(trades: pd.DataFrame, shift_days: int) -> pd.DataFrame:
        t = trades.reset_index(drop=True)
        plan_rows: list[dict | None] = []
        keys: set[tuple[str, pd.Timestamp]] = set()
        for row in t.itertuples(index=False):
            try:
                entry = cal.shift(pd.Timestamp(row.entry_date), int(shift_days))
                exit_ = cal.shift(pd.Timestamp(row.exit_date), int(shift_days))
            except KeyError:
                plan_rows.append(None)
                continue
            plan_rows.append({
                "event_id": row.event_id,
                "ticker": row.ticker,
                "event_date": pd.Timestamp(row.event_date),
                "session": row.session,
                "entry_date": entry,
                "exit_date": exit_,
            })
            keys.add((row.ticker, entry))
            keys.add((row.ticker, exit_))

        started = time.time()
        index = load_chain_index(keys, progress_every=0) if keys else ChainIndex({})
        print(
            f"  [repricer] {strategy} shift {int(shift_days):+d}d: "
            f"{len(index):,}/{len(keys):,} shifted chains loaded in {time.time() - started:.0f}s",
            flush=True,
        )

        out_rows: list[dict] = []
        for plan_row in plan_rows:
            if plan_row is None:
                continue
            priced, _reason = replay_one(struct, plan_row, index, alphas=(alpha,))
            out_rows.extend(priced)

        out = pd.DataFrame(out_rows) if out_rows else t.iloc[0:0].copy()
        coverage = float(len(out) / len(t)) if len(t) else float("nan")
        out.attrs["coverage"] = coverage
        return out

    return repricer


# --------------------------------------------------------------------------
# tail injection (CAL-P)
# --------------------------------------------------------------------------


def calp_tail_shock(trades: pd.DataFrame, worst_frac: float = 0.01) -> pd.DataFrame:
    """Double the realized move of the worst DOWN-move trades and re-price the exit.

    The ruin tail for CAL-P's short front put is a large fall, so the shock set
    is the most negative spot moves — not the worst returns: a trade can lose on
    an up move (time-value dynamics), and doubling an up move makes it better,
    which would let a "ruin" stress report a worst trade that improved. The
    return-tail losses are already in the base distribution; tail injection adds
    the doubled DOWN tail on top of it.
    """
    return _tail_shock(trades, worst_frac, signed=True)


def abs_move_tail_shock(trades: pd.DataFrame, worst_frac: float = 0.01) -> pd.DataFrame:
    """Double the realized move of the LARGEST-|move| trades and re-price the exit.

    The shock set for a structure that is short the event. A condor is hurt by a
    big move in EITHER direction — it settles worthless past either wing — so
    ranking by the signed move would shock only the down tail and leave the
    up-side blow-ups, half the ruin cases, untouched.
    """
    return _tail_shock(trades, worst_frac, signed=False)


def _tail_shock(trades: pd.DataFrame, worst_frac: float, *, signed: bool) -> pd.DataFrame:
    """Double the realized move of the worst ``worst_frac`` and re-price the exit.

    The harness does not know a structure's payoff, so this supplies the
    re-pricing. ``signed`` picks the ruin direction: the most negative moves for
    a structure that is long the downside risk, the largest absolute moves for
    one that is short the move in both directions.

    Re-pricing needs no chain reload: the stored ``legs`` blob carries both
    exit legs' quotes. Each shocked leg keeps the time value of its actual
    exit quote (mid minus intrinsic at the actual spot, floored at zero) and
    takes intrinsic at the SHOCKED spot — value can therefore never drop below
    the quoted time value, and the short leg's intrinsic grows without bound
    as the shocked spot falls. Entry cost is untouched (the shock is
    post-print). Rows whose blob cannot be parsed are left unchanged and
    counted in ``tail_shock_skipped``.
    """
    t = trades.copy().reset_index(drop=True)
    if t.empty or "legs" not in t.columns:
        t.attrs["tail_shock_applied"] = 0
        t.attrs["tail_shock_skipped"] = 0
        return t

    docs = []
    spot_entry = np.full(len(t), np.nan)
    spot_exit = np.full(len(t), np.nan)
    for i, blob in enumerate(t["legs"].to_numpy()):
        if not isinstance(blob, str):
            docs.append(None)
            continue
        try:
            doc = json.loads(blob)
        except ValueError:
            docs.append(None)
            continue
        docs.append(doc)
        try:
            spot_entry[i] = float(doc.get("spot_entry"))
            spot_exit[i] = float(doc.get("spot_exit"))
        except (TypeError, ValueError):
            pass

    move = np.where(spot_entry > 0, spot_exit / spot_entry - 1.0, np.nan)
    n_shock = max(1, int(round(len(t) * worst_frac)))
    eligible = np.where(np.isfinite(move))[0]
    if eligible.size == 0:
        t.attrs["tail_shock_applied"] = 0
        t.attrs["tail_shock_skipped"] = 0
        return t
    rank = move[eligible] if signed else -np.abs(move[eligible])
    order = eligible[np.argsort(rank)]
    worst_idx = order[:n_shock]

    applied = 0
    skipped = 0
    for i in worst_idx:
        doc = docs[i]
        exit_legs = (doc or {}).get("exit") or []
        if (
            doc is None
            or not exit_legs
            or not np.isfinite(move[i])
            or not np.isfinite(spot_entry[i])
            or spot_entry[i] <= 0
        ):
            skipped += 1
            continue
        shocked_spot = spot_entry[i] * (1.0 + 2.0 * move[i])
        exit_value = 0.0
        feasible = True
        for leg in exit_legs:
            try:
                strike = float(leg["strike"])
                bid = float(leg["bid"])
                ask = float(leg["ask"])
            except (KeyError, TypeError, ValueError):
                feasible = False
                break
            if not (np.isfinite(bid) and np.isfinite(ask) and ask >= bid >= 0):
                feasible = False
                break
            mid = 0.5 * (bid + ask)
            intrinsic_actual = max(strike - spot_exit[i], 0.0)
            time_value = max(mid - intrinsic_actual, 0.0)
            value = max(strike - shocked_spot, 0.0) + time_value
            # The stored exit legs carry the CLOSING side: sell-to-close
            # brings value in, buy-to-close pays it out.
            exit_value += value if leg.get("side") == "sell" else -value
        if not feasible:
            skipped += 1
            continue
        cost = float(t.loc[i, "entry_cost"])
        t.loc[i, "exit_value"] = exit_value
        t.loc[i, "pnl"] = exit_value - cost
        t.loc[i, "ret"] = (exit_value - cost) / cost if cost > 0 else np.nan
        applied += 1

    t.attrs["tail_shock_applied"] = applied
    t.attrs["tail_shock_skipped"] = skipped
    return t
