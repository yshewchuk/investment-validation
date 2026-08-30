"""Win-rate recalibration — make the shipped probability honest.

The model layer's ``win_model`` is not a model output; it is the share of payoff
draws that clear the real entry cost. When the payoff bridge sits too high —
which happens whenever the line fitted on trades closed before the decision was
priced in a stronger regime than the event being scored — every event looks more
winnable than it is, and the forecast loses to a constant predictor. That is a
level error, and it is correctable without improving any model.

The correction is a **monotone recalibration map**: fit ``raw_win -> realized
outcome`` on events that had already closed by the decision date, then apply it
to the raw probability. Monotone, so the ranking the models worked for is
preserved; fitted causally, so nothing after the decision leaks in. Recalibration
removes level error. It cannot manufacture discrimination — if the raw forecast
carries no signal, the calibrated one lands on the base rate, which is the honest
answer, and any movement above the base rate becomes a clean measure of the
models' signal.

The map is fitted from a **calibration pairs table**: ``(raw_win, outcome)`` for
a sample of historical events, each scored as of its own decision date. Building
that table is expensive (every pair costs one full score), so it is produced once
by :func:`build_pairs` and cached; fitting a map from it is then cheap and is
cached per ``(strategy, alpha, cutoff)``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from engine import paths

__all__ = [
    "RecalibrationMap",
    "PAIRS_PATH",
    "load_pairs",
    "fit_recalibration",
    "build_pairs",
]

#: The cached pairs table. Derived data, rebuildable by :func:`build_pairs`.
PAIRS_PATH = paths.FEATURES / "recalibration_pairs.parquet"

#: Below this many closed pairs a fitted map is noise shaped like a function.
MIN_PAIRS = 120


@dataclass(frozen=True)
class RecalibrationMap:
    """A fitted ``raw_win -> calibrated_win`` monotone map for one strategy+alpha."""

    strategy: str
    alpha: float
    n: int
    fitted_through: pd.Timestamp
    #: Parallel arrays defining the fitted step function (isotonic output).
    x_thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    y_thresholds: np.ndarray = field(default_factory=lambda: np.array([]))
    base_rate: float = float("nan")

    def transform(self, raw_win) -> np.ndarray:
        """Calibrated probability, linearly interpolated, clipped to [0, 1]."""
        raw = np.asarray(raw_win, dtype=float)
        if self.x_thresholds.size == 0:
            return raw
        cal = np.interp(raw, self.x_thresholds, self.y_thresholds)
        return np.clip(cal, 0.0, 1.0)

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "alpha": self.alpha,
            "n": self.n,
            "fitted_through": (
                str(self.fitted_through.date()) if self.fitted_through is not None else None
            ),
            "base_rate": round(self.base_rate, 4),
            "n_thresholds": int(self.x_thresholds.size),
        }


def load_pairs(path: Path | None = None) -> pd.DataFrame:
    """The cached ``(raw_win, outcome)`` table, or an empty frame if absent."""
    p = Path(path) if path is not None else PAIRS_PATH
    if not p.exists():
        return pd.DataFrame(
            columns=[
                "strategy", "fill_alpha", "event_id", "ticker", "event_date",
                "exit_date", "raw_win", "outcome",
            ]
        )
    return pd.read_parquet(p)


def fit_recalibration(
    strategy: str,
    alpha: float,
    *,
    before,
    pairs: pd.DataFrame | None = None,
    min_pairs: int = MIN_PAIRS,
) -> RecalibrationMap | None:
    """Fit the causal recalibration map for events scored before ``before``.

    Only pairs whose event had *closed* by the cutoff are eligible — the same
    discipline the payoff map applies — so a map used to score an event never
    knows that event's outcome, nor any contemporary's.
    """
    if pairs is None:
        pairs = load_pairs()
    if pairs.empty:
        return None

    stamp = pd.Timestamp(before).normalize() if before is not None else None
    rows = pairs[
        (pairs["strategy"] == strategy)
        & np.isclose(pairs["fill_alpha"].astype(float), float(alpha))
    ]
    if stamp is not None:
        rows = rows[pd.to_datetime(rows["exit_date"]) < stamp]
    rows = rows.dropna(subset=["raw_win", "outcome"])

    if len(rows) < min_pairs:
        return None

    from sklearn.isotonic import IsotonicRegression

    x = rows["raw_win"].to_numpy(dtype=float)
    y = rows["outcome"].to_numpy(dtype=float)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(x, y)

    return RecalibrationMap(
        strategy=strategy,
        alpha=float(alpha),
        n=int(len(rows)),
        fitted_through=stamp,
        x_thresholds=np.asarray(iso.X_thresholds_, dtype=float),
        y_thresholds=np.asarray(iso.y_thresholds_, dtype=float),
        base_rate=float(y.mean()),
    )


# --------------------------------------------------------------------------
# building the pairs table (the expensive, one-time part)
# --------------------------------------------------------------------------


def build_pairs(
    strategies=("STR-THRU", "STR-RUNUP"),
    *,
    alpha: float = 0.5,
    n_per_strategy: int = 1000,
    seed: int = 20260829,
    scorer=None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Score a year-stratified sample of realized events; land (raw_win, outcome).

    Each pair costs one full ``score()`` call as of the event's own decision
    date, so this is a background job, not a per-run cost. The result is cached
    at :data:`PAIRS_PATH` and merged with any pairs already there (idempotent on
    ``event_id + strategy``).
    """
    from engine.fills import FillModel
    from engine.replay import load_chain_index
    from engine.score import ScoreRequest, Scorer

    engine = scorer or Scorer()
    trades = engine.trades
    cols = [
        "event_id", "strategy", "fill_alpha", "ret", "ticker",
        "event_date", "entry_date", "exit_date",
    ]
    events_index = trades[[c for c in cols if c in trades.columns]].copy()
    events_index["event_date"] = pd.to_datetime(events_index["event_date"])

    out_rows: list[dict] = []
    for strategy in strategies:
        realized = events_index[
            (events_index["strategy"] == strategy)
            & np.isclose(events_index["fill_alpha"].astype(float), alpha)
        ].dropna(subset=["ret"])
        if realized.empty:
            continue

        # Year-stratified so the map spans regimes, not just the strong ones.
        realized = realized.assign(year=pd.to_datetime(realized["event_date"]).dt.year)
        n_years = int(realized["year"].nunique())
        per_year = max(1, n_per_strategy // max(1, n_years))
        picked = realized.groupby("year", group_keys=False).sample(
            n=per_year, random_state=seed
        )
        picked = picked.reset_index(drop=True)

        keys = set()
        for row in picked.itertuples(index=False):
            keys.add((str(row.ticker), pd.Timestamp(row.entry_date)))
            keys.add((str(row.ticker), pd.Timestamp(row.exit_date)))
        index = load_chain_index(keys, progress_every=0)

        from engine.data import store

        sessions = store.read_table(
            "earnings_events", columns=["event_id", "session"]
        ).set_index("event_id")

        started = time.time()
        done = 0
        for row in picked.itertuples(index=False):
            event_id = str(row.event_id)
            session = (
                str(sessions.loc[event_id, "session"])
                if event_id in sessions.index
                else None
            )
            try:
                result = engine.score(
                    ScoreRequest(
                        ticker=str(row.ticker),
                        strategy=strategy,
                        event_date=pd.Timestamp(row.event_date),
                        session=session,
                        fill=FillModel(alpha),
                    ),
                    chain_index=index,
                )
            except Exception:  # noqa: BLE001 - one bad event must not end the build
                continue
            if result.win_model is None:
                continue
            out_rows.append(
                {
                    "strategy": strategy,
                    "fill_alpha": alpha,
                    "event_id": event_id,
                    "ticker": str(row.ticker),
                    "event_date": pd.Timestamp(row.event_date),
                    "exit_date": pd.Timestamp(row.exit_date),
                    "raw_win": float(result.win_model),
                    "outcome": float(row.ret > 0),
                }
            )
            done += 1
            if done % 100 == 0:
                print(
                    f"  [recalib] {strategy}: {done}/{len(picked)} pairs, "
                    f"{time.time()-started:.0f}s",
                    flush=True,
                )

    built = pd.DataFrame(out_rows)
    if built.empty:
        return built

    p = Path(path) if path is not None else PAIRS_PATH
    paths.assert_writable(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_pairs(p)
    if not existing.empty:
        keep = existing[
            ~existing.set_index(["event_id", "strategy"]).index.isin(
                built.set_index(["event_id", "strategy"]).index
            )
        ]
        built = pd.concat([keep, built], ignore_index=True)
    built.to_parquet(p, index=False)
    print(f"  [recalib] wrote {len(built):,} pairs -> {p}", flush=True)
    return built
