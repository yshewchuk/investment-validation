#!/usr/bin/env python3
"""Score held-out events, then grade the scores against what actually happened.

    python3 checks/phase1_calibration.py
    python3 checks/phase1_calibration.py --from-year 2023 --sample 600

Guide acceptance test 4. The models were walk-forward evaluated during training,
but that grades the *models*. This grades the **scorer** — the model prediction
pushed through a payoff map against a real entry cost, which is the number a
human would actually act on and a different quantity from the model's own MAE.

Method: take engine-replayed trades from the held-out years, re-score each
event as of its historical decision date, and compare the scorer's predicted
win rate and expected P&L against the return the trade actually realized at the
same fill alpha.

Everything is causal by construction — the scorer's payoff maps and analog sets
are fitted only on trades closed before each decision date — so this is a real
out-of-sample calibration rather than a fit statistic wearing a different name.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import calibrate as cal  # noqa: E402
from engine.data import store  # noqa: E402
from engine.fills import FillModel  # noqa: E402
from engine.replay import load_chain_index  # noqa: E402
from engine.score import ScoreRequest, Scorer  # noqa: E402

#: Years held out of the calibration sample's own evidence. The scorer's payoff
#: maps and analogs are causal anyway, but drawing the sample from later years
#: keeps this comparable to the models' own walk-forward split.
DEFAULT_FROM_YEAR = 2023


def score_sample(
    strategy: str,
    *,
    from_year: int = DEFAULT_FROM_YEAR,
    sample: int = 500,
    alpha: float = 0.5,
    scorer: Scorer | None = None,
    seed: int = 20260829,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-score a sample of realized trades and attach what they actually did."""
    engine = scorer or Scorer()
    trades = engine.trades
    rows = trades[
        (trades["strategy"] == strategy)
        & np.isclose(trades["fill_alpha"].astype(float), alpha)
        & (pd.to_datetime(trades["event_date"]).dt.year >= from_year)
    ]
    if rows.empty:
        return pd.DataFrame()

    if len(rows) > sample:
        rng = np.random.default_rng(seed)
        rows = rows.iloc[rng.choice(len(rows), size=sample, replace=False)]
    rows = rows.sort_values("event_date").reset_index(drop=True)

    events = store.read_table(
        "earnings_events", columns=["event_id", "ticker", "event_date", "session"]
    ).set_index("event_id")

    keys = set()
    for row in rows.itertuples(index=False):
        keys.add((str(row.ticker), pd.Timestamp(row.entry_date)))
        keys.add((str(row.ticker), pd.Timestamp(row.exit_date)))
    index = load_chain_index(keys, progress_every=0)

    out = []
    started = time.time()
    for i, row in enumerate(rows.itertuples(index=False)):
        event_id = str(row.event_id)
        session = str(events.loc[event_id, "session"]) if event_id in events.index else None
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
        except Exception as exc:  # noqa: BLE001 - one bad event must not end the run
            out.append({"event_id": event_id, "error": str(exc)})
            continue
        out.append(
            {
                "event_id": event_id,
                "ticker": str(row.ticker),
                "event_date": pd.Timestamp(row.event_date),
                "year": pd.Timestamp(row.event_date).year,
                "win_model": result.win_model,
                "exp_pnl_model": result.exp_pnl_model,
                "win_analog": result.win_analog,
                "exp_pnl_analog": result.exp_pnl_analog,
                "gate_score": result.gate_score,
                "gate_pass": result.gate_pass,
                "n_analogs": result.n_analogs,
                "flags": ",".join(result.flags),
                "ret": float(row.ret),
            }
        )
        if verbose and i and i % 100 == 0:
            print(f"  [calib] {i}/{len(rows)} scored, {time.time()-started:.0f}s", flush=True)
    return pd.DataFrame(out)


def run(
    strategies=("STR-THRU", "STR-RUNUP"),
    *,
    from_year: int = DEFAULT_FROM_YEAR,
    sample: int = 500,
    alpha: float = 0.5,
    scorer: Scorer | None = None,
    verbose: bool = True,
) -> dict:
    engine = scorer or Scorer()
    reports = {}
    for strategy in strategies:
        scored = score_sample(
            strategy, from_year=from_year, sample=sample, alpha=alpha,
            scorer=engine, verbose=verbose,
        )
        if scored.empty or "win_model" not in scored.columns:
            reports[strategy] = {"skipped": "no scored events"}
            continue
        usable = scored.dropna(subset=["win_model", "ret"])
        if usable.empty:
            reports[strategy] = {
                "skipped": "no event produced a model-layer score",
                "n_attempted": int(len(scored)),
            }
            continue

        model = cal.calibrate(usable, label=f"{strategy} model layer")
        doc = {"model_layer": model.as_dict(), "n_scored": int(len(scored))}

        analog_usable = scored.dropna(subset=["win_analog", "ret"])
        if len(analog_usable) >= 50:
            analog = cal.calibrate(
                analog_usable,
                label=f"{strategy} analog layer",
                prob_column="win_analog",
                pnl_column="exp_pnl_analog",
            )
            doc["analog_layer"] = analog.as_dict()
        reports[strategy] = doc

        if verbose:
            print(
                f"\n{strategy}: n={model.n} base={model.base_rate:.3f} "
                f"brier={model.brier:.4f} (base {model.brier_base:.4f}) "
                f"skill={model.brier_skill:+.4f} "
                f"reliability monotonicity={model.reliability_monotonicity:+.3f}",
                flush=True,
            )
            print(model.reliability.to_string(index=False), flush=True)
    return reports


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", action="append", default=None)
    ap.add_argument("--from-year", type=int, default=DEFAULT_FROM_YEAR)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    strategies = args.strategy or ["STR-THRU", "STR-RUNUP"]
    print("Phase 1 calibration\n", flush=True)
    reports = run(
        strategies, from_year=args.from_year, sample=args.sample, alpha=args.alpha
    )
    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
