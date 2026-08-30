#!/usr/bin/env python3
"""Phase 1 acceptance tests.

    python3 checks/phase1_checks.py               # everything
    python3 checks/phase1_checks.py --list
    python3 checks/phase1_checks.py --only registry determinism
    python3 checks/phase1_checks.py --no-data     # only checks needing no store

The guide's six acceptance tests, plus the exit criteria, plus the constraints
the guide states as prose that are only real if something enforces them.

Unit-level behaviour lives in ``tests/`` and runs here as check 0. Everything
else needs the real store, the real artifacts, or the real chain cache — the
things a fixture cannot prove.

A phase without green checks is not done.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import phase1_calibration, phase1_replay  # noqa: E402
from engine.audit import LeakError  # noqa: E402
from engine.data import store  # noqa: E402
from engine.features import (  # noqa: E402
    LIVE_UNAVAILABLE,
    UnservableFeature,
    assert_live_available,
)
from engine.fills import MID  # noqa: E402
from engine.models.registry import RegistryError, load_registry  # noqa: E402
from engine.replay import load_chain_index  # noqa: E402
from engine.score import (  # noqa: E402
    ATM_TOLERANCE_PCT,
    DISABLED_STRATEGIES,
    FLAGS,
    ScoreRequest,
    Scorer,
    score_calendar,
)

#: The guide's exit criterion for a three-week calendar scoring run.
CALENDAR_BUDGET_S = 300

#: The calibration floor (decision record 2026-08-30): the shipped win rate must
#: not be over-confident. Honest recalibration lands the Brier skill near zero;
#: this tolerance absorbs sampling noise while still catching real anti-calibration
#: (the pre-recalibration scorer measured -0.12 and -0.20).
MIN_BRIER_SKILL = -0.05


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""
    elapsed_s: float = 0.0
    skipped: bool = False


REGISTRY: dict[str, dict] = {}
_SCORER: Scorer | None = None


def check(name: str, *, needs_data: bool = True, description: str = ""):
    def wrap(fn):
        REGISTRY[name] = {"fn": fn, "needs_data": needs_data, "description": description}
        return fn

    return wrap


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scorer() -> Scorer:
    """One Scorer for the whole run — loading it per check would dominate."""
    global _SCORER
    if _SCORER is None:
        _SCORER = Scorer()
    return _SCORER


def _engine_trades() -> pd.DataFrame:
    trades = store.read_table("trades")
    return trades[trades["provenance"].astype(str) == "engine.replay"]


# --------------------------------------------------------------------------
# 0. unit suite
# --------------------------------------------------------------------------


@check("unittests", needs_data=False, description="the pytest suite (pure logic)")
def check_unittests() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--no-header"],
        capture_output=True, text=True, cwd=ROOT,
    )
    tail = (result.stdout or result.stderr).strip().splitlines()[-1:]
    _require(result.returncode == 0, f"pytest failed: {' '.join(tail)}")
    return " ".join(tail)


# --------------------------------------------------------------------------
# 1. the trade set the whole phase rests on
# --------------------------------------------------------------------------


@check("trade_set", description="engine-replayed trades exist, at every fill alpha")
def check_trade_set() -> str:
    trades = _engine_trades()
    _require(
        len(trades) > 0,
        "no engine.replay trades in Tier 2 — run `python3 -m engine.build_trades`",
    )
    by_strategy = trades.groupby("strategy")["event_id"].nunique()
    alphas = sorted(trades["fill_alpha"].dropna().unique().tolist())
    _require(
        0.0 in alphas and 0.5 in alphas and 1.0 in alphas,
        f"worst/mid/best fills must all be present; found {alphas}",
    )
    for strategy in ("STR-THRU", "STR-RUNUP"):
        _require(
            by_strategy.get(strategy, 0) >= 1000,
            f"{strategy}: only {by_strategy.get(strategy, 0)} events replayed",
        )
    # Every event must be priced at every alpha, or an alpha sweep would be
    # comparing different samples rather than the same trades at different fills.
    per_event = trades.groupby(["strategy", "event_id"])["fill_alpha"].nunique()
    _require(
        per_event.nunique() == 1,
        f"events priced at differing numbers of alphas: {sorted(per_event.unique())}",
    )
    return (
        f"{len(trades):,} rows, {int(by_strategy.sum()):,} events, alphas {alphas}, "
        + ", ".join(f"{k}={v:,}" for k, v in by_strategy.items())
    )


@check("fill_monotonicity", description="better fills never produce worse returns")
def check_fill_monotonicity() -> str:
    """A structural property of the fill model, asserted on the real trade set.

    Every leg of every structure here is transacted in the direction that
    benefits from a tighter fill, so a trade's return must be non-decreasing in
    alpha. A violation means a sign error somewhere in the pricing path — the
    kind of bug that is invisible in aggregate and fatal in the verdict.
    """
    trades = _engine_trades()
    pivot = trades.pivot_table(
        index=["strategy", "event_id"], columns="fill_alpha", values="ret"
    )
    alphas = sorted(c for c in pivot.columns)
    violations = 0
    for lo, hi in zip(alphas, alphas[1:]):
        both = pivot[[lo, hi]].dropna()
        violations += int((both[hi] < both[lo] - 1e-9).sum())
    _require(violations == 0, f"{violations} trades where a better fill returned less")
    means = {a: round(float(pivot[a].mean()), 4) for a in alphas}
    return f"{len(pivot):,} trades monotone in alpha; mean ret by alpha {means}"


# --------------------------------------------------------------------------
# 2. registry
# --------------------------------------------------------------------------


@check("registry", description="champions load, hashes match, features are servable")
def check_registry() -> str:
    registry = load_registry()
    _require(
        registry.entries,
        "registry is empty — run `python3 -m engine.models.training.train_all`",
    )
    problems = registry.validate()
    _require(not problems, "; ".join(problems))

    loaded = []
    for entry in registry.entries:
        if not entry.champion:
            continue
        artifact = registry.load(entry)
        assert_live_available(artifact.features, label=entry.id)
        _require(
            artifact.residuals.size > 0,
            f"{entry.id}: no OOS residuals stored — the model layer cannot build "
            "a distribution without them",
        )
        loaded.append(f"{entry.id}({entry.role})")
    _require(loaded, "no champions registered")
    return f"{len(loaded)} champions verified: {', '.join(sorted(loaded))}"


@check("registry_poison", needs_data=False,
       description="a tampered artifact or feature list is refused, not used")
def check_registry_poison() -> str:
    """The integrity checks must fail loudly on a corrupted registry."""
    registry = load_registry()
    champions = [e for e in registry.entries if e.champion]
    if not champions:
        return "SKIP: no champions registered"
    entry = champions[0]

    original = entry.artifact_sha256
    entry.artifact_sha256 = "0" * 64
    try:
        registry.load(entry)
        raise AssertionError("a wrong artifact hash was accepted")
    except RegistryError as exc:
        _require("hash mismatch" in str(exc), f"unexpected error: {exc}")
    finally:
        entry.artifact_sha256 = original

    features = list(entry.features)
    entry.features = list(reversed(features))
    try:
        registry.load(entry)
        raise AssertionError("a permuted feature list was accepted")
    except RegistryError as exc:
        _require("feature list disagrees" in str(exc), f"unexpected error: {exc}")
    finally:
        entry.features = features

    try:
        assert_live_available(["ema12r_abs", *LIVE_UNAVAILABLE])
        raise AssertionError("an unservable feature was accepted")
    except UnservableFeature:
        pass
    return "hash, feature-order, and servability poisons all refused"


# --------------------------------------------------------------------------
# 3. the load-bearing equivalences
# --------------------------------------------------------------------------


@check("replay_equivalence",
       description="the scorer reproduces the replayed trades' entry pricing")
def check_replay_equivalence() -> str:
    engine = scorer()
    details = []
    for strategy in ("STR-THRU", "STR-RUNUP"):
        result = phase1_replay.check_replay_equivalence(
            strategy, n_dates=10, scorer=engine, verbose=False
        )
        if "skipped" in result:
            details.append(f"{strategy}: SKIP ({result['skipped']})")
            continue
        _require(
            result["passed"],
            f"{strategy}: {result['cost_mismatches']} cost mismatches "
            f"(max Δ {result['max_cost_delta']:.2e}), "
            f"{result['scorer_could_not_price']} unpriced. {result['worst']}",
        )
        details.append(
            f"{strategy}: {result['compared']} trades, max Δ {result['max_cost_delta']:.1e}"
        )
    return "; ".join(details)


@check("feature_equivalence",
       description="the live feature path reproduces the panel path")
def check_feature_equivalence() -> str:
    result = phase1_replay.check_feature_equivalence(
        n_events=150, verbose=False, context=scorer().context
    )
    _require(
        result["passed"],
        f"max |panel − live| = {result['max_delta']:.2e} > "
        f"{phase1_replay.FEATURE_TOLERANCE:.0e}. {result['worst']} {result['failures']}",
    )
    return (
        f"{result['events_compared']} events, max Δ {result['max_delta']:.1e} "
        f"(tolerance {phase1_replay.FEATURE_TOLERANCE:.0e})"
    )


# --------------------------------------------------------------------------
# 4. determinism
# --------------------------------------------------------------------------


@check("determinism", description="(request, snapshot) → a byte-identical result")
def check_determinism() -> str:
    """Guide acceptance test 2, on the real store rather than a fixture."""
    engine = scorer()
    trades = _engine_trades()
    sample = trades[np.isclose(trades["fill_alpha"].astype(float), 0.5)].head(200)
    _require(len(sample) > 0, "no mid-fill trades to score")

    events = store.read_table(
        "earnings_events", columns=["event_id", "ticker", "event_date", "session"]
    ).set_index("event_id")

    rng = np.random.default_rng(7)
    picked = sample.iloc[rng.choice(len(sample), size=min(8, len(sample)), replace=False)]
    keys = set()
    for row in picked.itertuples(index=False):
        keys.add((str(row.ticker), pd.Timestamp(row.entry_date)))
        keys.add((str(row.ticker), pd.Timestamp(row.exit_date)))
    index = load_chain_index(keys, progress_every=0)

    requests = []
    for row in picked.itertuples(index=False):
        event_id = str(row.event_id)
        session = str(events.loc[event_id, "session"]) if event_id in events.index else None
        requests.append(
            ScoreRequest(
                ticker=str(row.ticker), strategy=str(row.strategy),
                event_date=pd.Timestamp(row.event_date), session=session, fill=MID,
            )
        )

    digests = []
    for request in requests:
        first = engine.score(request, chain_index=index)
        second = engine.score(request, chain_index=index)
        _require(
            first.digest() == second.digest(),
            f"{request.ticker} {request.event_date}: two identical requests "
            "produced different results",
        )
        digests.append(first.digest())

    # A second Scorer must agree too: determinism cannot depend on a cache warmed
    # by an earlier call in the same process. It reuses the first Scorer's
    # immutable data context (panel + daily_market + calendar) but rebuilds every
    # piece of scoring state — registry, enriched trades, matcher, and the
    # model/payoff/recalibration caches — from scratch. Sharing the read-only
    # context is what lets two Scorers coexist on this machine; the guarantee
    # under test is about the scoring state, which is not shared.
    fresh = Scorer(context=engine.context, snapshot=engine.snapshot)
    _require(
        fresh.score(requests[0], chain_index=index).digest() == digests[0],
        f"{requests[0].ticker}: a freshly constructed Scorer disagreed",
    )
    return (
        f"{len(requests)} requests identical on repeat, and identical across a "
        "freshly constructed Scorer"
    )


# --------------------------------------------------------------------------
# 5. leak discipline on the real path
# --------------------------------------------------------------------------


@check("poison", description="a leaked feature or a late decision raises, not warns")
def check_poison() -> str:
    """Guide acceptance test 3, exercised through the real scoring path."""
    from engine.audit import assert_causal
    from engine.features import FeatureContext, panel_features

    context = FeatureContext.load()
    panel = context.panel
    row = panel.iloc[-1]
    vector = panel_features(
        str(row["ticker"]), pd.Timestamp(row["date"]), context=context, session="AMC"
    )
    name = next(iter(vector.feature_as_of))
    poisoned = vector.with_stamps(**{name: vector.as_of + pd.Timedelta(days=1)})
    try:
        assert_causal(poisoned)
        raise AssertionError("a feature stamped after the decision was accepted")
    except LeakError:
        pass

    # And the decision-level guard: a BMO print decided on its own event date.
    engine = scorer()
    events = store.read_table(
        "earnings_events", columns=["ticker", "event_date", "session"]
    )
    bmo = events[events["session"] == "BMO"]
    _require(len(bmo) > 0, "no BMO events to test the decision guard with")
    target = bmo.iloc[len(bmo) // 2]
    try:
        engine.score(
            ScoreRequest(
                ticker=str(target["ticker"]),
                strategy="STR-THRU",
                event_date=pd.Timestamp(target["event_date"]),
                session="BMO",
                as_of=pd.Timestamp(target["event_date"]),
                fill=MID,
            )
        )
        raise AssertionError("a BMO decision on the event date was accepted")
    except LeakError:
        pass
    return "feature-stamp poison and BMO same-day decision both raised"


@check("analog_causality", description="no analog closed on or after the decision")
def check_analog_causality() -> str:
    """The easiest leak in the system to introduce, so it is asserted directly."""
    engine = scorer()
    trades = engine.trades
    early = trades[
        (trades["strategy"] == "STR-THRU")
        & np.isclose(trades["fill_alpha"].astype(float), 0.5)
    ].sort_values("event_date")
    _require(len(early) > 0, "no STR-THRU mid-fill trades")

    # A decision date part-way through the sample: enough history before it for
    # a real pool, and plenty after it to leak from if the filter were absent.
    as_of = pd.Timestamp(early["event_date"].quantile(0.5))
    buckets = engine.matcher.buckets_for(
        mcap_usd=5e9, dte=2, moneyness_pct=0.0, implied_ratio=1.0
    )
    unbounded = engine.matcher.match("STR-THRU", buckets, alpha=0.5)
    bounded = engine.matcher.match("STR-THRU", buckets, alpha=0.5, as_of=as_of)
    _require(
        bounded.n < unbounded.n,
        f"the as-of bound changed nothing ({bounded.n} vs {unbounded.n}) — "
        "the causality filter is not being applied",
    )
    _require(bounded.n > 0, "the as-of bound emptied the pool; nothing was tested")
    _require(
        max(bounded.years) <= as_of.year,
        f"analogs from after the decision year {as_of.year}: {bounded.years}",
    )

    # The years tuple is coarse, so assert the exact rule on the rows themselves:
    # every eligible trade must have CLOSED strictly before the decision.
    pool = engine.trades
    eligible = pool[
        (pool["strategy"] == "STR-THRU")
        & np.isclose(pool["fill_alpha"].astype(float), 0.5)
        & (pd.to_datetime(pool["exit_date"]) < as_of)
    ]
    latest = pd.to_datetime(eligible["exit_date"]).max()
    _require(
        latest < as_of,
        f"an eligible analog closed at {latest} — on or after the {as_of.date()} decision",
    )
    return (
        f"as-of {as_of.date()} cuts the analog pool from {unbounded.n:,} to "
        f"{bounded.n:,}; latest eligible exit {latest.date()}"
    )


# --------------------------------------------------------------------------
# 6. the guide's hard constraints
# --------------------------------------------------------------------------


@check("cal_p_disabled", description="CAL-P returns a refusal, never a number")
def check_cal_p_disabled() -> str:
    engine = scorer()
    events = store.read_table(
        "earnings_events", years=[2024], columns=["ticker", "event_date", "session"]
    )
    events = events[events["session"].notna()].head(5)
    _require(len(events) > 0, "no events to test with")
    for row in events.itertuples(index=False):
        result = engine.score(
            ScoreRequest(
                ticker=str(row.ticker), strategy="CAL-P",
                event_date=pd.Timestamp(row.event_date), session=str(row.session),
                fill=MID,
            )
        )
        _require(
            result.exp_pnl_model is None and result.exp_pnl_analog is None,
            f"CAL-P produced numbers for {row.ticker}",
        )
        _require(
            "UNVALIDATED_STRUCTURE" in result.flags,
            f"CAL-P missing its refusal flag for {row.ticker}",
        )
    _require("CAL-P" in DISABLED_STRATEGIES, "CAL-P is not in DISABLED_STRATEGIES")
    from engine.payoff import PAYOFF_DRIVER

    _require("CAL-P" not in PAYOFF_DRIVER, "CAL-P has a payoff map it should not have")
    registry = load_registry()
    _require(
        not registry.has_champion("gate", "CAL-P"),
        "a CAL-P gate is registered for a structure nobody may trade",
    )
    return f"{len(events)} CAL-P requests all refused with UNVALIDATED_STRUCTURE"


@check("flags", description="EXTRAPOLATED and the layer flags reach the output")
def check_flags() -> str:
    """Guide acceptance test 6."""
    engine = scorer()
    trades = _engine_trades()
    sample = trades[
        (trades["strategy"] == "STR-THRU")
        & np.isclose(trades["fill_alpha"].astype(float), 0.5)
    ].tail(50)
    _require(len(sample) > 0, "no trades to score")

    events = store.read_table(
        "earnings_events", columns=["event_id", "ticker", "event_date", "session"]
    ).set_index("event_id")
    keys = set()
    for row in sample.itertuples(index=False):
        keys.add((str(row.ticker), pd.Timestamp(row.entry_date)))
        keys.add((str(row.ticker), pd.Timestamp(row.exit_date)))
    index = load_chain_index(keys, progress_every=0)

    seen: set[str] = set()
    atm_seen = far_seen = False
    for row in sample.itertuples(index=False):
        event_id = str(row.event_id)
        session = str(events.loc[event_id, "session"]) if event_id in events.index else None
        base = dict(
            ticker=str(row.ticker), strategy="STR-THRU",
            event_date=pd.Timestamp(row.event_date), session=session, fill=MID,
        )
        atm = engine.score(ScoreRequest(**base), chain_index=index)
        seen |= set(atm.flags)
        if atm.entry_cost is not None and not atm.extrapolated:
            atm_seen = True
        if atm.entry_cost is None or not atm.spot:
            continue

        # A strike the chain actually lists, beyond the ATM band. An arbitrary
        # spot × 1.30 is usually not a listed strike, and since `strike=` binds a
        # real contract it is correctly refused as NO_CHAIN rather than priced
        # ATM and mislabelled — so testing EXTRAPOLATED with one would only
        # re-test the refusal.
        chain = index.get(str(row.ticker), pd.Timestamp(row.entry_date))
        if chain is None or chain.empty:
            continue
        at_expiry = chain[chain["expiry"] == pd.Timestamp(atm.expiry)]
        strikes = at_expiry["strike"].dropna().unique()
        away = [
            float(k) for k in strikes
            if abs(float(k) / atm.spot - 1.0) * 100.0 > ATM_TOLERANCE_PCT
        ]
        if not away:
            continue
        far = engine.score(ScoreRequest(**base, strike=away[0]), chain_index=index)
        seen |= set(far.flags)
        if far.entry_cost is not None and far.extrapolated:
            far_seen = True
        if atm_seen and far_seen:
            break

    _require(atm_seen, "no ATM request came back un-extrapolated")
    _require(far_seen, "a listed strike beyond the ATM band was not labelled EXTRAPOLATED")
    undeclared = seen - set(FLAGS)
    _require(not undeclared, f"flags raised that FLAGS does not declare: {undeclared}")
    return f"ATM/EXTRAPOLATED both observed; flags seen: {sorted(seen)}"


# --------------------------------------------------------------------------
# 7. calibration
# --------------------------------------------------------------------------


@check("calibration", description="shipped win rates are honest (not over-confident)")
def check_calibration() -> str:
    """Guide acceptance test 4, reclassified by the 2026-08-30 decision record.

    The floor is no longer "beats the base-rate predictor" — that is a property
    of the registered models, which Phase 2 exists to improve, and holding Phase 1
    hostage to it was circular (see reports/phase1_decision_calibration_reclassification.md).
    The floor is now the correctness half: the shipped win rate must not be
    over-confident, i.e. its Brier skill must not be meaningfully negative once the
    recalibration layer has had its say. Beating the base rate is reported as the
    baseline Phase 2 must improve on, not asserted here.
    """
    reports = phase1_calibration.run(
        ("STR-THRU",), from_year=2023, sample=300, scorer=scorer(), verbose=False
    )
    doc = reports.get("STR-THRU", {})
    if "skipped" in doc:
        return f"SKIP: {doc['skipped']}"
    model = doc["model_layer"]
    _require(
        model["n"] >= 100,
        f"only {model['n']} scored events — too few to grade calibration",
    )
    _require(
        model["brier_skill"] >= MIN_BRIER_SKILL,
        f"Brier skill {model['brier_skill']:+.4f} is worse than the "
        f"{MIN_BRIER_SKILL:+.2f} floor — the shipped win rates are over-confident "
        f"(Brier {model['brier']:.4f} vs base-rate {model['brier_base_rate']:.4f}). "
        "The recalibration layer should prevent this; see the decision record.",
    )
    verdict = "beats base rate" if model["beats_base_rate"] else "below base rate (Phase 2 target)"
    return (
        f"n={model['n']} base={model['base_rate']:.3f} brier={model['brier']:.4f} "
        f"vs base {model['brier_base_rate']:.4f} (skill {model['brier_skill']:+.4f}, "
        f"{verdict}); reliability monotonicity {model['reliability_monotonicity']:+.2f}"
    )


# --------------------------------------------------------------------------
# 8. exit criterion — speed
# --------------------------------------------------------------------------


@check("calendar_speed", description="a 3-week calendar scores inside the budget")
def check_calendar_speed() -> str:
    """The guide's exit criterion: under five minutes from cache.

    Benchmarked on the **last three weeks the calendar actually covers**, not on
    the three weeks ahead of today. The cached ORATS calendar currently ends
    2026-08-27 and carries no forward events at all, so a forward window scores
    zero rows in zero seconds and the budget check passes having measured
    nothing. A historical window of the same shape exercises the same code over a
    real board and is a real measurement; the forward calendar is a data gap for
    the Sep-1 pull to close, and it is reported separately rather than hidden
    inside a green tick here.
    """
    engine = scorer()
    events = store.read_table("earnings_events", columns=["event_date", "session"])
    events = events[events["session"].notna()]
    _require(len(events) > 0, "no events with a session in the calendar")

    last = pd.Timestamp(events["event_date"].max()).normalize()
    today = pd.Timestamp.today().normalize()
    forward = int((events["event_date"] >= today).sum())
    as_of = last - pd.Timedelta(days=21)

    started = time.time()
    board = score_calendar(
        as_of=as_of, horizon_days=21, scorer=engine, progress_every=0
    )
    elapsed = time.time() - started

    _require(
        len(board) > 0,
        f"the {as_of.date()}–{last.date()} window scored no rows; the budget "
        "check would prove nothing",
    )
    _require(
        elapsed < CALENDAR_BUDGET_S,
        f"scoring a 3-week calendar took {elapsed:.0f}s, over the "
        f"{CALENDAR_BUDGET_S}s budget",
    )
    scored = int(board["exp_pnl_analog"].notna().sum())
    note = "" if forward else f"; NOTE: 0 forward events (calendar ends {last.date()})"
    return (
        f"{len(board):,} rows in {elapsed:.0f}s ({scored:,} with an analog estimate) "
        f"over {as_of.date()}–{last.date()}{note}"
    )


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

ORDER = [
    "unittests",
    "trade_set",
    "fill_monotonicity",
    "registry",
    "registry_poison",
    "replay_equivalence",
    "feature_equivalence",
    "determinism",
    "poison",
    "analog_causality",
    "cal_p_disabled",
    "flags",
    "calibration",
    "calendar_speed",
]


def run(names: list[str], skip_data: bool = False) -> list[CheckOutcome]:
    outcomes: list[CheckOutcome] = []
    for name in names:
        spec = REGISTRY[name]
        if skip_data and spec["needs_data"]:
            outcomes.append(CheckOutcome(name, True, "skipped (--no-data)", skipped=True))
            print(f"  SKIP  {name}", flush=True)
            continue
        started = time.time()
        print(f"  ...   {name}", flush=True)
        try:
            detail = spec["fn"]() or ""
            passed = True
        except Exception as exc:  # noqa: BLE001 - a failing check must not end the run
            detail = f"{type(exc).__name__}: {exc}"
            passed = False
        elapsed = time.time() - started
        skipped = isinstance(detail, str) and detail.startswith("SKIP")
        outcomes.append(CheckOutcome(name, passed, detail, elapsed, skipped))
        status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        print(f"  {status:5s} {name}  ({elapsed:.1f}s)  {detail}", flush=True)
    return outcomes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", choices=ORDER, default=None)
    ap.add_argument("--no-data", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.list:
        for name in ORDER:
            spec = REGISTRY[name]
            flag = "data" if spec["needs_data"] else "pure"
            print(f"  {name:20s} [{flag}]  {spec['description']}")
        return 0

    names = args.only or ORDER
    print(f"Phase 1 acceptance checks ({len(names)} checks)\n", flush=True)
    started = time.time()
    outcomes = run(names, skip_data=args.no_data)

    failed = [o for o in outcomes if not o.passed]
    skipped = [o for o in outcomes if o.skipped]
    print(
        f"\n{len(outcomes) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped in {time.time()-started:.0f}s"
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps([o.__dict__ for o in outcomes], indent=1, default=str)
        )
    if failed:
        print("\nFAILED:", file=sys.stderr)
        for outcome in failed:
            print(f"  {outcome.name}: {outcome.detail}", file=sys.stderr)
        return 1
    print("\nPHASE 1 CHECKS: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
