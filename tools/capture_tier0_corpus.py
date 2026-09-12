#!/usr/bin/env python3
"""Capture the tier-0 corpus: frozen ``(request, record)`` pairs (phase 0 step 4).

    python3 tools/capture_tier0_corpus.py                 # fixtures/tier0/
    python3 tools/capture_tier0_corpus.py --out /tmp/c1
    python3 tools/capture_tier0_corpus.py --forward-days 35 --max-events 60

This is the slow half of the loop and it runs once: it builds a real
:class:`engine.score.Scorer` (which loads the panel and half a million replayed
trades), scores real events through the real public entry points, and writes
the answers down. Everything after it — ``checks/tier0_corpus.py`` — runs in
seconds against what this wrote, with no panel, no network and no fitting.

Four capture rules, each of them a defect this program has already paid for:

* **Full precision.** Replay inputs are serialized unrounded. ``b33036c`` and
  ``6b9d5cf`` are exactly this: ``json_safe`` rounded ``structure_params`` to
  six places and ``_write_pair`` re-rounded it after the exemption. A corpus
  written through the board's display path would freeze the bug as the
  baseline, so nothing here goes near ``round_to``.
* **Deterministic payload, separate envelope.** Wall-clock time, worker id and
  duration live outside the hashed payload (contracts §2.2), so a replay
  reproduces the payload without reproducing the elapsed time.
* **Real public entry points.** ``engine.score.score`` for scores,
  ``engine.score.dynamic_short_vol`` for the chooser, ``engine.replay.replay_one``
  for a disabled structure priced under research. §3.2: do not invent a column
  such as ``event_id`` in a fixture if the current serving row does not carry
  one.
* **Private.** The fixtures carry real quotes. ``checks/repo_hygiene.py`` blocks
  ``fixtures/`` from the public repo, and that block landed before this script
  was first run rather than after.

**Coverage is reported, never faked.** The §7.1 table is encoded below as a set
of axes; the capture scores a wide window and then *selects* the covering
subset from what the store actually produced. An axis nothing covered is
written into ``INDEX.json`` as a named gap with the reason. A fixture invented
to fill a row of a table proves nothing about the engine.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import replay as replay_mod, score as score_mod  # noqa: E402
from engine.data import store  # noqa: E402
from engine.fills import MID  # noqa: E402
from engine.structures import STRUCTURES  # noqa: E402
from engine.v2.diagnosis import content_hash  # noqa: E402

SCHEMA_VERSION = "tier0_pair.v1.0"
INDEX_VERSION = "tier0_corpus.v1.0"
DEFAULT_OUT = ROOT / "fixtures" / "tier0"

#: How a NaN is frozen. Not ``null``: contracts §2.1 forbids sending a missing
#: value as NaN, and collapsing the two here would lose the distinction between
#: "the engine produced NaN" and "the engine produced nothing" — which is half
#: of what the null-mask comparison exists to catch.
NONFINITE = "__nonfinite__"

#: The refusal codes of §7.1, and what the current engine actually emits for
#: each. `BAD_QUOTE_COST_PCT` is a *constant* in `engine.fills`, not a flag:
#: the flag is BAD_QUOTE and the constant is the threshold named in its detail.
#: Recorded as a compatibility mapping rather than resolved silently, per
#: contracts §9.4.
REFUSAL_CODES = {
    "UNVALIDATED_STRUCTURE": "UNVALIDATED_STRUCTURE",
    "OUT_OF_DOMAIN": "OUT_OF_DOMAIN",
    "NO_CHAIN": "NO_CHAIN",
    "BAD_QUOTE": "BAD_QUOTE",
    "BAD_QUOTE_COST_PCT": "BAD_QUOTE",   # same flag; the detail names the bar
    "COARSE_LADDER": "COARSE_LADDER",
    "NO_FORECAST": "NO_FORECAST",
}

MODEL_ROLES = ("size", "implied_t1", "runup_move", "iv_crush", "gate", "chooser")


# --------------------------------------------------------------------------
# full-precision serialization
# --------------------------------------------------------------------------


def jsonable(value: Any) -> Any:
    """Convert to JSON-writable form **without rounding anything, ever**."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return {NONFINITE: repr(out)} if not math.isfinite(out) else out
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int)):
        return value
    if hasattr(value, "_asdict"):
        return jsonable(value._asdict())
    if hasattr(value, "__dataclass_fields__"):
        return {f.name: jsonable(getattr(value, f.name))
                for f in dataclass_fields(value)}
    return str(value)


def request_to_dict(request: score_mod.ScoreRequest) -> dict:
    """The exact replay request, at full precision.

    Per contracts §9.5 this is persisted independently of any display
    projection: a client replays from the saved request or the score id, never
    from rounded values copied out of a table.
    """
    out = {f.name: jsonable(getattr(request, f.name))
           for f in dataclass_fields(request) if f.name != "fill"}
    out["fill"] = {"policy_id": "legacy.fill_alpha.v1",
                   "alpha": float(request.fill.alpha)}
    out["identity_key"] = request.key()
    return out


# --------------------------------------------------------------------------
# one pair
# --------------------------------------------------------------------------


def make_pair(fixture_id: str, covers: list[str], request: dict, record: dict,
              *, record_kind: str, duration: float, notes: str = "") -> dict:
    payload = {"request": request, "record": record, "record_kind": record_kind}
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "covers": sorted(set(covers)),
        "notes": notes,
        "payload": payload,
        "payload_hash": content_hash(payload),
        "request_hash": content_hash(request),
        # contracts §2.2: the envelope is excluded from the payload hash, so a
        # replay reproduces the payload without reproducing the elapsed time.
        "envelope": {
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "worker_ref": f"{platform.node()}:{os.getpid()}",
            "duration_seconds": duration,
        },
    }


# --------------------------------------------------------------------------
# what each record covers
# --------------------------------------------------------------------------


def _roles_exercised(record: dict) -> set[str]:
    """Which of the six registered model roles this record shows running."""
    out: set[str] = set()
    versions = record.get("model_versions") or {}
    for role in MODEL_ROLES:
        if role in versions or any(role in str(k) for k in versions):
            out.add(role)
    if record.get("forecast_model"):
        out.add("size")
    if record.get("driver_prediction") is not None:
        out.add("size")
    if record.get("runup_move_prediction") is not None:
        out.add("runup_move")
    if record.get("implied_move_at_entry") is not None:
        out.add("implied_t1")
    if record.get("exp_pnl_sim") is not None:
        out.add("iv_crush")
    if record.get("gate_score") is not None or record.get("gate_pass") is not None:
        out.add("gate")
    if record.get("chooser_score") is not None:
        out.add("chooser")
    return out


def _boundary(record: dict) -> set[str]:
    """Whether this trade's window crosses a month or a year boundary."""
    entry, exit_ = record.get("entry_date"), record.get("exit_date")
    if not entry or not exit_:
        return set()
    out = set()
    if entry[:4] != exit_[:4]:
        out.add("boundary:year")
    if entry[:7] != exit_[:7]:
        out.add("boundary:month")
    return out


def _geometry(record: dict, request: dict) -> set[str]:
    out: set[str] = set()
    params = record.get("structure_params")
    if request.get("structure_params"):
        out.add("geometry:pinned")
    elif params:
        out.add("geometry:selector")
    if isinstance(params, dict) and params.get("width_moneyness") is not None:
        out.add("geometry:computed_width")
    if request.get("strike") is not None:
        out.add("geometry:round_listed_strike")
    if "COARSE_LADDER" in (record.get("flags") or []):
        out.add("geometry:coarse_ladder")
    legs = record.get("legs") or []
    # SORTED strikes: leg order is position order (the short anchor first, wings
    # after), which is a property of the structure's leg list, not of its
    # geometry. Mirror symmetry lives in the strike SET — a priced BFLY-P whose
    # legs read [7.5, 10, 5] is exactly as symmetric as one that reads
    # [5, 7.5, 10], and reading gaps in leg order made the axis unreachable.
    strikes = sorted(leg.get("strike") for leg in legs
                     if isinstance(leg, dict) and leg.get("strike") is not None)
    if len(strikes) >= 3:
        gaps = [round(b - a, 6) for a, b in zip(strikes, strikes[1:])]
        if len(set(gaps)) == 1:
            out.add("geometry:exact_mirror")
    return out


def covers_of(record: dict, request: dict) -> list[str]:
    out = {f"strategy:{record.get('strategy')}"}
    if record.get("session"):
        out.add(f"session:{record['session']}")
    for flag in record.get("flags") or []:
        for code, emitted in REFUSAL_CODES.items():
            if emitted == flag:
                out.add(f"refusal:{code}")
    out |= {f"model_role:{r}" for r in _roles_exercised(record)}
    out |= _boundary(record)
    out |= _geometry(record, request)
    return sorted(out)


# --------------------------------------------------------------------------
# required coverage (§7.1)
# --------------------------------------------------------------------------


def required_axes() -> list[str]:
    axes = [f"strategy:{name}" for name in STRUCTURES]
    axes.append(f"strategy:{score_mod.DYNAMIC_STRATEGY}")
    axes += [f"model_role:{r}" for r in MODEL_ROLES]
    axes += [f"refusal:{c}" for c in REFUSAL_CODES]
    axes += ["session:BMO", "session:AMC", "boundary:year", "boundary:month"]
    axes += ["geometry:pinned", "geometry:selector", "geometry:computed_width",
             "geometry:round_listed_strike", "geometry:coarse_ladder",
             "geometry:exact_mirror"]
    axes += ["dyn_sv:full_menu", "dyn_sv:partial_menu", "dyn_sv:tie",
             "dyn_sv:fallback"]
    for name in score_mod.DISABLED_STRATEGIES:
        axes += [f"disabled:{name}:refused", f"disabled:{name}:research_replay"]
    return sorted(set(axes))


# --------------------------------------------------------------------------
# scoring passes
# --------------------------------------------------------------------------


def _events(as_of: pd.Timestamp, forward_days: int, max_events: int) -> pd.DataFrame:
    events = store.read_table(
        "earnings_events", columns=["event_id", "ticker", "event_date", "session"]
    )
    horizon = as_of + pd.Timedelta(days=forward_days)
    forward = events[(events["event_date"] >= as_of)
                     & (events["event_date"] <= horizon)
                     & events["session"].notna()]
    forward = forward.sort_values(["event_date", "ticker"]).head(max_events)
    return forward.reset_index(drop=True)


def _with_chains(candidates: pd.DataFrame, calendar, per_kind: int,
                 structure: str = "STR-THRU") -> pd.DataFrame:
    """Keep only events whose entry and exit chains are both in the store.

    Without this the boundary fixtures come back as NO_CHAIN placeholders, which
    carry no entry or exit date and therefore cannot demonstrate a boundary at
    all — a fixture that covers the axis in name only.

    ``structure`` is the structure whose plan defines the window. The year kind
    checks STR-RUNUP rather than STR-THRU because STR-THRU enters on the last
    pre-print session and exits on the first post-print one — a one-session
    window that cannot cross a year boundary for ANY event, so filtering the
    year candidates through it filtered out the axis itself.
    """
    if candidates.empty:
        return candidates
    available = replay_mod.available_chain_keys()
    plan = replay_mod.plan_events(STRUCTURES[structure](), candidates,
                                  calendar=calendar)
    keep = []
    for row in plan.frame.to_dict("records"):
        entry = (row["ticker"], pd.Timestamp(row["entry_date"]).normalize())
        exit_ = (row["ticker"], pd.Timestamp(row["exit_date"]).normalize())
        if entry in available and exit_ in available:
            keep.append(row["event_id"])
        if len(keep) >= per_kind:
            break
    return candidates[candidates["event_id"].isin(keep)]


def _boundary_events(as_of: pd.Timestamp, per_kind: int, calendar) -> pd.DataFrame:
    """Past events whose trade window crosses a month or a year boundary.

    A year boundary is the scarce one, and measured against the store it needs
    TWO things the first cut did not have. The print must sit early enough in
    January that a d-14 entry lands in December — the 2025-01-06..10 cohort
    (ACI, AIR, AYI, AZZ, CALM, CMC, GBX, HELE, MSM, NEOG, SMPL, STZ, TLRY:
    thirteen events with both December entry and January exit chains, against
    ZERO for the Jan-1..3 prints, which are small names the chain store does
    not carry in December). And the structure scored must actually enter
    pre-print: STR-THRU's one-session window never crosses the boundary, so the
    year fixtures ride on STR-RUNUP.
    """
    events = store.read_table(
        "earnings_events", columns=["event_id", "ticker", "event_date", "session"]
    )
    past = events[(events["event_date"] < as_of)
                  & (events["event_date"] >= as_of - pd.Timedelta(days=2500))
                  & events["session"].notna()].copy()
    past["day"] = past["event_date"].dt.day
    past["month"] = past["event_date"].dt.month
    year = _with_chains(
        past[(past["month"] == 1) & (past["day"] <= 12)].sort_values(
            "event_date", ascending=False),
        calendar, per_kind, structure="STR-RUNUP")
    month = _with_chains(
        past[past["day"] <= 2].sort_values("event_date", ascending=False),
        calendar, per_kind)
    return pd.concat([year, month]).drop_duplicates("event_id").reset_index(drop=True)


def _score(scorer, request, *, index=None, as_of) -> tuple[dict, float]:
    started = time.monotonic()
    try:
        result = (scorer.score(request, chain_index=index) if index is not None
                  else scorer.score(request))
    except score_mod.UNSCORABLE as exc:
        result = score_mod.unscorable_result(
            request, as_of=as_of, snapshot=scorer.snapshot, exc=exc
        )
    return jsonable(result.as_dict()), time.monotonic() - started


def forward_pass(scorer, events: pd.DataFrame, as_of: pd.Timestamp,
                 quote_max_age: int) -> list[dict]:
    """Every strategy on every forward event, through ``Scorer.score``."""
    keys: set[tuple[str, pd.Timestamp]] = set()
    for strategy in STRUCTURES:
        if strategy in score_mod.DISABLED_STRATEGIES:
            continue
        plan = replay_mod.plan_events(
            STRUCTURES[strategy](), events, calendar=scorer.calendar
        )
        keys |= plan.chain_keys
    for ticker in events["ticker"].astype(str).unique():
        try:
            newest = replay_mod.latest_chain_date(ticker, as_of)
        except Exception:  # pragma: no cover - store-dependent
            newest = None
        if newest is not None:
            keys.add((ticker, pd.Timestamp(newest).normalize()))
    index = replay_mod.load_chain_index(keys, progress_every=0) if keys else None

    out: list[dict] = []
    for row in events.itertuples(index=False):
        for strategy in STRUCTURES:
            request = score_mod.ScoreRequest(
                ticker=str(row.ticker), strategy=strategy, as_of=None,
                event_date=pd.Timestamp(row.event_date), session=str(row.session),
                fill=MID, quote_max_age_sessions=quote_max_age, chain_as_of=as_of,
            )
            record, took = _score(scorer, request, index=index, as_of=as_of)
            out.append({"request": request_to_dict(request), "record": record,
                        "duration": took, "kind": "score_result"})
    return out


def boundary_pass(scorer, events: pd.DataFrame) -> list[dict]:
    """Past events, scored at their own decision close, for the two boundaries.

    ``as_of`` is the structure's DECISION date, resolved through the calendar,
    not the print date. Scoring a BMO print as of the print itself is a leak —
    ``engine.audit`` refuses it — and the refusal is correct: the last
    information-free close for a BMO print is the session before.
    """
    out: list[dict] = []
    # STR-RUNUP carries the year-boundary axis (its d-14 entry is the only one
    # that lands in December); RAMP7 and CTR5 carry geometry:exact_mirror —
    # they are the mirror-placed ladders whose seven/five unique strikes are
    # evenly gapped whenever the listed grid around the anchor is uniform, and
    # historical events are where they price at all: in the forward window the
    # forecast-sized families come back NO_FORECAST with empty legs.
    for strategy in ("STR-THRU", "TWIN-P5", "STR-RUNUP", "RAMP7", "CTR5"):
        structure = STRUCTURES[strategy]()
        plan = replay_mod.plan_events(structure, events, calendar=scorer.calendar)
        for row in plan.frame.to_dict("records"):
            as_of = pd.Timestamp(row["decision_date"])
            request = score_mod.ScoreRequest(
                ticker=str(row["ticker"]), strategy=strategy, as_of=as_of,
                event_date=pd.Timestamp(row["event_date"]),
                session=str(row["session"]), fill=MID, chain_as_of=as_of,
            )
            try:
                record, took = _score(scorer, request, as_of=as_of)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                print(f"[corpus]   skipped {row['ticker']} {strategy}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            out.append({"request": request_to_dict(request), "record": record,
                        "duration": took, "kind": "score_result"})
    return out


def pinned_and_ladder_pass(scorer, scored: list[dict], as_of) -> list[dict]:
    """Re-score a resolved row with its geometry pinned, and at a ladder strike.

    The pinned pair is the `e845f3e` regression case made permanent: a replay
    that pins the shape must still record the forecast that chose it. A fixture
    of the selector-resolved row alone cannot show that, because there is
    nothing to compare the pinned one against.
    """
    out: list[dict] = []
    for row in scored:
        record = row["record"]
        params = record.get("structure_params")
        spot = record.get("spot")
        if not params or not isinstance(params, dict) or spot is None:
            continue
        # A row that priced: it has legs and a cost. A refusal has neither, and
        # pinning its (absent) geometry would freeze a fixture of nothing.
        if not record.get("legs") or record.get("entry_cost") is None:
            continue
        base = score_mod.ScoreRequest(
            ticker=record["ticker"], strategy=record["strategy"],
            as_of=None, event_date=pd.Timestamp(record["event_date"]),
            session=record.get("session"), fill=MID,
            quote_max_age_sessions=row["request"].get("quote_max_age_sessions"),
            chain_as_of=pd.Timestamp(as_of),
            structure_params={k: v for k, v in params.items() if v is not None},
        )
        try:
            record_pinned, took = _score(scorer, base, as_of=as_of)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"[corpus]   pinned skip {base.ticker}: {exc}", flush=True)
            continue
        out.append({"request": request_to_dict(base), "record": record_pinned,
                    "duration": took, "kind": "score_result"})
        ladder = score_mod.ScoreRequest(
            ticker=record["ticker"], strategy=record["strategy"], as_of=None,
            event_date=pd.Timestamp(record["event_date"]),
            session=record.get("session"), fill=MID,
            strike=score_mod.ladder_strike(float(spot), -score_mod.LADDER_STEP),
            quote_max_age_sessions=row["request"].get("quote_max_age_sessions"),
            chain_as_of=pd.Timestamp(as_of),
        )
        try:
            record_ladder, took = _score(scorer, ladder, as_of=as_of)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"[corpus]   ladder skip {ladder.ticker}: {exc}", flush=True)
            continue
        out.append({"request": request_to_dict(ladder), "record": record_ladder,
                    "duration": took, "kind": "score_result"})
        if len(out) >= 8:
            break
    return out


#: Widths tried, narrowest first, when hunting a COARSE_LADDER refusal.
_COARSE_WIDTHS = (0.002, 0.004, 0.006, 0.01)


def coarse_ladder_pass(scorer, scored: list[dict], as_of) -> list[dict]:
    """Ask for a width the ticker's listed ladder cannot carry.

    §7.1 requires a coarse ladder in the corpus and it cannot be waited for:
    whether one appears depends on which names happen to print this month. So
    it is *requested* — a real `structure_params` width, through the real entry
    point, narrow enough that two legs resolve onto one contract. That is the
    refusal `guides/coarse_ladder_collision.md` documents, and asking for it
    is not the same as inventing it.
    """
    out: list[dict] = []
    for row in scored:
        record = row["record"]
        spot = record.get("spot")
        if spot is None or record.get("strategy") not in ("TWIN-P5", "TWIN-P"):
            continue
        for width in _COARSE_WIDTHS:
            request = score_mod.ScoreRequest(
                ticker=record["ticker"], strategy=record["strategy"], as_of=None,
                event_date=pd.Timestamp(record["event_date"]),
                session=record.get("session"), fill=MID,
                quote_max_age_sessions=row["request"].get("quote_max_age_sessions"),
                chain_as_of=pd.Timestamp(as_of),
                structure_params={"width_moneyness": width},
            )
            try:
                got, took = _score(scorer, request, as_of=as_of)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                print(f"[corpus]   coarse skip {record['ticker']}: {exc}", flush=True)
                break
            if "COARSE_LADDER" in (got.get("flags") or []):
                out.append({"request": request_to_dict(request), "record": got,
                            "duration": took, "kind": "score_result"})
                return out
    return out


def dyn_sv_pass(scored: list[dict]) -> list[dict]:
    """The chooser, resolved through ``dynamic_short_vol`` over the scored frame."""
    frame = pd.DataFrame([r["record"] for r in scored])
    if frame.empty:
        return []
    chosen = score_mod.dynamic_short_vol(frame)
    out: list[dict] = []
    for _, row in chosen.iterrows():
        record = jsonable(row.to_dict())
        menu_size = int(record.get("menu_size") or 0)
        margin = record.get("chosen_margin")
        covers = [f"strategy:{score_mod.DYNAMIC_STRATEGY}"]
        covers.append("dyn_sv:full_menu" if menu_size >= len(score_mod.DYNAMIC_MENU)
                      else "dyn_sv:partial_menu")
        if margin is not None and not isinstance(margin, dict) and float(margin) == 0.0:
            covers.append("dyn_sv:tie")
        # The fallback test reads the RAW row, not the jsonable record:
        # jsonable turns a NaN chooser_score into a ``__nonfinite__`` marker
        # dict, so `is None` never fired on exactly the rows that took the
        # resolver path. A chosen row with no finite chooser score IS the
        # fallback — `dynamic_short_vol` ranks a mixed event on the scores
        # that exist and filters NaN out, so a NaN winner means NO candidate
        # carried a score and the pre-champion resolver ranked the event.
        if pd.isna(row.get("chooser_score")):
            covers.append("dyn_sv:fallback")
        request = {
            "kind": "dyn_sv_resolution",
            "ticker": record.get("ticker"),
            "event_date": record.get("event_date"),
            "menu": list(score_mod.DYNAMIC_MENU),
            "entry_point": "engine.score.dynamic_short_vol",
        }
        out.append({"request": request, "record": record, "duration": 0.0,
                    "kind": "dyn_sv_choice", "extra_covers": covers})
    return out


def research_replay_pass(scorer, events: pd.DataFrame, limit: int = 2) -> list[dict]:
    """Price CAL-P and CND-P under research, where the scorer refuses them.

    §7.1 requires both to appear as *refusals* on the production path and to
    *replay* under research. They are in ``STRUCTURES`` precisely so
    ``engine.replay`` can price them; that is a different entry point with a
    different record, and conflating the two would lose the distinction the
    coverage table is drawing.
    """
    out: list[dict] = []
    for strategy in score_mod.DISABLED_STRATEGIES:
        structure = STRUCTURES[strategy]()
        plan = replay_mod.plan_events(structure, events, calendar=scorer.calendar)
        index = replay_mod.load_chain_index(plan.chain_keys, progress_every=0)
        taken = 0
        for row in plan.frame.to_dict("records"):
            started = time.monotonic()
            rows, skip = replay_mod.replay_one(structure, row, index,
                                               include_legs=True)
            if not rows:
                continue
            request = {
                "kind": "research_replay",
                "entry_point": "engine.replay.replay_one",
                "strategy": strategy,
                "structure": structure.to_dict(),
                "plan_row": jsonable(row),
            }
            out.append({
                "request": request,
                "record": {"rows": jsonable(rows), "skip_reason": skip,
                           "strategy": strategy},
                "duration": time.monotonic() - started,
                "kind": "research_replay",
                "extra_covers": [f"disabled:{strategy}:research_replay"],
            })
            taken += 1
            if taken >= limit:
                break
    return out


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def select(candidates: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """A minimal covering subset, greedily, plus the axis -> fixtures index.

    Greedy rather than exhaustive: the corpus has to answer in seconds, so what
    matters is that every axis is covered by *some* frozen pair, not that the
    subset is provably the smallest one.
    """
    for cand in candidates:
        covers = covers_of(cand["record"], cand["request"])
        covers += cand.get("extra_covers", [])
        if cand["record"].get("strategy") in score_mod.DISABLED_STRATEGIES:
            flags = cand["record"].get("flags") or []
            if "UNVALIDATED_STRUCTURE" in flags:
                covers.append(f"disabled:{cand['record']['strategy']}:refused")
        cand["covers"] = sorted(set(covers))

    wanted = set(required_axes())
    chosen: list[dict] = []
    remaining = list(candidates)
    while wanted and remaining:
        remaining.sort(key=lambda c: -len(wanted & set(c["covers"])))
        best = remaining.pop(0)
        gain = wanted & set(best["covers"])
        if not gain:
            break
        chosen.append(best)
        wanted -= gain

    index: dict[str, list[str]] = {}
    for i, cand in enumerate(chosen):
        cand["fixture_id"] = _fixture_id(cand, i)
        for axis in cand["covers"]:
            index.setdefault(axis, []).append(cand["fixture_id"])
    return chosen, index


def _fixture_id(cand: dict, i: int) -> str:
    record = cand["record"]
    stem = "-".join(str(x) for x in (
        record.get("strategy", cand["kind"]),
        record.get("ticker", ""),
        record.get("event_date", ""),
    ) if x)
    digest = content_hash(cand["request"])[7:15]
    return f"{i:03d}_{stem}_{digest}".replace("/", "-").replace(" ", "")


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def write(out_dir: Path, chosen: list[dict], index: dict[str, list[str]],
          as_of: pd.Timestamp, snapshot: str) -> dict:
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for existing in pairs_dir.glob("*.json"):
        existing.unlink()

    manifest_pairs = {}
    for cand in chosen:
        pair = make_pair(
            cand["fixture_id"], cand["covers"], cand["request"], cand["record"],
            record_kind=cand["kind"], duration=cand["duration"],
        )
        text = json.dumps(pair, indent=2, sort_keys=True) + "\n"
        (pairs_dir / f"{cand['fixture_id']}.json").write_text(text)
        manifest_pairs[cand["fixture_id"]] = {
            "payload_hash": pair["payload_hash"],
            "request_hash": pair["request_hash"],
            "record_kind": cand["kind"],
            "covers": cand["covers"],
        }

    missing = sorted(set(required_axes()) - set(index))
    doc = {
        "schema_version": INDEX_VERSION,
        "as_of": str(as_of.date()),
        "snapshot": snapshot,
        "tier": 0,
        "stage_plan_ref": "scorer.v1",
        "tolerance_policy_ref": "score_record.exact.v1",
        "refusal_code_mapping": REFUSAL_CODES,
        "pairs": manifest_pairs,
        "coverage": {axis: sorted(ids) for axis, ids in sorted(index.items())},
        "required_axes": required_axes(),
        "uncovered_axes": missing,
        "corpus_hash": content_hash(
            {k: v["payload_hash"] for k, v in sorted(manifest_pairs.items())}
        ),
    }
    (out_dir / "INDEX.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--forward-days", type=int, default=35)
    ap.add_argument("--max-events", type=int, default=40)
    ap.add_argument("--boundary-events", type=int, default=4)
    ap.add_argument("--quote-max-age", type=int, default=5)
    args = ap.parse_args(list(argv) if argv is not None else None)

    as_of = (pd.Timestamp(args.as_of).normalize() if args.as_of
             else pd.Timestamp.today().normalize())
    started = time.time()
    print("[corpus] building the scorer (panel + replayed trades)...", flush=True)
    scorer = score_mod.Scorer()
    print(f"[corpus] scorer ready in {time.time()-started:.0f}s", flush=True)

    forward = _events(as_of, args.forward_days, args.max_events)
    print(f"[corpus] forward events: {len(forward)}", flush=True)
    candidates = forward_pass(scorer, forward, as_of, args.quote_max_age)
    print(f"[corpus] forward scores: {len(candidates)}", flush=True)

    boundaries = _boundary_events(as_of, args.boundary_events,
                                  scorer.calendar)
    print(f"[corpus] boundary events: {len(boundaries)}", flush=True)
    candidates += boundary_pass(scorer, boundaries)

    candidates += pinned_and_ladder_pass(scorer, candidates, as_of)
    candidates += coarse_ladder_pass(scorer, candidates, as_of)
    candidates += dyn_sv_pass(candidates)
    candidates += research_replay_pass(scorer, boundaries)
    print(f"[corpus] candidates: {len(candidates)}", flush=True)

    chosen, index = select(candidates)
    doc = write(Path(args.out), chosen, index, as_of, scorer.snapshot)

    print(f"[corpus] wrote {len(chosen)} pairs to {args.out}")
    print(f"[corpus] corpus hash {doc['corpus_hash']}")
    covered = len(doc["coverage"])
    total = len(doc["required_axes"])
    print(f"[corpus] coverage {covered}/{total} required axes")
    if doc["uncovered_axes"]:
        print("[corpus] UNCOVERED (recorded as gaps, not faked):")
        for axis in doc["uncovered_axes"]:
            print(f"    {axis}")
    print(f"[corpus] total {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
