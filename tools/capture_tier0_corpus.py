#!/usr/bin/env python3
"""Capture the tier-0 corpus: frozen ``(request, record)`` pairs (phase 0 step 4).

    python3 tools/capture_tier0_corpus.py                 # fixtures/tier0/<version>/
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
* **Real public entry points.** ``engine.score.Scorer.score`` for scores,
  ``engine.score.dynamic_short_vol`` for the chooser, ``engine.replay.replay_one``
  for a disabled structure priced under research. §3.2: do not invent a column
  such as ``event_id`` in a fixture if the current serving row does not carry
  one.
* **Private.** The fixtures carry real quotes. ``checks/repo_hygiene.py`` blocks
  ``fixtures/`` from the public repo.

**Coverage is reported, never faked.** The §7.1 table is a set of axes, and
what each axis MEANS is :func:`checks.tier0_corpus.derive_covers` — one
definition, used here to select and there to re-derive. The capture scores a
wide window and then selects the covering subset from what the store actually
produced. An axis nothing covered is written into ``INDEX.json`` as a named
gap. A fixture invented to fill a row of a table proves nothing about the
engine.

**Relations are frozen, not implied.** A pinned fixture records which
selector-resolved pair it was pinned FROM, and that pair is kept, so the
`e845f3e` regression can be checked on real data. A DYN-SV fixture freezes the
exact rows, in order, the chooser ranked, so tier 1 can re-score them and
re-run the choice; tie-breaking depends on that order.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.tier0_corpus import derive_covers, priced  # noqa: E402
from engine import replay as replay_mod, score as score_mod  # noqa: E402
from engine.data import store  # noqa: E402
from engine.fills import MID  # noqa: E402
from engine.structures import STRUCTURES  # noqa: E402
from engine.v2.diagnosis import content_hash  # noqa: E402

SCHEMA_VERSION = "tier0_pair.v1.1"
INDEX_VERSION = "tier0_corpus.v1.1"
DEFAULT_OUT = ROOT / "fixtures" / "tier0"

#: How a NaN is frozen. Not ``null``: contracts §2.1 forbids sending a missing
#: value as NaN, and collapsing the two here would lose the distinction between
#: "the engine produced NaN" and "the engine produced nothing" — which is half
#: of what the null-mask comparison exists to catch.
NONFINITE = "__nonfinite__"

#: The refusal codes of §7.1, and the flag the current engine emits for each.
#: Six, not seven: ``BAD_QUOTE_COST_PCT`` is the 30% threshold constant in
#: ``engine.fills`` behind the single ``BAD_QUOTE`` flag (``engine/score.py``
#: emits ``BAD_QUOTE`` in exactly one place, on that bar), not a separate
#: refusal. The baseline package exports the constant.
REFUSAL_CODES = {code: code for code in (
    "UNVALIDATED_STRUCTURE", "OUT_OF_DOMAIN", "NO_CHAIN", "BAD_QUOTE",
    "COARSE_LADDER", "NO_FORECAST",
)}

MODEL_ROLES = ("size", "implied_t1", "runup_move", "iv_crush", "gate", "chooser")

#: ``ScoreRequest`` fields serialized as dates.
_DATE_FIELDS = frozenset({"as_of", "event_date", "expiry", "chain_as_of"})


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


def request_from_dict(data: dict) -> score_mod.ScoreRequest:
    """Inverse of :func:`request_to_dict`, field by field."""
    kwargs = {}
    for f in dataclass_fields(score_mod.ScoreRequest):
        if f.name not in data:
            continue
        value = data[f.name]
        if f.name == "fill":
            value = score_mod.FillModel(alpha=float(value["alpha"]))
        elif f.name in _DATE_FIELDS and isinstance(value, str):
            value = pd.Timestamp(value)
        kwargs[f.name] = value
    return score_mod.ScoreRequest(**kwargs)


# --------------------------------------------------------------------------
# one pair
# --------------------------------------------------------------------------


def make_pair(fixture_id: str, covers: list[str], request: dict, record: dict,
              *, record_kind: str, duration: float,
              legacy_trace: dict | None = None,
              relations: dict | None = None, notes: str = "") -> dict:
    payload: dict[str, Any] = {"request": request, "record": record,
                               "record_kind": record_kind}
    if legacy_trace is not None:
        # This is the source execution trace, not a Phase 4 acceptance bundle.
        # Acceptance requires a typed request, sidecars, and native receipts;
        # publication records the current disposition explicitly so an
        # incomplete trace cannot be mistaken for a completed one.
        payload["legacy_trace"] = legacy_trace
        payload["trace_disposition"] = "incomplete"
    if relations:
        payload["relations"] = relations
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
# required coverage (§7.1)
# --------------------------------------------------------------------------


def axis_inputs() -> dict:
    """Everything :func:`derive_covers` needs, frozen into the index."""
    return {
        "structures": sorted(STRUCTURES),
        "dynamic_strategy": score_mod.DYNAMIC_STRATEGY,
        "menu": list(score_mod.DYNAMIC_MENU),
        "disabled": list(score_mod.DISABLED_STRATEGIES),
        "model_roles": list(MODEL_ROLES),
        "refusal_code_mapping": dict(REFUSAL_CODES),
    }


def required_axes() -> list[str]:
    axes = [f"strategy:{name}" for name in STRUCTURES]
    axes.append(f"strategy:{score_mod.DYNAMIC_STRATEGY}")
    # Every SERVED strategy must also appear PRICED — legs and an entry cost,
    # not a refusal. The disabled pair is exempt: production refuses them by
    # design, and their priced behaviour is the research_replay axis instead.
    axes += [f"priced:{name}" for name in STRUCTURES
             if name not in score_mod.DISABLED_STRATEGIES]
    axes.append(f"priced:{score_mod.DYNAMIC_STRATEGY}")
    axes += [f"model_role:{r}" for r in MODEL_ROLES]
    axes += [f"refusal:{c}" for c in REFUSAL_CODES]
    axes += ["session:BMO", "session:AMC", "boundary:year", "boundary:month"]
    axes += ["geometry:pinned", "geometry:selector", "geometry:computed_width",
             "geometry:round_listed_strike", "geometry:coarse_ladder",
             "geometry:exact_mirror"]
    # `dyn_sv:tie` is deliberately NOT required (decision 2026-09-12). No
    # genuine tie between two different structures exists in the store or in
    # the prediction ledger — chooser scores are continuous — and the corpus
    # may not invent one. The tie RULE (input-row order breaks a tie, on both
    # ranking paths) is guarded instead by the frozen definition in
    # `definitions/dyn_sv.json` and by
    # `tests/test_baseline_export.py::test_the_exported_tie_rule_is_the_measured_behaviour`,
    # which runs the real `dynamic_short_vol` on tied rows in both orders.
    # `derive_covers` still reports the axis if a real tie is ever captured.
    axes += ["dyn_sv:full_menu", "dyn_sv:partial_menu", "dyn_sv:fallback"]
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
    window that cannot cross a year boundary for ANY event.
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

    The year boundary needs a print early enough in January that a d-14 entry
    lands in December, on a name the chain store carries in December, and a
    structure that actually enters pre-print — so the year candidates ride on
    STR-RUNUP.
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


def _score(scorer, request, *, index=None) -> tuple[dict, dict, float, dict]:
    """``(raw as_dict, jsonable record, seconds)`` through ``Scorer.score``.

    The raw row is kept for the chooser frame: ``dynamic_short_vol`` reads the
    board's own rows, with NaN where the engine produced NaN, not the frozen
    ``__nonfinite__`` markers.
    """
    started = time.monotonic()
    as_of = request.as_of if request.as_of is not None else request.chain_as_of
    try:
        trace = score_mod.Phase4TraceCollector()
        result = (scorer.score(request, chain_index=index, trace=trace) if index is not None
                  else scorer.score(request, trace=trace))
    except score_mod.UNSCORABLE as exc:
        result = score_mod.unscorable_result(
            request, as_of=as_of, snapshot=scorer.snapshot, exc=exc
        )
    raw = result.as_dict()
    return raw, jsonable(raw), time.monotonic() - started, trace.document()


def _candidate(request, raw: dict | None, record: dict, took: float, *,
               kind: str = "score_result", frame: str | None = None,
               relations: dict | None = None, legacy_trace: dict | None = None) -> dict:
    return {"request": request if isinstance(request, dict) else request_to_dict(request),
            "raw": raw, "record": record, "duration": took, "kind": kind,
            "frame": frame, "relations": relations or {},
            "legacy_trace": legacy_trace}


def forward_pass(scorer, events: pd.DataFrame, as_of: pd.Timestamp,
                 quote_max_age: int) -> list[dict]:
    """Every strategy on every forward event, as the board's scoring loop does."""
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
            raw, record, took, trace = _score(scorer, request, index=index)
            out.append(_candidate(request, raw, record, took, frame="forward",
                                  legacy_trace=trace))
    return out


def boundary_pass(scorer, events: pd.DataFrame) -> list[dict]:
    """Past events, scored at their own decision close, for the two boundaries.

    ``as_of`` is the structure's DECISION date, resolved through the calendar,
    not the print date. Scoring a BMO print as of the print itself is a leak —
    ``engine.audit`` refuses it. These are also where the priced:S axes are
    won: in the forward window the forecast-sized families come back
    NO_FORECAST with empty legs.
    """
    out: list[dict] = []
    for strategy in STRUCTURES:
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
                raw, record, took, trace = _score(scorer, request)
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                print(f"[corpus]   skipped {row['ticker']} {strategy}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            out.append(_candidate(request, raw, record, took, frame="boundary",
                                  legacy_trace=trace))
    return out


def _rescore(scorer, source: dict, label: str, **changes) -> dict | None:
    """Re-score a captured request with some fields changed — same clock.

    The changed request inherits the source's ``as_of``, ``chain_as_of`` and
    quote-age policy, so a pinned or strike variant of a historical row is
    scored at that row's decision date rather than today's.
    """
    request = replace(request_from_dict(source["request"]), **changes)
    try:
        raw, record, took, trace = _score(scorer, request)
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        print(f"[corpus]   {label} skip {request.ticker} {request.strategy}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None
    return _candidate(request, raw, record, took, legacy_trace=trace)


def _anchor_strike(record: dict) -> float | None:
    """A strike the row's legs actually resolved to — a LISTED strike."""
    legs = [leg for leg in record.get("legs") or [] if isinstance(leg, dict)]
    anchor = next((leg for leg in legs if leg.get("name") == "atm"), legs[0] if legs else None)
    strike = (anchor or {}).get("strike")
    return float(strike) if isinstance(strike, (int, float)) else None


def pinned_and_strike_pass(scorer, scored: list[dict], limit: int = 8) -> list[dict]:
    """Re-score priced rows with their geometry pinned, and at a listed strike.

    The pinned pair is the `e845f3e` regression case made permanent: a replay
    that pins the shape must still record the forecast that chose it. It is
    evidence only beside the selector-resolved row it was pinned FROM, so the
    relation is recorded and :func:`select` keeps the source.

    The strike pair asks for a strike the source's legs resolved to — a real
    listed strike, not a computed moneyness the chain would snap away from.
    """
    out: list[dict] = []
    for source in scored:
        record = source["record"]
        params = record.get("structure_params")
        if not priced(record) or not isinstance(params, dict) or not params:
            continue
        pinned = _rescore(scorer, source, "pinned",
                          structure_params={k: v for k, v in params.items() if v is not None})
        if pinned is not None:
            pinned["relations"] = {"pinned_from": content_hash(source["request"])}
            out.append(pinned)
        listed = _anchor_strike(record)
        at_strike = (_rescore(scorer, source, "strike", strike=listed)
                     if listed is not None else None)
        if at_strike is not None:
            out.append(at_strike)
        if len(out) >= limit:
            break
    return out


#: Widths tried, narrowest first, when hunting a COARSE_LADDER refusal.
_COARSE_WIDTHS = (0.002, 0.004, 0.006, 0.01)


def coarse_ladder_pass(scorer, scored: list[dict]) -> list[dict]:
    """Ask for a width the ticker's listed ladder cannot carry.

    §7.1 requires a coarse ladder and it cannot be waited for, so it is
    *requested* — a real ``structure_params`` width, through the real entry
    point, narrow enough that two legs resolve onto one contract. Asking for a
    refusal is not the same as inventing one.
    """
    for source in scored:
        record = source["record"]
        if record.get("spot") is None or record.get("strategy") not in ("TWIN-P5", "TWIN-P"):
            continue
        for width in _COARSE_WIDTHS:
            got = _rescore(scorer, source, "coarse",
                           structure_params={"width_moneyness": width})
            if got is None:
                break
            if "COARSE_LADDER" in (got["record"].get("flags") or []):
                return [got]
    return []


def _event_key(record: dict) -> tuple:
    return (record.get("ticker"), record.get("event_date"))


def dyn_sv_pass(candidates: list[dict]) -> list[dict]:
    """The chooser, run per event over a BOARD-SHAPED frame.

    Only rows the board's scoring loop produces — one per structure per event,
    at the ATM pass — enter the frame. The first corpus fed the chooser every
    candidate, pinned copies included, and its only "tie" was BFLY-P tying with
    its own pinned re-score. The event's rows are frozen in frame order inside
    the request: ``dynamic_short_vol`` breaks a tie by that order, so a replay
    that did not reproduce it would not reproduce the choice.
    """
    out: list[dict] = []
    for frame_name in ("forward", "boundary"):
        members = [c for c in candidates if c.get("frame") == frame_name]
        events: dict[tuple, list[dict]] = {}
        for cand in members:
            events.setdefault(_event_key(cand["record"]), []).append(cand)
        for key, siblings in events.items():
            frame = pd.DataFrame([c["raw"] | {"strike_offset": None} for c in siblings])
            chosen = score_mod.dynamic_short_vol(frame)
            if chosen.empty:
                continue
            request = {
                "kind": "dyn_sv_resolution",
                "entry_point": "engine.score.dynamic_short_vol",
                "menu": list(score_mod.DYNAMIC_MENU),
                "frame": frame_name,
                "frame_rows": [{"request": c["request"], "record": c["record"]}
                               for c in siblings],
            }
            record = jsonable(chosen.iloc[0].to_dict())
            out.append(_candidate(request, None, record, 0.0, kind="dyn_sv_choice"))
    return out


def research_replay_pass(scorer, events: pd.DataFrame, limit: int = 2) -> list[dict]:
    """Price CAL-P and CND-P under research, where the scorer refuses them.

    §7.1 requires both to appear as *refusals* on the production path and to
    *replay* under research. That is a different entry point with a different
    record, and conflating the two would lose the distinction.
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
            record = {"rows": jsonable(rows), "skip_reason": skip, "strategy": strategy}
            out.append(_candidate(request, None, record, time.monotonic() - started,
                                  kind="research_replay"))
            taken += 1
            if taken >= limit:
                break
    return out


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def select(candidates: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """A minimal covering subset, greedily, plus the axis -> fixtures index.

    Greedy rather than exhaustive: what matters is that every axis is covered
    by *some* frozen pair, not that the subset is provably the smallest. After
    the greedy pass every chosen pinned fixture pulls in the pair it was
    pinned from — without it the pinned pair demonstrates nothing.
    """
    inputs = axis_inputs()
    for cand in candidates:
        cand["covers"] = derive_covers(cand["record"], cand["request"], cand["kind"],
                                       inputs, cand.get("relations"))

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

    by_request = {content_hash(c["request"]): c for c in candidates}
    chosen_hashes = {content_hash(c["request"]) for c in chosen}
    for cand in list(chosen):
        source = (cand.get("relations") or {}).get("pinned_from")
        if source and source not in chosen_hashes and source in by_request:
            chosen.append(by_request[source])
            chosen_hashes.add(source)

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


def _publish_current(root: Path, version: str) -> None:
    """Point ``CURRENT`` at a version directory, atomically."""
    tmp = root / "CURRENT.tmp"
    tmp.write_text(json.dumps({"version": version}, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, root / "CURRENT")


def write(out_dir: Path, chosen: list[dict], index: dict[str, list[str]],
          as_of: pd.Timestamp, snapshot: str, *, replace_existing: bool = False) -> dict:
    """Publish one immutable version directory, atomically.

    The version is built under a temporary sibling and published with one
    rename; an existing non-empty version directory refuses without an
    explicit ``--replace``.
    """
    if out_dir.exists() and any(out_dir.iterdir()) and not replace_existing:
        raise SystemExit(
            f"{out_dir} already exists and is not empty. A frozen corpus is "
            "never overwritten in place: capture a NEW version directory, or "
            "re-run with --replace to authorize replacing this one explicitly.")
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    pairs_dir = tmp / "pairs"
    pairs_dir.mkdir(parents=True)

    manifest_pairs = {}
    for cand in chosen:
        pair = make_pair(
            cand["fixture_id"], cand["covers"], cand["request"], cand["record"],
            record_kind=cand["kind"], duration=cand["duration"],
            legacy_trace=cand.get("legacy_trace"),
            relations=cand.get("relations"),
        )
        text = json.dumps(pair, indent=2, sort_keys=True) + "\n"
        (pairs_dir / f"{cand['fixture_id']}.json").write_text(text)
        manifest_pairs[cand["fixture_id"]] = {
            "payload_hash": pair["payload_hash"],
            "request_hash": pair["request_hash"],
            "record_kind": cand["kind"],
            "covers": pair["covers"],
            "trace_disposition": pair["payload"].get("trace_disposition", "absent"),
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
        # Everything checks/tier0_corpus.py needs to RE-DERIVE coverage from
        # the surviving records alone, with no engine import.
        "axis_inputs": axis_inputs(),
        "pairs": manifest_pairs,
        "coverage": {axis: sorted(ids) for axis, ids in sorted(index.items())},
        "required_axes": required_axes(),
        "uncovered_axes": missing,
        "corpus_hash": content_hash(
            {k: v["payload_hash"] for k, v in sorted(manifest_pairs.items())}
        ),
    }
    (tmp / "INDEX.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.rename(tmp, out_dir)
    return doc


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None,
                    help="write exactly here (skips the versioned layout)")
    ap.add_argument("--version", default=None,
                    help="version directory name under fixtures/tier0 "
                         "(default: UTC timestamp)")
    ap.add_argument("--replace", action="store_true",
                    help="authorize replacing an existing non-empty version")
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

    boundaries = _boundary_events(as_of, args.boundary_events, scorer.calendar)
    print(f"[corpus] boundary events: {len(boundaries)}", flush=True)
    candidates += boundary_pass(scorer, boundaries)

    candidates += pinned_and_strike_pass(scorer, candidates)
    candidates += coarse_ladder_pass(scorer, candidates)
    candidates += dyn_sv_pass(candidates)
    candidates += research_replay_pass(scorer, boundaries)
    print(f"[corpus] candidates: {len(candidates)}", flush=True)

    chosen, index = select(candidates)
    if args.out:
        out_dir = Path(args.out)
    else:
        version = args.version or datetime.now(
            timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = DEFAULT_OUT / version
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    doc = write(out_dir, chosen, index, as_of, scorer.snapshot,
                replace_existing=args.replace)
    if not args.out and out_dir.parent == DEFAULT_OUT:
        _publish_current(DEFAULT_OUT, out_dir.name)
        print(f"[corpus] CURRENT -> {out_dir.name}")

    print(f"[corpus] wrote {len(chosen)} pairs to {out_dir}")
    print(f"[corpus] corpus hash {doc['corpus_hash']}")
    total = len(doc["required_axes"])
    print(f"[corpus] coverage {total - len(doc['uncovered_axes'])}/{total} required axes")
    if doc["uncovered_axes"]:
        print("[corpus] UNCOVERED (recorded as gaps, not faked):")
        for axis in doc["uncovered_axes"]:
            print(f"    {axis}")
    print(f"[corpus] total {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
