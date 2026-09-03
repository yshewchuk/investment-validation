#!/usr/bin/env python3
"""Phase 3 acceptance tests.

    python3 checks/phase3_checks.py               # everything
    python3 checks/phase3_checks.py --list
    python3 checks/phase3_checks.py --only offline_bundle secret_scan
    python3 checks/phase3_checks.py --no-data     # only checks needing no store

The guide's seven acceptance tests, plus the constraints §4 states as prose —
which are only real if something enforces them:

 0. ``unittests``        the pytest suite for the dashboard pipeline
 1. ``selfcheck``        render → green; poison a row → red AND publish refuses
 2. ``offline_bundle``   the bundle renders from file:// with the network down
 3. ``atomicity``        a kill mid-upload leaves the previous snapshot serving
 4. ``validation_gate``  a red battery stops the run before it scores or ships
 5. ``historical_dryrun`` five replayed nights, idempotent, ledger appended once
 5b ``forward_calendar`` the universe comes from a source that can see ahead
 6. ``secret_scan``      no credential, .env value or internal path in a bundle
 7. ``access_rule``      an unauthenticated 200 on the target refuses the publish
 8. ``ui_no_compute``    every field the client reads is one the renderer wrote
 9. ``calp_unvalidated`` CAL-P renders as unvalidated, matching the scorer
10. ``server_local``     the desk server binds 127.0.0.1 and keeps the quota
                         actions off the published bundle
11. ``board_budget``     board.json fits the mobile budget
12. ``registry_current`` every champion matches the code that trains it

Test 7 in the guide ("phone loads the URL through the Access login") is a manual
check the agent cannot perform; what is automated here is the rule that protects
it — publish refuses a target proven publicly readable. The manual check is
recorded, with its date, in ``reports/phase3_dashboard.md``.

A phase without green checks is not done.
"""
from __future__ import annotations

import argparse
import contextlib
import http.server
import json
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine.dashboard.publish as publish_mod  # noqa: E402
from engine import paths  # noqa: E402
from engine.dashboard.nightly import (  # noqa: E402
    NightlyStop,
    run_nightly,
    strike_ladder,
)
from engine.dashboard.render import BOARD_MAX_BYTES, render_bundle, row_digest  # noqa: E402
from engine.dashboard.selfcheck import selfcheck  # noqa: E402
from engine.score import DISABLED_STRATEGIES, Scorer, score_calendar  # noqa: E402

#: Replayed nights for the dry-run check. Small enough to run in minutes,
#: enough dates to prove the ledger appends once per as-of.
DRYRUN_NIGHTS = 5

#: The dry run scores a bounded universe: the point is the pipeline's behaviour
#: across nights, and the full board is ~2,200 events at roughly a second each.
DRYRUN_TICKERS = ["AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ORCL", "CRM", "ADBE"]


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""
    elapsed_s: float = 0.0
    skipped: bool = False


REGISTRY: dict[str, dict] = {}
_SCORER: Scorer | None = None
_BUNDLE: Path | None = None


def check(name: str, *, needs_data: bool = True, description: str = ""):
    def wrap(fn):
        REGISTRY[name] = {"fn": fn, "needs_data": needs_data, "description": description}
        return fn

    return wrap


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scorer() -> Scorer:
    """One Scorer for the whole run — building it per check would dominate."""
    global _SCORER
    if _SCORER is None:
        _SCORER = Scorer()
    return _SCORER


def _scratch(name: str) -> Path:
    """A writable working directory under reports/, cleaned per run.

    Not ``/tmp``: ``paths.assert_writable`` guards the engine's write paths, and
    keeping the artifacts inside the repo means a failed check leaves the bundle
    it failed on behind to look at.
    """
    out = paths.assert_writable(paths.REPORTS / "phase3_checks" / name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


@contextlib.contextmanager
def _isolated_ledger(name: str):
    """Point the ledger at a throwaway tree for the duration of a check.

    The real ledger is append-only, irreplaceable, and the out-of-time evidence
    the whole program leans on. A rehearsal that writes backdated rows into it
    would pad the calibration sample with predictions nobody made — so any check
    that runs the nightly (which freezes predictions at step 4) redirects it
    first.
    """
    root = _scratch(name)
    original = paths.LEDGER
    paths.LEDGER = root
    try:
        yield root
    finally:
        paths.LEDGER = original


def _board_window() -> pd.Timestamp:
    """An as-of whose next three weeks actually contain events.

    The cached ORATS calendar ends before today until the Sep-1 pull lands, so a
    forward window would score zero rows and every check below would pass having
    measured nothing.
    """
    from engine.data import store

    events = store.read_table("earnings_events", columns=["event_date", "session"])
    events = events[events["session"].notna()]
    _require(len(events) > 0, "no events with a session in the calendar")
    last = pd.Timestamp(events["event_date"].max()).normalize()
    today = pd.Timestamp.today().normalize()
    return min(today, last - pd.Timedelta(days=14))


def bundle() -> Path:
    """One real rendered bundle, reused across the checks that inspect one."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    as_of = _board_window()
    engine = scorer()
    scores = score_calendar(
        as_of, horizon_days=21, scorer=engine, tickers=DRYRUN_TICKERS,
        alt_strikes=0, progress_every=0,
    )
    _require(len(scores) > 0, f"no rows scored for {as_of.date()} — nothing to check")
    ladder = strike_ladder(scores, scorer=engine, alt_strikes=1, as_of=as_of)
    if ladder:
        scores = pd.concat([scores, pd.DataFrame(ladder)], ignore_index=True)

    out = _scratch("bundle")
    render_bundle(
        scores, out, as_of=as_of, panel=engine.context.panel, trades=engine.trades,
        registry=engine.registry,
    )
    _BUNDLE = out
    return out


# --------------------------------------------------------------------------
# 0. unit tests
# --------------------------------------------------------------------------


@check("unittests", needs_data=False, description="the dashboard pytest suite")
def check_unittests() -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_dashboard.py", "-q"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
    )
    _require(proc.returncode == 0, f"pytest failed:\n{(proc.stdout + proc.stderr)[-2000:]}")
    return proc.stdout.strip().splitlines()[-1]


@check("board_row_ids_unique",
       description="one board row, one identity — including on the strike ladder")
def check_board_row_ids_unique() -> str:
    """The self-check and the derivation view both address a row by `row_id`.

    Ladder rows that failed to resolve a strike used to share one id, so a
    digest mismatch named a row the reader could not find and the derivation
    view opened whichever of them came first. Nothing enforced uniqueness,
    which is why 24 rows sharing 12 identities shipped unnoticed.
    """
    out = bundle()
    rows = json.loads((out / "data" / "board.json").read_text())["rows"]
    _require(rows, "the board has no rows to check")
    seen: dict[str, int] = {}
    for row in rows:
        seen[row["row_id"]] = seen.get(row["row_id"], 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    _require(
        not dupes,
        f"{len(dupes)} duplicate row_id(s) covering {sum(dupes.values())} rows: "
        + ", ".join(sorted(dupes)[:5]),
    )
    return f"{len(rows):,} rows, {len(seen):,} distinct ids"


@check("data_scripts_referenced",
       description="every rendered data/*.js file is either eager-loaded or "
                   "reachable through the client's own on-demand loader")
def check_data_scripts_referenced() -> str:
    """A generated payload nobody's <script> tag points at is dead on arrival.

    `data/book.js` shipped correctly for a full session — real rows, `_clean`ed,
    `_write_pair`ed alongside its `.json` twin — and the client never loaded it:
    `index.html` had no `<script src="data/book.js">`, and unlike `models.js` it
    had no on-demand loader either (`app.js` reads `window.BOOK` at parse time).
    The Book tab therefore rendered its empty state against the SAME bundle in
    every environment, for as long as the tag was missing, and every check that
    inspects book.json's CONTENT would have stayed green throughout.

    Every `data/*.js` file must be either named directly in `index.html`, or
    named as a `src` inside `app.js`'s own on-demand loaders (the `models.js`
    pattern) — checked by grepping app.js for the literal filename, which is
    what a real loader's `script.src = "data/NAME.js"` assignment always is.
    """
    out = bundle()
    html = (out / "index.html").read_text()
    app_js = (out / "assets" / "app.js").read_text()
    referenced = set(re.findall(r'<script src="data/([^"]+\.js)"></script>', html))
    on_demand = {p.name for p in (out / "data").glob("*.js") if p.name in app_js}
    present = {p.name for p in (out / "data").glob("*.js")}
    orphaned = present - referenced - on_demand
    _require(
        not orphaned,
        f"{len(orphaned)} data script(s) rendered but never loaded by the "
        f"client (no <script src> in index.html, no on-demand reference in "
        f"app.js): {sorted(orphaned)}",
    )
    return (f"{len(present)} data scripts: {len(referenced)} eager, "
            f"{len(on_demand)} on-demand, 0 orphaned")


# --------------------------------------------------------------------------
# 1. self-check (guide test 1)
# --------------------------------------------------------------------------


@check("selfcheck", description="render → green; a poisoned row → red + publish refused")
def check_selfcheck() -> str:
    out = bundle()
    report = selfcheck(out, scorer=scorer())
    _require(report.ok, f"self-check red on a freshly rendered bundle: {report.detail}")
    _require(report.n_checked > 0, "the self-check re-scored nothing")

    # Poison one row the way a tampered or stale bundle would, and confirm both
    # gates close: the check goes red, and nothing ships.
    poisoned = _scratch("poisoned")
    shutil.copytree(out, poisoned, dirs_exist_ok=True)
    board_path = poisoned / "data" / "board.json"
    board = json.loads(board_path.read_text())
    board["rows"][0]["exp_pnl_model"] = 0.99
    board_path.write_text(json.dumps(board))

    poisoned_report = selfcheck(poisoned, scorer=scorer())
    _require(not poisoned_report.ok, "the self-check passed a poisoned board row")

    # A red check must also STOP THE PUBLISH — the publisher does not re-run the
    # self-check, the nightly gates on it, so the gate is what gets tested.
    _require(_publish_refused_on_red(), "a red self-check did not stop the publish")
    return (
        f"{report.n_checked}/{report.n_board_rows} rows re-scored clean; "
        f"poisoned row caught ({poisoned_report.mismatches[0]['reason']}) and publish refused"
    )


def _publish_refused_on_red() -> bool:
    """Run a real night whose self-check comes back red; nothing may ship."""
    import engine.dashboard.selfcheck as selfcheck_mod
    from engine.dashboard.selfcheck import SelfCheckReport

    out = _scratch("red_bundle")
    target = _scratch("red_target")
    as_of = _board_window()

    # run_nightly imports the checker at call time, so the patch goes on the
    # source module rather than on the orchestrator's namespace.
    original = selfcheck_mod.selfcheck
    selfcheck_mod.selfcheck = lambda *a, **kw: SelfCheckReport(
        ok=False, n_checked=1, n_board_rows=1,
        mismatches=[{"row_id": "injected", "reason": "injected mismatch"}],
        detail="injected failure for the acceptance check",
    )
    try:
        # This run reaches step 4, which freezes predictions — into a throwaway
        # ledger, not the real one.
        with _isolated_ledger("red_ledger"):
            try:
                run_nightly(
                    as_of, tickers=DRYRUN_TICKERS, bundle_dir=out, target=target,
                    refresh=False, publish=True, backfill=False, alt_strikes=0,
                    scorer=scorer(),
                )
            except NightlyStop as exc:
                flag_file = paths.REPORTS / "phase3_flags" / f"{as_of.date()}.json"
                if flag_file.exists():
                    flag_file.unlink()  # an injected failure is not a real night
                return exc.step == "selfcheck" and not (target / "current").exists()
            return False
    finally:
        selfcheck_mod.selfcheck = original


# --------------------------------------------------------------------------
# 2. offline bundle (guide test 2)
# --------------------------------------------------------------------------

#: A bundle that reaches the network is a bundle that breaks on a phone in a
#: tunnel — and one that could leak a request to a third party.
_NETWORK_CALLS = (
    re.compile(r"\bfetch\s*\("),
    re.compile(r"XMLHttpRequest"),
    re.compile(r"\bimport\s*\("),
    re.compile(r"navigator\.sendBeacon"),
    re.compile(r"\bWebSocket\b"),
)


def _strip_comments(text: str) -> str:
    """Drop JS/CSS comments before scanning for calls.

    The client's own header explains that ``fetch()`` is blocked on ``file://``
    — a sentence that must not read as a fetch call. Comments are removed so the
    scan sees code.
    """
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", " ", text)


@check("offline_bundle", needs_data=False,
       description="every asset is local; the client makes no network call")
def check_offline_bundle() -> str:
    out = bundle()
    html = (out / "index.html").read_text()

    external = re.findall(r'(?:src|href)\s*=\s*"([^"]+)"', html)
    # A `data:` URI is the content itself, so it is neither remote nor a file
    # that could be missing — it is the strongest form of what this checks for.
    external = [u for u in external if not u.startswith("data:")]
    remote = [u for u in external if u.startswith(("http://", "https://", "//"))]
    _require(not remote, f"index.html references remote assets: {remote}")
    for url in external:
        _require((out / url).exists(), f"index.html references a missing asset: {url}")

    for path in list(out.rglob("*.js")) + list(out.rglob("*.css")):
        text = _strip_comments(path.read_text())
        for pattern in _NETWORK_CALLS:
            hit = pattern.search(text)
            _require(
                hit is None,
                f"{path.relative_to(out)} makes a network call: {hit.group(0) if hit else ''}",
            )

    # The data the first paint needs must arrive through <script>, because
    # fetch() is blocked on file:// origins. Per-ticker files load lazily by
    # injecting their own .js wrapper, which is the same mechanism.
    for stem in ("meta", "board", "health", "flags"):
        _require(
            f'src="data/{stem}.js"' in html,
            f"index.html does not load data/{stem}.js — the offline path is broken",
        )
        js = (out / "data" / f"{stem}.js").read_text()
        payload = js.split("=", 1)[1].rstrip().rstrip(";")
        _require(
            json.loads(payload) == json.loads((out / "data" / f"{stem}.json").read_text()),
            f"data/{stem}.js and data/{stem}.json disagree",
        )

    tickers = sorted((out / "data" / "tickers").glob("*.js"))
    _require(tickers, "no per-ticker files in the bundle")
    sample = json.loads(
        tickers[0].read_text().split("] = ", 1)[1].rstrip().rstrip(";\n").rstrip(";")
    )
    _require("events" in sample, f"{tickers[0].name} carries no events block")
    return f"{len(external)} local assets, {len(tickers)} ticker files, no network calls"


# --------------------------------------------------------------------------
# 3. atomicity (guide test 3)
# --------------------------------------------------------------------------


@check("atomicity", needs_data=False,
       description="a kill mid-upload leaves the previous snapshot serving")
def check_atomicity() -> str:
    out = bundle()
    target = _scratch("target")
    first = publish_mod.LocalPublisher(target).publish(out)
    served_before = json.loads(
        (target / "current" / "data" / "meta.json").read_text()
    )["as_of"]
    release_before = (target / "current").resolve()

    # A second, different snapshot, killed part-way through its upload.
    second = _scratch("bundle_next")
    shutil.copytree(out, second, dirs_exist_ok=True)
    meta_path = second / "data" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["as_of"] = str(pd.Timestamp(meta["as_of"]) + pd.Timedelta(days=1))[:10]
    meta_path.write_text(json.dumps(meta))

    def die(path: Path) -> None:
        if path.name == "board.json":
            raise OSError("simulated kill mid-upload")

    killed = False
    try:
        publish_mod.LocalPublisher(target, copy_hook=die).publish(second)
    except OSError:
        killed = True
    _require(killed, "the simulated kill did not interrupt the upload")

    served_after = json.loads(
        (target / "current" / "data" / "meta.json").read_text()
    )["as_of"]
    _require(
        served_after == served_before and (target / "current").resolve() == release_before,
        f"the served snapshot changed under a killed publish: {served_before} → {served_after}",
    )
    return f"served as_of {served_before} unchanged through a killed publish (release {Path(first.release).name})"


# --------------------------------------------------------------------------
# 4. validation gate (guide test 4)
# --------------------------------------------------------------------------


@check("validation_gate", description="a red battery stops the run before scoring")
def check_validation_gate() -> str:
    from engine.dashboard import nightly as nightly_mod

    out = _scratch("gate_bundle")
    target = _scratch("gate_target")
    as_of = _board_window()

    original = nightly_mod.validate_refresh
    scored = {"called": False}

    def red(*args, **kwargs):
        return [{"name": "daily_freshness", "passed": False,
                 "detail": "injected failure for the acceptance check"}]

    class Tripwire:
        """Any attempt to score after a red battery is the failure itself."""

        snapshot = "unused"

        def score(self, *args, **kwargs):
            scored["called"] = True
            raise AssertionError("scored after a red validation battery")

    nightly_mod.validate_refresh = red
    try:
        stopped = None
        with _isolated_ledger("gate_ledger"):
            try:
                run_nightly(
                    as_of, tickers=DRYRUN_TICKERS, bundle_dir=out, target=target,
                    refresh=False, publish=False, backfill=False, scorer=Tripwire(),
                )
            except NightlyStop as exc:
                stopped = exc
    finally:
        nightly_mod.validate_refresh = original

    _require(stopped is not None, "a red validation battery did not stop the run")
    _require(stopped.step == "validate", f"stopped at {stopped.step}, expected validate")
    _require(not scored["called"], "the run scored despite the red battery")
    _require(not (out / "data" / "board.json").exists(), "a bundle was rendered anyway")
    _require(not (target / "current").exists(), "something was published anyway")

    flag_file = paths.REPORTS / "phase3_flags" / f"{as_of.date()}.json"
    _require(flag_file.exists(), "no flag report written for the stopped run")
    flags = json.loads(flag_file.read_text())["flags"]
    _require(
        any(f["kind"] == "validation_red" for f in flags),
        f"no validation_red flag raised: {[f['kind'] for f in flags]}",
    )
    flag_file.unlink()  # the injected failure is not a real night
    return "run stopped at validate; nothing scored, rendered or published; flag raised"


# --------------------------------------------------------------------------
# 5. historical dry-run (guide test 5)
# --------------------------------------------------------------------------


@check("historical_dryrun",
       description="five replayed nights: coherent snapshots, ledger appended once")
def check_historical_dryrun() -> str:
    from engine import ledger

    out = _scratch("dryrun_bundle")
    target = _scratch("dryrun_target")

    last = _board_window()
    nights = [last - pd.Timedelta(days=k) for k in range(DRYRUN_NIGHTS - 1, -1, -1)]
    with _isolated_ledger("dryrun_ledger") as ledger_root:
        reports = []
        for as_of in nights:
            reports.append(run_nightly(
                as_of, tickers=DRYRUN_TICKERS, bundle_dir=out, target=target,
                refresh=False, publish=True, backfill=False, alt_strikes=0,
                scorer=scorer(),
            ))
            _require(
                json.loads((out / "data" / "meta.json").read_text())["as_of"]
                == str(as_of.date()),
                f"the bundle's meta does not match the night it rendered ({as_of.date()})",
            )
            _require(
                (target / "current" / "data" / "board.json").exists(),
                f"nothing published for {as_of.date()}",
            )

        rows_per_night = {r.as_of: r.steps["ledger"].get("rows", 0) for r in reports}
        files = sorted(p.name for p in (ledger_root / "predictions").glob("*.jsonl"))
        _require(
            len(files) == len([n for n, c in rows_per_night.items() if c]),
            f"ledger files {files} do not match the nights that wrote rows {rows_per_night}",
        )

        # Idempotence: the same night again must append nothing and still render.
        before = {p: p.read_bytes() for p in (ledger_root / "predictions").glob("*.jsonl")}
        again = run_nightly(
            nights[-1], tickers=DRYRUN_TICKERS, bundle_dir=out, target=target,
            refresh=False, publish=True, backfill=False, alt_strikes=0, scorer=scorer(),
        )
        _require(
            again.steps["ledger"].get("rows", 0) == 0,
            f"a re-run appended {again.steps['ledger'].get('rows')} ledger rows",
        )
        after = {p: p.read_bytes() for p in (ledger_root / "predictions").glob("*.jsonl")}
        _require(before == after, "a re-run changed the ledger files on disk")

        # Every written row must be strict JSON: the ledger cannot be corrected.
        n_rows = 0
        for path in (ledger_root / "predictions").glob("*.jsonl"):
            for line in path.read_text().splitlines():
                if line.strip():
                    json.loads(line)  # raises on NaN
                    n_rows += 1

    return (
        f"{len(nights)} nights replayed, {n_rows} ledger rows across {len(files)} file(s), "
        f"re-run appended 0 and published again"
    )


# --------------------------------------------------------------------------
# 5b. the forward calendar
# --------------------------------------------------------------------------


@check("forward_calendar",
       description="the board's universe comes from a source that can see ahead")
def check_forward_calendar() -> str:
    """ORATS cannot supply this, and a board with no forward events is empty.

    The failure this guards against is silent: ``/hist/earnings`` returns 200
    with real rows and simply has nothing in the future, so a refresh built on
    it looks healthy and produces a blank board forever.
    """
    from engine.calendar import SESSION_PRIORITY
    from engine.data import store

    events = store.read_table(
        "earnings_events",
        columns=["ticker", "event_date", "session", "session_src", "src_orats",
                 "src_nasdaq", "src_yfinance", "date_conflict"],
    )
    events["event_date"] = pd.to_datetime(events["event_date"])
    today = pd.Timestamp.today().normalize()
    forward = events[events["event_date"] >= today]

    _require(
        len(forward) > 0,
        "the calendar carries no forward events — run "
        "`python3 -m engine.data.pulls.forward_calendar`",
    )
    _require(
        not forward["src_orats"].astype(bool).any(),
        "ORATS appears to carry forward events; if /hist/earnings has started "
        "returning them, SESSION_PRIORITY and this check should be revisited",
    )
    scoreable = forward[forward["session"].notna()]
    _require(
        len(scoreable) > 0,
        f"{len(forward)} forward events but none with a session — the scorer "
        "skips every one of them, so the board would still be empty",
    )

    # Every session must name the source accountable for it, in priority order.
    sources = set(scoreable["session_src"].dropna().unique())
    _require(
        sources <= set(SESSION_PRIORITY),
        f"session_src holds unknown sources: {sorted(sources - set(SESSION_PRIORITY))}",
    )
    _require(
        scoreable["session_src"].notna().all(),
        "a row carries a session with no source — untraceable BMO/AMC is what "
        "shifts an entry by a day",
    )

    # A rival date must be flagged, never silently resolved. "Rival" means
    # inside the conflict window: two dates a quarter apart are the next two
    # prints, which is the cadence working, not a disagreement.
    #
    # Evaluated over every ORATS-unconfirmed row, not just the forward slice —
    # the pairs that matter most straddle today (one source thinks the company
    # already reported, the other that it is about to), and a forward-only view
    # would see one half of the pair and call the flag spurious.
    from engine.calendar import CONFLICT_WINDOW_DAYS

    unconfirmed = events[~events["src_orats"].astype(bool)]
    rivals = set()
    for ticker, group in unconfirmed.groupby("ticker"):
        dates = group["event_date"].sort_values()
        if (dates.diff().dt.days <= CONFLICT_WINDOW_DAYS).any():
            rivals.add(ticker)
    flagged = set(
        unconfirmed[unconfirmed["date_conflict"].fillna(False).astype(bool)]["ticker"]
    )
    _require(
        rivals == flagged,
        f"date_conflict disagrees with the {CONFLICT_WINDOW_DAYS}d rule: "
        f"{len(rivals - flagged)} unflagged (e.g. {sorted(rivals - flagged)[:3]}), "
        f"{len(flagged - rivals)} flagged without a rival "
        f"(e.g. {sorted(flagged - rivals)[:3]})",
    )
    conflicts = forward[forward["date_conflict"].fillna(False).astype(bool)]
    return (
        f"{len(forward)} forward events, {len(scoreable)} scoreable "
        f"({dict(scoreable['session_src'].value_counts())}); "
        f"{conflicts['ticker'].nunique()} ticker(s) flagged date_conflict"
    )


# --------------------------------------------------------------------------
# 6. secret scan (guide test 6)
# --------------------------------------------------------------------------


@check("secret_scan", needs_data=False, description="no secret, .env value or /root/ path ships")
def check_secret_scan() -> str:
    out = bundle()
    hits = publish_mod.secret_scan(out)
    _require(
        not hits,
        f"{len(hits)} secret-scan hit(s), e.g. "
        + (f"{hits[0]['file']} / {hits[0]['pattern']}" if hits else ""),
    )

    # And the scanner must actually be able to fail: a planted value proves the
    # green above is a measurement rather than a scanner that never fires.
    planted = _scratch("planted")
    shutil.copytree(out, planted, dirs_exist_ok=True)
    (planted / "data" / "planted.json").write_text('{"api_key": "not-a-real-key-000"}')
    _require(publish_mod.secret_scan(planted), "the secret scanner missed a planted key")
    try:
        publish_mod.LocalPublisher(_scratch("planted_target")).publish(planted)
        raise AssertionError("a bundle with a planted secret was published")
    except publish_mod.PublishError:
        pass
    return f"{sum(1 for _ in out.rglob('*'))} bundle files clean; planted key caught and refused"


# --------------------------------------------------------------------------
# 7. the access rule (guide test 7, automated half)
# --------------------------------------------------------------------------


@check("access_rule", needs_data=False,
       description="an unauthenticated 200 on the target refuses the publish")
def check_access_rule() -> str:
    out = bundle()

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — stdlib naming
            body = b'{"as_of": "public"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Quiet) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{port}/board"
            probe = publish_mod.access_probe(url)
            _require(probe["public"], f"the probe did not see a public 200: {probe}")

            publisher = publish_mod.CommandPublisher("true {bundle}", probe_url=url)
            try:
                publisher.publish(out)
                raise AssertionError("published to a target that is publicly readable")
            except publish_mod.PublishError as exc:
                _require(
                    "publicly readable" in str(exc),
                    f"refused for the wrong reason: {exc}",
                )
        finally:
            httpd.shutdown()
            thread.join(timeout=5)

    # A login redirect is not proof of publicness, and must not block a publish.
    original = publish_mod.access_probe
    publish_mod.access_probe = lambda url, **kw: {"status": 302, "public": False, "error": None}
    try:
        result = publish_mod.CommandPublisher(
            "true {bundle}", probe_url="https://example.invalid/board"
        ).publish(out)
        _require(result.probe["status"] == 302, "the probe result was not recorded")
    finally:
        publish_mod.access_probe = original
    return "public 200 refused; 302 (Access login) allowed and recorded"


# --------------------------------------------------------------------------
# 8. the UI computes nothing (guide §4)
# --------------------------------------------------------------------------

#: Names the client reads off a board row / ticker row that the RENDERER must
#: therefore have written. Anything else means the UI is deriving a number.
_ROW_ACCESS = re.compile(r"\b(?:r|row|entry|h|t|s|b)\.([a-z_][a-z0-9_]*)\b")


@check("ui_no_compute", needs_data=False,
       description="every field the client displays is one the renderer wrote")
def check_ui_no_compute() -> str:
    out = bundle()
    app = (out / "assets" / "app.js").read_text()
    board = json.loads((out / "data" / "board.json").read_text())
    _require(board["rows"], "the board has no rows to check against")

    known: set[str] = set(board["rows"][0])
    for row in board["rows"]:
        known |= set(row)
    ticker_file = next((out / "data" / "tickers").glob("*.json"))
    ticker = json.loads(ticker_file.read_text())
    known |= set(ticker)

    # The derivation view reads strategies.json, so its keys are renderer
    # output too — the rule is that the client displays what the renderer
    # wrote, whichever payload it came from.
    strategies = json.loads((out / "data" / "strategies.json").read_text())
    for block in strategies.values():
        known |= set(block)
        for nested in ("structure", "layers", "model", "gate"):
            if isinstance(block.get(nested), dict):
                known |= set(block[nested])

    models = json.loads((out / "data" / "models.json").read_text())
    known |= set(models)
    for block in (models.get("models") or {}).values():
        known |= set(block)
        for entry in block.get("inputs") or []:
            known |= set(entry)
        for entry in (block.get("inputs") or [])[:1]:
            for decile in entry.get("deciles") or []:
                known |= set(decile)
        for nested in ("kind", "sampled"):
            if isinstance(block.get(nested), dict):
                known |= set(block[nested])
    for event in ticker["events"]:
        known |= set(event)
        for row in event["rows"]:
            known |= set(row)
    for section in ("history", "analogs"):
        for entry in ticker.get(section) or []:
            known |= set(entry)

    #: Locals and DOM/JS members the pattern also catches. Listed rather than
    #: pattern-excluded so a new one is a deliberate edit, not a silent hole.
    allowed = known | {
        "length", "rows", "flags", "kind", "detail", "changes", "as_ofs", "step",
        "ticker", "old", "new", "reasons", "remaining", "cells", "strike",
        "offset", "value", "textContent", "innerHTML", "classList", "dataset",
        "onclick", "onchange", "oninput", "onload", "onerror", "src", "head",
        "map", "join", "filter", "sort", "slice", "push", "forEach", "toFixed",
        "replace", "split", "trim", "toLowerCase", "localeCompare", "includes",
        "appendChild", "createElement", "querySelectorAll", "getElementById",
        "add", "remove", "toggle", "concat", "eidx", "ridx", "reverse",
        "available", "reason", "series", "n", "model_mae_pp", "as_of",
        "implied_baseline_mae_pp", "per_strategy", "brier_skill", "base_rate",
        "predicted_mean_pnl", "realized_mean_pnl", "n_predictions", "n_scored",
        "ok", "n_checked", "n_board_rows", "mismatches", "snapshot_ok",
        "flagged", "date", "start", "end", "note", "side", "qty", "right",
        "oos", "features", "id", "target", "train_window", "promoted",
        "threshold", "artifact_sha256", "enabled", "disabled_reason",
    }
    unknown = sorted(set(_ROW_ACCESS.findall(app)) - allowed)
    _require(
        not unknown,
        f"the client reads {unknown} — fields the renderer does not write, so the "
        "self-check cannot cover them",
    )

    # The derived display values must exist in the bundle, not in the client.
    for field in ("rank", "entry_cost_pct", "model_fair_pct", "premium_vs_fair"):
        _require(
            any(field in row for row in board["rows"]),
            f"derived field {field!r} is missing from board.json — if the client "
            "computes it, the self-check does not cover it",
        )
    return f"{len(known)} bundle fields; client reads no field the renderer did not write"


# --------------------------------------------------------------------------
# 9. CAL-P stays labelled (guide §4)
# --------------------------------------------------------------------------


@check("calp_unvalidated", needs_data=False,
       description="CAL-P renders as unvalidated, matching the scorer's disabled state")
def check_calp_unvalidated() -> str:
    _require(
        "CAL-P" in DISABLED_STRATEGIES,
        "CAL-P is no longer disabled in the scorer — the board's label must move with it",
    )
    app = (bundle() / "assets" / "app.js").read_text()
    _require(
        "EXP-101/102" in app and "unvalidated" in app,
        "the client does not badge CAL-P rows as unvalidated — pending EXP-101/102",
    )
    out = bundle()
    board = json.loads((out / "data" / "board.json").read_text())
    calp = [r for r in board["rows"] if r["strategy"] == "CAL-P"]
    for row in calp:
        _require(
            row.get("exp_pnl_model") is None and row.get("exp_pnl_analog") is None,
            f"a CAL-P row carries a P&L estimate: {row['row_id']}",
        )
    meta = json.loads((out / "data" / "meta.json").read_text())
    _require(
        meta["strategies"]["CAL-P"]["enabled"] is False,
        "meta.json does not report CAL-P as disabled",
    )
    return f"{len(calp)} CAL-P rows, all unscored and badged; meta reports it disabled"


# --------------------------------------------------------------------------
# 10. the desk server's rules (guide §3)
# --------------------------------------------------------------------------


@check("server_local", needs_data=False,
       description="the desk server binds 127.0.0.1; the bundle has no mutating surface")
def check_server_local() -> str:
    from dashboard import earnings_app

    source = (ROOT / "dashboard" / "earnings_app.py").read_text()
    _require(
        earnings_app.DEFAULT_HOST == "127.0.0.1",
        f"the desk server defaults to {earnings_app.DEFAULT_HOST}, not loopback",
    )
    _require(
        earnings_app.DEFAULT_PORT == 8712,
        f"the desk server defaults to port {earnings_app.DEFAULT_PORT}; 8711 is the "
        "semis scanner's and the two must not share one",
    )
    # An override exists for a containerised desk, but an all-interfaces bind
    # must never be what the file itself chooses.
    _require(
        "0.0.0.0" not in source,
        "the desk server hardcodes an all-interfaces bind — that must stay an "
        "explicit environment override, never the default",
    )
    mutating = re.findall(r'@app\.(post|put|delete|patch)\("([^"]+)"', source)
    _require(
        all(path.startswith("/api/") for _, path in mutating),
        f"a mutating endpoint sits outside /api/: {mutating}",
    )

    # The published bundle is static by construction: nothing in it can spend
    # quota, and the refresh path exists only on the desk app.
    out = bundle()
    for path in out.rglob("*.js"):
        text = path.read_text()
        _require("api/refresh" not in text, f"{path.name} references the refresh endpoint")
    _require(
        "--no-publish" in source,
        "the desk refresh action can publish — a desk button must not ship a snapshot",
    )
    return f"binds 127.0.0.1; {len(mutating)} mutating endpoint(s), all desk-only"


# --------------------------------------------------------------------------
# 11. the mobile budget (guide §4)
# --------------------------------------------------------------------------


@check("registry_current", needs_data=False,
       description="every champion matches the code that trains it")
def check_registry_current() -> str:
    """A promoted experiment that was never retrained is invisible otherwise.

    ``size_model.FEATURES`` changed twice — EXP-111 added ``has_implied_quote``,
    EXP-113 removed ``abs_dist_high`` — and neither reached the registered
    artifact, because nothing retrained it. ``Scorer`` builds its feature matrix
    from ``artifact.features``, so the board scored on the pre-EXP-111 feature
    set for as long as that went unnoticed: the code said one thing, the shipped
    model did another, and every test passed throughout.

    Nothing here re-trains or judges quality. It asks only whether what is
    registered is what the current training code would produce.
    """
    import json

    from engine.models.registry import artifact_sha256, load_registry
    from engine.models.training import gate as gate_mod
    from engine.models.training import implied_t1 as implied_mod
    from engine.models.training import size_model as size_mod

    code = {
        "size": set(size_mod.FEATURES),
        "implied_t1": set(implied_mod.FEATURES),
        "gate": set(gate_mod.FEATURES),
    }
    registry_path = ROOT / "engine" / "models" / "registry.json"
    champions = [
        entry for entry in json.loads(registry_path.read_text())["models"]
        if entry.get("champion")
    ]
    _require(champions, "the registry lists no champions")

    registry = load_registry()
    problems: list[str] = []
    for entry in champions:
        role, registered = entry["role"], set(entry["features"])
        expected = code.get(role)
        if expected is None:
            problems.append(f"{entry['id']}: role {role!r} has no training module")
            continue
        if expected != registered:
            missing = sorted(expected - registered)
            extra = sorted(registered - expected)
            problems.append(
                f"{entry['id']}: features drifted from {role} training code — "
                f"never deployed {missing}, still deployed {extra}"
            )
        artifact = ROOT / entry["artifact"]
        if not artifact.exists():
            problems.append(f"{entry['id']}: artifact missing at {entry['artifact']}")
        elif artifact_sha256(artifact) != entry["artifact_sha256"]:
            problems.append(f"{entry['id']}: artifact sha does not match the registry")
        # The registry records the feature list; the ARTIFACT carries the one
        # actually used at scoring time. They can disagree, and the artifact wins.
        loaded = registry.load_champion(role, entry.get("strategy") or "*", verify=False)
        if set(loaded[1].features) != registered:
            problems.append(
                f"{entry['id']}: the artifact's own features differ from the "
                "registry entry — the registry is describing a model that is not "
                "the one being scored with"
            )

    _require(not problems, "registry is stale:\n  " + "\n  ".join(problems))
    return f"{len(champions)} champion(s) match their training code and their artifacts"


@check("board_budget", needs_data=False, description="board.json fits the mobile budget")
def check_board_budget() -> str:
    out = bundle()
    size = (out / "data" / "board.json").stat().st_size
    _require(
        size <= BOARD_MAX_BYTES,
        f"board.json is {size:,} bytes, over the {BOARD_MAX_BYTES:,} mobile budget",
    )
    tickers = list((out / "data" / "tickers").glob("*.json"))
    biggest = max((p.stat().st_size for p in tickers), default=0)
    meta = json.loads((out / "data" / "meta.json").read_text())
    _require(
        meta.get("board_oversized") is False,
        "meta.json reports the board oversized",
    )
    return (
        f"board.json {size:,}B of {BOARD_MAX_BYTES:,}B; "
        f"largest of {len(tickers)} ticker files {biggest:,}B (loaded lazily)"
    )


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

ORDER = [
    "unittests",
    "selfcheck",
    "offline_bundle",
    "atomicity",
    "validation_gate",
    "historical_dryrun",
    "forward_calendar",
    "secret_scan",
    "access_rule",
    "ui_no_compute",
    "calp_unvalidated",
    "server_local",
    "board_budget",
    "registry_current",
]


def run(names: list[str], *, skip_data: bool = False) -> list[CheckOutcome]:
    outcomes: list[CheckOutcome] = []
    for name in names:
        spec = REGISTRY[name]
        if skip_data and spec["needs_data"]:
            print(f"  {name:20s} SKIP (needs data)", flush=True)
            outcomes.append(CheckOutcome(name, True, "skipped", skipped=True))
            continue
        started = time.time()
        try:
            detail = spec["fn"]() or ""
            elapsed = time.time() - started
            print(f"  {name:20s} PASS  {detail} ({elapsed:.0f}s)", flush=True)
            outcomes.append(CheckOutcome(name, True, detail, elapsed))
        except Exception as exc:  # noqa: BLE001 — a check failure is a result
            elapsed = time.time() - started
            print(f"  {name:20s} FAIL  {exc} ({elapsed:.0f}s)", flush=True)
            outcomes.append(CheckOutcome(name, False, str(exc), elapsed))
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
    print(f"Phase 3 acceptance checks ({len(names)} checks)\n", flush=True)
    started = time.time()
    outcomes = run(names, skip_data=args.no_data)

    failed = [o for o in outcomes if not o.passed]
    skipped = [o for o in outcomes if o.skipped]
    print(f"\n{len(outcomes) - len(failed) - len(skipped)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped in {time.time()-started:.0f}s")
    if args.json:
        Path(args.json).write_text(
            json.dumps([o.__dict__ for o in outcomes], indent=1, default=str))
    if failed:
        print("\nFAILED:", file=sys.stderr)
        for outcome in failed:
            print(f"  {outcome.name}: {outcome.detail}", file=sys.stderr)
        return 1
    print("\nPHASE 3 CHECKS: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
