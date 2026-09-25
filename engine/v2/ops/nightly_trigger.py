"""Schedule the native shadow nightly behind an ORATS-finality gate.

Slice 12's scheduled path is deliberately narrow. The legacy nightly keeps its
own crontab line and this module never touches it; what is scheduled here is the
parallel shadow-mode qualification DAG that already exists (``ops plan nightly``
+ ``ops submit`` + the supervisor serve loop), gated on two facts:

* the as-of session is published at the provider -- a single lightweight ORATS
  market-wide probe (the summaries/cores pair legacy nightly step 1 already
  makes), never a refresh job; and
* no other heavy run holds ``<repo>/reports/.nightly.lock`` -- a NON-BLOCKING
  ``fcntl.flock`` (LOCK_EX|LOCK_NB) taken before the probe and held for the
  ENTIRE run (probe -> plan -> submit -> serve to terminal). A legacy nightly
  started meanwhile refuses to start, and a manual legacy invocation during a
  native run refuses the same way -- intended: only one heavy job at a time.
  The legacy cron time (21:30 weekdays) ends well before 00:00 ET, so holding
  the lock overnight never touches it.

The default as-of is the most recent completed trading session strictly before
the current ET calendar date (weekends and US market holidays skipped). Its
retry window is 00:00 ET through 06:00 ET on the calendar day after the as-of
(a Friday as-of is retried Saturday 00:00-06:00 and MISSED at Saturday 06:00),
with five minutes of grace so the 06:00 timer tick is still inside. A tick
whose pending as-of is already terminal exits 0 IDLE without probing or
writing anything.

Everything decision-shaped is pure logic plus injected seams -- the finality
provider, the plan call, the (idempotent) submit call and the serve driver --
so tests exercise every branch without a network or a catalog. The state file
(``reports/phase6/nightly_trigger/<as_of>.json``) is the idempotency record:
``submitting`` is written with the plan_ref BEFORE submit is called, so a crash
between submit and the state write resubmits that same plan_ref on the next
tick (never a re-plan), and submission of an already-submitted plan_ref is a
no-op by plan identity. A finished run is ``completed`` (every job succeeded)
or ``failed`` (terminal with failures). ``error`` becomes terminal as
``failed_setup`` after three consecutive errors for the same as-of.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from pathlib import Path
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

from engine.v2.foundation import SystemClock, format_timestamp
from engine.v2.ops.errors import OpsError, fail

__all__ = [
    "TriggerReceipt",
    "default_as_of",
    "full_population",
    "legacy_lock_path",
    "load_state",
    "main",
    "probe_finality",
    "repo_root",
    "run_trigger",
    "state_path",
    "write_state",
]

ET = ZoneInfo("America/New_York")
DEFAULT_WINDOW_START_ET = "00:00"
DEFAULT_DEADLINE_ET = "06:00"
#: The 06:00 tick may fire seconds late; this keeps it inside the window while
#: still missing a tick that arrives past it.
DEFAULT_DEADLINE_GRACE = timedelta(minutes=5)
#: ``error`` turns into the terminal ``failed_setup`` after this many in a row.
MAX_CONSECUTIVE_ERRORS = 3
STATE_DIR = ("reports", "phase6", "nightly_trigger")
#: Where the operator drops the native nightly's own qualification documents.
QUALIFICATION_INPUT_MANIFEST = "input_manifest.json"
QUALIFICATION_POPULATION = "expected_population.json"

STATUSES = ("submitted", "not_yet", "missed", "already_submitted", "busy_legacy", "error",
            "submitting", "completed", "failed", "failed_setup", "idle")
#: A terminal as-of never probes, never writes and never submits again.
TERMINAL_STATUSES = frozenset({"already_submitted", "completed", "failed", "missed",
                               "failed_setup"})
#: A recorded plan_ref means the decision is made: resume it, never re-plan.
RESUME_STATUSES = frozenset({"submitting", "submitted", "error"})
FAILURE_STATUSES = frozenset({"error", "failed", "failed_setup", "missed"})
SUCCESS_JOB_STATES = frozenset({"succeeded"})
TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "cancelled", "blocked"})
_HANDLED_FAILURES = (OpsError, OSError, ValueError, TypeError)

FinalityProvider = Callable[[str, Iterable[str]], "tuple[bool, str]"]
PlanCallable = Callable[..., str]
SubmitCallable = Callable[..., object]
ServeCallable = Callable[..., str]


@dataclass(frozen=True)
class TriggerReceipt:
    schema_version: str = "nightly_trigger.v1.0"
    as_of: str = ""
    status: str = ""
    detail: str = ""
    checked_at: str = ""
    plan_ref: str | None = None
    error_count: int = 0


class _LegacyLock:
    """A held, non-blocking flock on the legacy nightly's own lock file.

    Unlike a probe, the handle is NOT closed until the whole native run is
    done: while it is held, a legacy nightly (cron or manual) cannot take the
    same lock and refuses to start, and this trigger refuses while a legacy run
    holds it. An unopenable lock file is treated as held: fail closed, and the
    next timer tick retries.
    """

    def __init__(self, path: Path) -> None:
        self.path, self.handle = Path(path), None

    def acquire(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("a+")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.release()
            return False
        return True

    def release(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc_info) -> bool:
        self.release()
        return False


def repo_root() -> Path:
    """The code checkout's repo root, the way v2 ops code resolves it."""
    return Path(__file__).resolve().parents[3]


def legacy_lock_path(root: Path | None = None) -> Path:
    """``<repo>/reports/.nightly.lock`` -- the legacy nightly's own lock file."""
    return Path(root if root is not None else repo_root()) / "reports" / ".nightly.lock"


def state_path(root: Path, as_of: str) -> Path:
    """``reports/phase6/nightly_trigger/<as_of>.json`` -- the idempotency record."""
    return Path(root).joinpath(*STATE_DIR, f"{as_of}.json")


def load_state(root: Path, as_of: str) -> TriggerReceipt | None:
    """The recorded receipt for ``as_of``, or ``None`` for absent/corrupt state.

    A missing, unreadable, mistyped or mismatched-dated document is treated as
    no prior state -- never an error. The decision then runs normally and the
    fresh receipt replaces it.
    """
    try:
        document = json.loads(state_path(root, as_of).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("as_of") != as_of:
        return None
    status = document.get("status")
    if status not in STATUSES:
        return None
    plan_ref = document.get("plan_ref")
    error_count = document.get("error_count", 0)
    if not isinstance(error_count, int) or error_count < 0:
        error_count = 0
    return TriggerReceipt(
        schema_version=str(document.get("schema_version", TriggerReceipt.schema_version)),
        as_of=as_of, status=status, detail=str(document.get("detail", "")),
        checked_at=str(document.get("checked_at", "")),
        plan_ref=plan_ref if isinstance(plan_ref, str) else None,
        error_count=error_count)


def write_state(root: Path, receipt: TriggerReceipt) -> None:
    """Atomic write (tmp + ``os.replace``), matching the nightly's publish rule."""
    path = state_path(root, receipt.as_of)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(asdict(receipt), sort_keys=True))
    os.replace(tmp, path)


def probe_finality(as_of: str, tickers: Iterable[str], *,
                   provider: FinalityProvider | None = None) -> tuple[bool, str]:
    """Is the as-of session published yet? One lightweight provider read.

    ``provider`` is injectable and returns ``(is_final, detail)``; the
    production default is :func:`_orats_probe`, which wraps the native ORATS
    market-wide edge. Never runs a refresh job and never writes anything.
    """
    probe = provider or _orats_probe
    is_final, detail = probe(as_of, tuple(tickers))
    return bool(is_final), str(detail)


def _orats_probe(as_of: str, tickers: Iterable[str]) -> tuple[bool, str]:
    """The native provider classification path, read-only.

    ``orats_daily_market_fetcher`` is the same edge legacy nightly step 1 uses
    for the market-wide pull, so the provider response is classified by the
    existing incremental-data classifier (never a second ORATS parser). A
    published date yields complete coverage; an unpublished one (404 on both
    endpoints, or an empty 2xx) is the classifier's ``SOURCE_NOT_FINAL``. The
    fetched bytes are discarded -- only the verdict is kept.
    """
    from engine.v2.ops.providers import orats_daily_market_fetcher

    unit = {"request_id": "finality-probe:" + as_of, "table_name": "daily_market",
            "partition_key": as_of,
            "expected_keys": tuple(sorted({str(item) for item in tickers if item}))}
    try:
        orats_daily_market_fetcher()(unit)
    except OpsError as exc:
        if exc.code == "SOURCE_NOT_FINAL":
            return False, "ORATS has not published the as-of session yet"
        raise
    return True, "ORATS market-wide summaries and cores are published"


def default_as_of(clock=None) -> str:
    """The most recent completed trading session strictly before today in ET.

    Weekends and the scheduled US market holidays (``_is_trading_day``: the
    same pure rule the native nightly plan code already uses) are skipped; the
    current ET date itself is never the as-of, even on a trading day, because
    its session has not completed by the time the timer runs. ``clock`` is any
    ``.now()`` provider or a ``datetime``; naive datetimes are read as ET.
    """
    if clock is None:
        moment = SystemClock().now()
    elif isinstance(clock, datetime):
        moment = clock
    else:
        moment = clock.now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ET)
    day = moment.astimezone(ET).date() - timedelta(days=1)
    while not _is_trading_day(day):
        day -= timedelta(days=1)
    return day.isoformat()


def _is_trading_day(day: date) -> bool:
    """Is ``day`` a scheduled US market session? Weekends and NYSE holidays.

    ``legacy_adapter.projected_trading_sessions`` is the same pure
    weekday/US-market-holiday rule the native nightly plan code already uses
    (``capture_inputs._lookback_sessions``, ``health.trailing_occurrences``),
    reached through the package's one declared legacy adapter: this module
    never imports ``engine.calendar`` itself. ``projected_trading_days``' own
    ``(start, end]`` bound makes a one-day window the exact membership test.
    """
    if day.weekday() >= 5:
        return False
    from engine.v2.ops.legacy_adapter import projected_trading_sessions

    return day.isoformat() in projected_trading_sessions(day - timedelta(days=1), day)


def _boundary(value: str) -> clock_time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except (TypeError, ValueError):
        raise fail("INVALID_REQUEST", "ET window boundaries must be HH:MM") from None


def _validate_as_of(as_of: str) -> None:
    try:
        date.fromisoformat(as_of)
    except (TypeError, ValueError):
        raise fail("INVALID_REQUEST", "as_of must be an ISO date") from None


def _window(as_of: str, window_start_et: str, deadline_et: str) -> tuple[datetime, datetime]:
    """The one-morning retry window on the calendar day after the as-of.

    A Friday as-of is retried Saturday 00:00-06:00 and MISSED at Saturday
    06:00: the opening day is a calendar day, not a trading day, and the
    deadline is 06:00 ET on that same day (plus the caller's grace).
    """
    opened_on = date.fromisoformat(as_of) + timedelta(days=1)
    return (datetime.combine(opened_on, _boundary(window_start_et), tzinfo=ET),
            datetime.combine(opened_on, _boundary(deadline_et), tzinfo=ET))


def _receipt(clock, as_of: str, status: str, detail: str,
             plan_ref: str | None = None, error_count: int = 0) -> TriggerReceipt:
    return TriggerReceipt(as_of=as_of, status=status, detail=detail,
                          checked_at=format_timestamp(clock.now()), plan_ref=plan_ref,
                          error_count=error_count)


def _record(root: Path, receipt: TriggerReceipt) -> TriggerReceipt:
    write_state(root, receipt)
    return receipt


def _idle(clock, as_of: str, prior: TriggerReceipt) -> TriggerReceipt:
    return TriggerReceipt(as_of=as_of, status="idle",
                          detail=f"no pending as-of: {prior.status}",
                          checked_at=format_timestamp(clock.now()), plan_ref=prior.plan_ref,
                          error_count=prior.error_count)


def _problem_detail(exc: BaseException) -> str:
    if isinstance(exc, OpsError):
        return f"{exc.code}: {exc.problem.message}"
    return type(exc).__name__


def _failure(root: Path, clock, as_of: str, plan_ref: str | None,
             exc: BaseException, prior: TriggerReceipt | None) -> TriggerReceipt:
    detail = _problem_detail(exc)
    previous = prior.error_count if prior is not None and prior.status == "error" else 0
    count = previous + 1
    if count >= MAX_CONSECUTIVE_ERRORS:
        return _record(root, _receipt(
            clock, as_of, "failed_setup",
            f"setup failed {count} consecutive times; giving up: {detail}",
            plan_ref=plan_ref, error_count=count))
    return _record(root, _receipt(clock, as_of, "error", detail,
                                  plan_ref=plan_ref, error_count=count))


def run_trigger(root: Path, as_of: str, *, tickers: Iterable[str] = (),
                context_tickers: Iterable[str] = (),
                deadline_et: str = DEFAULT_DEADLINE_ET,
                window_start_et: str = DEFAULT_WINDOW_START_ET,
                provider: FinalityProvider | None = None, clock=None,
                plan_fn: PlanCallable | None = None,
                submit_fn: SubmitCallable | None = None,
                serve_fn: ServeCallable | None = None,
                full_run: bool = True) -> TriggerReceipt:
    """The whole tick: terminal -> lock -> resume -> window -> probe -> submit -> serve.

    ``tickers``/``context_tickers`` are the plan's watchlist and historical
    evidence universe (``full_population`` derives both from the native
    nightly plan's own population document in production). ``plan_fn``,
    ``submit_fn``, ``serve_fn`` and ``provider`` are injected seams; the
    production defaults are the real in-process plan/submit/serve and the
    native ORATS probe.
    """
    clock = clock or SystemClock()
    root = Path(root)
    _validate_as_of(as_of)
    prior = load_state(root, as_of)
    if prior is not None and prior.status in TERMINAL_STATUSES:
        return _idle(clock, as_of, prior)
    with _LegacyLock(legacy_lock_path(root)) as held:
        if not held:
            return _record(root, _receipt(
                clock, as_of, "busy_legacy",
                "another heavy run holds the legacy nightly lock; retrying next tick"))
        if prior is not None and prior.plan_ref and prior.status in RESUME_STATUSES:
            return _submit_plan(root, as_of, tickers=(), context_tickers=(), clock=clock,
                                plan_fn=None, submit_fn=submit_fn, serve_fn=serve_fn,
                                full_run=full_run, prior=prior, plan_ref=prior.plan_ref)
        return _decide(root, as_of, tickers=tickers, context_tickers=context_tickers,
                       deadline_et=deadline_et, window_start_et=window_start_et,
                       provider=provider, clock=clock, plan_fn=plan_fn, submit_fn=submit_fn,
                       serve_fn=serve_fn, full_run=full_run, prior=prior)


def _decide(root: Path, as_of: str, *, tickers, context_tickers, deadline_et, window_start_et,
            provider, clock, plan_fn, submit_fn, serve_fn, full_run,
            prior: TriggerReceipt | None) -> TriggerReceipt:
    opened, deadline = _window(as_of, window_start_et, deadline_et)
    now_et = clock.now().astimezone(ET)
    if now_et < opened:
        return _receipt(clock, as_of, "not_yet", "before the retry window opens")
    if now_et > deadline + DEFAULT_DEADLINE_GRACE:
        return _record(root, _receipt(
            clock, as_of, "missed", "the retry window closed before the session was final"))
    is_final, detail = probe_finality(as_of, tickers, provider=provider)
    if not is_final:
        return _record(root, _receipt(clock, as_of, "not_yet", detail))
    return _submit_plan(root, as_of, tickers=tickers, context_tickers=context_tickers,
                        clock=clock, plan_fn=plan_fn, submit_fn=submit_fn, serve_fn=serve_fn,
                        full_run=full_run, prior=prior, plan_ref=None)


def _submit_plan(root: Path, as_of: str, *, tickers, context_tickers, clock, plan_fn,
                 submit_fn, serve_fn, full_run, prior: TriggerReceipt | None,
                 plan_ref: str | None) -> TriggerReceipt:
    """Plan (unless resuming), write ``submitting``, submit, then serve."""
    plan_fn = plan_fn or _default_plan
    submit_fn = submit_fn or _default_submit
    serve_fn = serve_fn or _default_serve
    if plan_ref is None:
        try:
            plan_ref = plan_fn(root, as_of, tuple(tickers), tuple(context_tickers), clock,
                               full_run=full_run)
        except _HANDLED_FAILURES as exc:
            return _failure(root, clock, as_of, None, exc, prior)
        _record(root, _receipt(clock, as_of, "submitting",
                               "the plan is saved; submitting it", plan_ref=plan_ref))
    try:
        submit_fn(root, as_of, plan_ref, clock)
    except _HANDLED_FAILURES as exc:
        return _failure(root, clock, as_of, plan_ref, exc, prior)
    _record(root, _receipt(clock, as_of, "submitted",
                           "jobs submitted; running the plan to terminal", plan_ref=plan_ref))
    try:
        final = serve_fn(root, plan_ref, clock)
    except _HANDLED_FAILURES as exc:
        return _failure(root, clock, as_of, plan_ref, exc, prior)
    status = "completed" if str(final) == "completed" else "failed"
    return _record(root, _receipt(clock, as_of, status,
                                  f"the submitted plan finished {status}", plan_ref=plan_ref))


def _read_document(path: Path | None):
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _qualification_path(root: Path, name: str) -> Path | None:
    path = Path(root).joinpath(*STATE_DIR, name)
    return path if path.is_file() else None


def _population_tickers(path: Path | None) -> tuple[str, ...]:
    document = _read_document(path)
    if not isinstance(document, list):
        return ()
    return tuple(sorted({str(key).split("|")[0] for key in document
                         if isinstance(key, str) and "|" in key and key.split("|")[0]}))


def full_population(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The scheduled run's full default population ticker universe.

    The native nightly plan's own population document
    (``reports/phase6/nightly_trigger/expected_population.json``, a JSON list
    of ``ticker|strategy|event_date`` keys) IS the full default population:
    every distinct ticker in it is both the watchlist and the historical
    evidence context, and the document is passed to the plan unchanged as
    ``--expected-population``. The trigger takes no ticker-list argument, and
    the document is not required for the trigger to run: without it the plan
    still declares ``full_run=True`` and ``ops submit`` records its own
    planned-population refusal as an ``error`` receipt rather than the trigger
    silently scoring nothing.
    """
    tickers = _population_tickers(_qualification_path(root, QUALIFICATION_POPULATION))
    return tickers, tickers


def _ops_root(root: Path) -> Path:
    return Path(root) / "data" / "operations"


def _default_plan(root: Path, as_of: str, tickers=(), context_tickers=(), clock=None, *,
                  full_run: bool = True) -> str:
    """The production plan: the real ``cli._plan_command``, in-process.

    ``full_run=True`` is the native nightly plan's full-population option and
    is always declared for a scheduled run, so the plan's effect scope is the
    global ``shadow`` scope, never a slice. ``expected_population`` is the
    operator's population document when present (``full_population`` derived
    the universe from the same file); absent, the plan still carries the
    full-run declaration and the refusal is ``ops submit``'s to make.
    """
    from engine.v2.foundation import ensure_directory
    from engine.v2.ops import cli
    from engine.v2.ops.bootstrap import open_catalog

    clock = clock or SystemClock()
    ops_root = _ops_root(root)
    ensure_directory(ops_root)
    population = _qualification_path(root, QUALIFICATION_POPULATION)
    universe = tuple(tickers) or _population_tickers(population)
    context = tuple(context_tickers) or universe
    plan_args = argparse.Namespace(
        command="plan", kind="nightly", as_of=as_of, mode="shadow", spec=None,
        no_ledger=False,
        input_manifest=_qualification_path(root, QUALIFICATION_INPUT_MANIFEST),
        expected_population=population,
        tickers=",".join(universe), context_tickers=",".join(context),
        full_run=bool(full_run), year_start=2024, year_end=2026, input_mode="legacy",
        snapshot_scope=None, refresh_mode="legacy", refresh_plan=None)
    conn = open_catalog(ops_root / "catalog.sqlite", clock=clock)
    try:
        planned = cli._plan_command(plan_args, ops_root, conn, clock)
    finally:
        conn.close()
    return str(planned["plan_ref"])


def _default_submit(root: Path, as_of: str, plan_ref: str | None, clock) -> object:
    """The production submit: ``cli._submit_command``, in-process.

    Nightly job identity comes from the plan document itself, so submitting
    the SAME ``plan_ref`` again is a no-op (``submission._insert_or_match``
    matches the existing rows by request digest and inserts nothing) -- the
    property the ``submitting`` state relies on after a crash between submit
    and the state write.
    """
    from engine.v2.ops import cli
    from engine.v2.ops.bootstrap import open_catalog

    if not plan_ref:
        raise fail("INVALID_REQUEST", "the trigger has no plan_ref to submit")
    ops_root = _ops_root(root)
    conn = open_catalog(ops_root / "catalog.sqlite", clock=clock)
    try:
        return cli._submit_command(
            argparse.Namespace(plan=plan_ref, idempotency_key="nightly-" + as_of),
            ops_root, conn, clock)
    finally:
        conn.close()


def _default_serve(root: Path, plan_ref: str, clock) -> str:
    """Drive the submitted plan to terminal with the supervisor's own loop.

    ``supervisor.serve`` is the exact entry ``ops serve`` uses; the trigger
    owns the process for the duration (while holding the legacy lock), so the
    native DAG runs in-process here instead of needing a second supervisor.
    The job set is resolved through the idempotent ``cli._submit_command`` (a
    no-op resubmission), then polled until every job is terminal. The final
    status is ``completed`` only when every job succeeded, else ``failed``.
    """
    from engine.v2.ops import cli
    from engine.v2.ops.bootstrap import open_catalog
    from engine.v2.ops.profiles import DEFAULT_POLICY
    from engine.v2.ops.stages import registry
    from engine.v2.ops.supervisor import Service, serve

    ops_root = _ops_root(root)
    conn = open_catalog(ops_root / "catalog.sqlite", clock=clock)
    try:
        submitted = cli._submit_command(
            argparse.Namespace(plan=plan_ref, idempotency_key="nightly-serve"),
            ops_root, conn, clock)
        rows = submitted.get("jobs", ()) if isinstance(submitted, dict) else ()
        job_ids = tuple(str(row["job_id"]) for row in rows
                        if isinstance(row, dict) and row.get("job_id"))
        if not job_ids:
            raise fail("INVALID_REQUEST", "the submitted plan produced no jobs")
        service = Service(conn, ops_root, registry(), DEFAULT_POLICY, clock=clock,
                          code_source=repo_root())
        serve(service, until=lambda: _jobs_terminal(conn, job_ids))
        return "completed" if _jobs_succeeded(conn, job_ids) else "failed"
    finally:
        conn.close()


def _job_states(conn, job_ids) -> tuple[str, ...]:
    if not job_ids:
        return ()
    placeholders = ", ".join("?" for _ in job_ids)
    rows = conn.execute(f"SELECT state FROM jobs WHERE job_id IN ({placeholders})",
                        tuple(job_ids)).fetchall()
    return tuple(str(row["state"]) for row in rows)


def _jobs_terminal(conn, job_ids) -> bool:
    states = _job_states(conn, job_ids)
    return bool(states) and all(state in TERMINAL_JOB_STATES for state in states)


def _jobs_succeeded(conn, job_ids) -> bool:
    states = _job_states(conn, job_ids)
    return bool(states) and all(state in SUCCESS_JOB_STATES for state in states)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--as-of", default=None,
                        help="YYYY-MM-DD; defaults to the most recent completed trading "
                             "session strictly before today in America/New_York")
    parser.add_argument("--root", default=".",
                        help="the root the state file and reports/ live under (default: .)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    clock = SystemClock()
    root = Path(args.root).resolve()
    as_of = args.as_of or default_as_of(clock)
    tickers, context_tickers = full_population(root)
    try:
        receipt = run_trigger(root, as_of, tickers=tickers, context_tickers=context_tickers,
                              deadline_et=DEFAULT_DEADLINE_ET,
                              window_start_et=DEFAULT_WINDOW_START_ET, clock=clock,
                              full_run=True)
    except OpsError as exc:
        print(json.dumps({"code": exc.code, "message": exc.problem.message}, sort_keys=True))
        return 1
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"code": "INVALID_REQUEST",
                          "message": "the trigger could not read or write its inputs",
                          "details": {"exception_type": type(exc).__name__}}, sort_keys=True))
        return 1
    print(json.dumps(asdict(receipt), sort_keys=True))
    return 1 if receipt.status in FAILURE_STATUSES else 0


if __name__ == "__main__":
    raise SystemExit(main())
