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
from typing import Callable, Iterable, Literal
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
#: An ABSOLUTE ET cutoff, on whatever calendar day ``serve`` is called (never a duration from
#: when this particular call started, which a resumed ``timed_out`` serve would otherwise get
#: fresh, reaching arbitrarily late into the day) -- OUR OWN judgment call (not measured or
#: externally specified): 1.5 hours of margin before the legacy cron's 21:30 ET window.
DEFAULT_SERVE_DEADLINE_ET = "20:00"
STATE_DIR = ("reports", "phase6", "nightly_trigger")
#: Where the operator drops the native nightly's own qualification documents.
QUALIFICATION_INPUT_MANIFEST = "input_manifest.json"
QUALIFICATION_POPULATION = "expected_population.json"

STATUSES = ("submitted", "not_yet", "missed", "already_submitted", "busy_legacy", "error",
            "submitting", "completed", "failed", "failed_setup", "idle", "timed_out")
#: A terminal as-of never probes, never writes and never submits again.
TERMINAL_STATUSES = frozenset({"already_submitted", "completed", "failed", "missed",
                               "failed_setup"})
#: A recorded plan_ref means the decision is made: resume it, never re-plan.
RESUME_STATUSES = frozenset({"submitting", "submitted", "error", "timed_out"})
FAILURE_STATUSES = frozenset({"error", "failed", "failed_setup", "missed", "timed_out"})
SUCCESS_JOB_STATES = frozenset({"succeeded"})
TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "cancelled", "blocked"})
_HANDLED_FAILURES = (OpsError, OSError, ValueError, TypeError)

FinalityProvider = Callable[[str, Iterable[str]], "tuple[bool, str]"]
PlanCallable = Callable[..., str]
SubmitCallable = Callable[..., object]
ServeCallable = Callable[..., str]
EnsureSnapshotCallable = Callable[..., "tuple[str, str | None]"]
#: ``_drive_jobs_to_terminal``'s three-way outcome, shared by ``_default_serve``
#: and ``_ensure_shadow_snapshot``'s snapshot-import drive (Cutover PR-7b).
ServeOutcome = Literal["timed_out", "completed", "failed"]


@dataclass(frozen=True)
class TriggerReceipt:
    schema_version: str = "nightly_trigger.v1.0"
    as_of: str = ""
    status: str = ""
    detail: str = ""
    checked_at: str = ""
    plan_ref: str | None = None
    error_count: int = 0
    snapshot_attempt: int = 0


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
    snapshot_attempt = document.get("snapshot_attempt", 0)
    if not isinstance(snapshot_attempt, int) or snapshot_attempt < 0:
        snapshot_attempt = 0
    return TriggerReceipt(
        schema_version=str(document.get("schema_version", TriggerReceipt.schema_version)),
        as_of=as_of, status=status, detail=str(document.get("detail", "")),
        checked_at=str(document.get("checked_at", "")),
        plan_ref=plan_ref if isinstance(plan_ref, str) else None,
        error_count=error_count, snapshot_attempt=snapshot_attempt)


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


#: Mirrors ``engine/dashboard/nightly.py``'s own scoring horizon exactly (``HORIZON_DAYS = 35``,
#: ``context_years = range(as_of.year - 1, horizon.year + 1)``) so the native plan's context
#: window tracks the same calendar legacy's does, with no fixed end date to age past.
_LEGACY_HORIZON_DAYS = 35


def _derive_years(as_of: str) -> tuple[int, int]:
    """``(year_start, year_end)`` for ``as_of``, matching legacy's own
    ``context_years = range(as_of.year - 1, horizon.year + 1)`` formula
    (``engine/dashboard/nightly.py``) -- computed fresh from ``as_of`` every
    call, never a fixed pair.
    """
    session = date.fromisoformat(as_of)
    horizon = session + timedelta(days=_LEGACY_HORIZON_DAYS)
    return session.year - 1, horizon.year


def _serve_deadline(clock) -> datetime:
    """Today's ET calendar-day cutoff (``DEFAULT_SERVE_DEADLINE_ET``), from ``clock.now()``'s
    OWN date -- never the plan's ``as_of`` and never a duration from when this call started.
    Every call on the same calendar day (the first serve and every resumed one) therefore
    computes the IDENTICAL absolute cutoff, so no number of same-day resumes can push serving
    past it; a call made after the cutoff has already passed returns a ``deadline_at`` already
    in the past, so ``serve`` stops on its very first tick rather than running another cycle.
    """
    today_et = clock.now().astimezone(ET).date()
    return datetime.combine(today_et, _boundary(DEFAULT_SERVE_DEADLINE_ET), tzinfo=ET)


def _receipt(clock, as_of: str, status: str, detail: str,
             plan_ref: str | None = None, error_count: int = 0,
             snapshot_attempt: int = 0) -> TriggerReceipt:
    return TriggerReceipt(as_of=as_of, status=status, detail=detail,
                          checked_at=format_timestamp(clock.now()), plan_ref=plan_ref,
                          error_count=error_count, snapshot_attempt=snapshot_attempt)


def _record(root: Path, receipt: TriggerReceipt) -> TriggerReceipt:
    write_state(root, receipt)
    return receipt


def _idle(clock, as_of: str, prior: TriggerReceipt) -> TriggerReceipt:
    return TriggerReceipt(as_of=as_of, status="idle",
                          detail=f"no pending as-of: {prior.status}",
                          checked_at=format_timestamp(clock.now()), plan_ref=prior.plan_ref,
                          error_count=prior.error_count, snapshot_attempt=prior.snapshot_attempt)


def _problem_detail(exc: BaseException) -> str:
    if isinstance(exc, OpsError):
        return f"{exc.code}: {exc.problem.message}"
    return type(exc).__name__


def _failure(root: Path, clock, as_of: str, plan_ref: str | None,
             exc: BaseException, prior: TriggerReceipt | None, *,
             snapshot_attempt: int | None = None) -> TriggerReceipt:
    detail = _problem_detail(exc)
    previous = prior.error_count if prior is not None and prior.status == "error" else 0
    count = previous + 1
    resolved_snapshot_attempt = (
        snapshot_attempt if snapshot_attempt is not None
        else (prior.snapshot_attempt if prior is not None else 0))
    if count >= MAX_CONSECUTIVE_ERRORS or resolved_snapshot_attempt >= MAX_CONSECUTIVE_ERRORS:
        giving_up_count = count if count >= MAX_CONSECUTIVE_ERRORS else resolved_snapshot_attempt
        return _record(root, _receipt(
            clock, as_of, "failed_setup",
            f"setup failed {giving_up_count} consecutive times; giving up: {detail}",
            plan_ref=plan_ref, error_count=count, snapshot_attempt=resolved_snapshot_attempt))
    return _record(root, _receipt(clock, as_of, "error", detail,
                                  plan_ref=plan_ref, error_count=count,
                                  snapshot_attempt=resolved_snapshot_attempt))


def run_trigger(root: Path, as_of: str, *, tickers: Iterable[str] = (),
                context_tickers: Iterable[str] = (),
                deadline_et: str = DEFAULT_DEADLINE_ET,
                window_start_et: str = DEFAULT_WINDOW_START_ET,
                provider: FinalityProvider | None = None, clock=None,
                plan_fn: PlanCallable | None = None,
                submit_fn: SubmitCallable | None = None,
                serve_fn: ServeCallable | None = None,
                ensure_snapshot_fn: EnsureSnapshotCallable | None = None,
                full_run: bool = True) -> TriggerReceipt:
    """The whole tick: terminal -> resume -> lock -> window -> probe -> submit -> serve.

    The resume check runs before the legacy lock is attempted, so a busy lock
    can never overwrite a resumable state's plan_ref.

    ``tickers``/``context_tickers`` are the plan's watchlist and historical
    evidence universe (``full_population`` derives both from the native
    nightly plan's own population document in production). ``plan_fn``,
    ``submit_fn``, ``serve_fn``, ``ensure_snapshot_fn`` and ``provider`` are
    injected seams; the production defaults are the real in-process
    plan/submit/serve and the native ORATS probe.
    """
    clock = clock or SystemClock()
    root = Path(root)
    _validate_as_of(as_of)
    prior = load_state(root, as_of)
    if prior is not None and prior.status in TERMINAL_STATUSES:
        return _idle(clock, as_of, prior)
    resuming = prior is not None and bool(prior.plan_ref) and prior.status in RESUME_STATUSES
    with _LegacyLock(legacy_lock_path(root)) as held:
        if not held:
            if resuming:
                # issue #102: a resumable state must never be overwritten by a busy-lock tick.
                # This receipt is for THIS tick's own visibility only -- write_state is never
                # called, so the durable state is untouched and the next tick loads the SAME
                # prior state again, exactly as if this busy tick had not happened.
                return _receipt(
                    clock, as_of, "busy_legacy",
                    "another heavy run holds the legacy nightly lock; retrying the resume next tick",
                    plan_ref=prior.plan_ref, error_count=prior.error_count,
                    snapshot_attempt=prior.snapshot_attempt)
            return _record(root, _receipt(
                clock, as_of, "busy_legacy",
                "another heavy run holds the legacy nightly lock; retrying next tick",
                snapshot_attempt=prior.snapshot_attempt if prior is not None else 0))
        if resuming:
            return _submit_plan(root, as_of, tickers=(), context_tickers=(), clock=clock,
                                plan_fn=None, submit_fn=submit_fn, serve_fn=serve_fn,
                                ensure_snapshot_fn=ensure_snapshot_fn,
                                full_run=full_run, prior=prior, plan_ref=prior.plan_ref)
        return _decide(root, as_of, tickers=tickers, context_tickers=context_tickers,
                       deadline_et=deadline_et, window_start_et=window_start_et,
                       provider=provider, clock=clock, plan_fn=plan_fn, submit_fn=submit_fn,
                       serve_fn=serve_fn, ensure_snapshot_fn=ensure_snapshot_fn,
                       full_run=full_run, prior=prior)


def _decide(root: Path, as_of: str, *, tickers, context_tickers, deadline_et, window_start_et,
            provider, clock, plan_fn, submit_fn, serve_fn, ensure_snapshot_fn, full_run,
            prior: TriggerReceipt | None) -> TriggerReceipt:
    opened, deadline = _window(as_of, window_start_et, deadline_et)
    now_et = clock.now().astimezone(ET)
    prior_snapshot_attempt = prior.snapshot_attempt if prior is not None else 0
    if now_et < opened:
        return _receipt(clock, as_of, "not_yet", "before the retry window opens",
                        snapshot_attempt=prior_snapshot_attempt)
    if now_et > deadline + DEFAULT_DEADLINE_GRACE:
        return _record(root, _receipt(
            clock, as_of, "missed", "the retry window closed before the session was final",
            snapshot_attempt=prior_snapshot_attempt))
    is_final, detail = probe_finality(as_of, tickers, provider=provider)
    if not is_final:
        return _record(root, _receipt(clock, as_of, "not_yet", detail,
                                      snapshot_attempt=prior_snapshot_attempt))
    return _submit_plan(root, as_of, tickers=tickers, context_tickers=context_tickers,
                        clock=clock, plan_fn=plan_fn, submit_fn=submit_fn, serve_fn=serve_fn,
                        ensure_snapshot_fn=ensure_snapshot_fn,
                        full_run=full_run, prior=prior, plan_ref=None)


def _timeout_receipt(clock, as_of: str, plan_ref: str | None, snapshot_attempt: int,
                     prior: TriggerReceipt | None, *, timed_out_detail: str,
                     give_up_detail: str) -> TriggerReceipt:
    """The shared ``timed_out`` counting behind both of ``_submit_plan``'s
    timeout arms (pre-plan snapshot import, post-plan serve): the same
    consecutive-timeout give-up rule, differing only in its detail texts.
    """
    previous = prior.error_count if prior is not None and prior.status == "timed_out" else 0
    count = previous + 1
    if count >= MAX_CONSECUTIVE_ERRORS:
        return _receipt(clock, as_of, "failed", give_up_detail.format(count=count),
                        plan_ref=plan_ref, error_count=count,
                        snapshot_attempt=snapshot_attempt)
    return _receipt(clock, as_of, "timed_out", timed_out_detail,
                    plan_ref=plan_ref, error_count=count, snapshot_attempt=snapshot_attempt)


def _snapshot_attempt_bump(exc: BaseException) -> int:
    """Only a terminal ``INPUT_CHANGED`` refusal from ``ensure_snapshot_fn``
    mints a new idempotency key (CodeRabbit round 1); a transient failure
    (e.g. a catalog-I/O ``OSError``) must not, since the import job already
    submitted under the OLD attempt's key may still be running or already
    have succeeded there."""
    return 1 if isinstance(exc, OpsError) and exc.code == "INPUT_CHANGED" else 0


def _submit_plan(root: Path, as_of: str, *, tickers, context_tickers, clock, plan_fn,
                 submit_fn, serve_fn, ensure_snapshot_fn, full_run,
                 prior: TriggerReceipt | None, plan_ref: str | None) -> TriggerReceipt:
    """Plan (unless resuming), write ``submitting``, submit, then serve.

    ``snapshot_attempt`` (Cutover PR-7b) is a single, monotonic counter for the
    WHOLE ``as_of`` run, carried on every receipt below regardless of status and
    bumped in exactly one place -- the ``ensure_snapshot_fn`` failure branch --
    never reused from ``error_count``, which resets on any non-``"error"``
    status this function (and the rest of the module) already has several of.
    """
    plan_fn = plan_fn or _default_plan
    submit_fn = submit_fn or _default_submit
    serve_fn = serve_fn or _default_serve
    ensure_snapshot_fn = ensure_snapshot_fn or _ensure_shadow_snapshot
    snapshot_attempt = prior.snapshot_attempt if prior is not None else 0
    if plan_ref is None:
        try:
            readiness, snapshot_id = ensure_snapshot_fn(root, as_of, clock, snapshot_attempt)
        except _HANDLED_FAILURES as exc:
            return _failure(root, clock, as_of, None, exc, prior,
                            snapshot_attempt=snapshot_attempt + _snapshot_attempt_bump(exc))
        if readiness == "not_yet":
            return _record(root, _receipt(
                clock, as_of, "not_yet", "the shadow snapshot has not caught up to as_of yet",
                snapshot_attempt=snapshot_attempt))
        if readiness == "timed_out":
            return _record(root, _timeout_receipt(
                clock, as_of, None, snapshot_attempt, prior,
                timed_out_detail="the shadow snapshot import has not finished; the legacy lock "
                                 "is released, resuming next tick",
                give_up_detail="the shadow snapshot import exceeded its deadline {count} "
                               "consecutive times; giving up"))
        try:
            plan_ref = plan_fn(root, as_of, tuple(tickers), tuple(context_tickers), clock,
                               full_run=full_run, expected_shadow_snapshot_id=snapshot_id)
        except _HANDLED_FAILURES as exc:
            return _failure(root, clock, as_of, None, exc, prior)
        _record(root, _receipt(clock, as_of, "submitting",
                               "the plan is saved; submitting it", plan_ref=plan_ref,
                               snapshot_attempt=snapshot_attempt))
    try:
        submit_fn(root, as_of, plan_ref, clock)
    except _HANDLED_FAILURES as exc:
        return _failure(root, clock, as_of, plan_ref, exc, prior)
    _record(root, _receipt(clock, as_of, "submitted",
                           "jobs submitted; running the plan to terminal", plan_ref=plan_ref,
                           snapshot_attempt=snapshot_attempt))
    try:
        final = serve_fn(root, plan_ref, clock)
    except _HANDLED_FAILURES as exc:
        return _failure(root, clock, as_of, plan_ref, exc, prior)
    if final == "timed_out":
        return _record(root, _timeout_receipt(
            clock, as_of, plan_ref, snapshot_attempt, prior,
            timed_out_detail="serve exceeded its deadline; the legacy lock is released, "
                             "resuming next tick",
            give_up_detail="serve exceeded its deadline {count} consecutive times; giving up"))
    status = "completed" if str(final) == "completed" else "failed"
    return _record(root, _receipt(clock, as_of, status,
                                  f"the submitted plan finished {status}", plan_ref=plan_ref,
                                  snapshot_attempt=snapshot_attempt))


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


def _capture_input_manifest(root: Path, as_of: str, tickers: tuple[str, ...],
                            context_tickers: tuple[str, ...], year_start: int, year_end: int,
                            *, capture_fn=None) -> Path:
    """Capture THIS call's own legacy input manifest for ``as_of`` (issue #104)
    and write it to a per-``as_of`` path -- never the one static, shared
    filename every prior night also wrote (or read stale). ``capture_fn``
    defaults to ``capture_inputs.capture`` and exists only so a test can
    substitute a stub without touching the real legacy store.

    Raises the typed, non-retryable ``INPUT_CHANGED`` (the same code
    ``store_barrier.py`` already uses for this family of failure) if the
    captured manifest's own ``selected_session`` does not match ``as_of`` --
    defensive insurance against a manifest silently pinned to the wrong
    session (issue #104's second failure scenario), even though
    ``capture_inputs.capture`` derives ``selected_session`` from this same
    ``as_of`` today and so cannot currently disagree with it on its own.
    """
    from engine.v2.ops.capture_inputs import capture, write_manifest

    capture_fn = capture_fn or capture
    manifest = capture_fn(root, as_of=as_of, tickers=tickers, context_tickers=context_tickers,
                          year_start=year_start, year_end=year_end)
    if manifest.selected_session != as_of:
        raise fail("INPUT_CHANGED",
                   "captured input manifest is pinned to a different session than the plan",
                   details={"as_of": as_of, "selected_session": manifest.selected_session})
    output = Path(root).joinpath(*STATE_DIR, f"{as_of}.input_manifest.json")
    return write_manifest(manifest, output)


def _default_plan(root: Path, as_of: str, tickers=(), context_tickers=(), clock=None, *,
                  full_run: bool = True,
                  expected_shadow_snapshot_id: str | None = None) -> str:
    """The production plan: the real ``cli._plan_command``, in-process.

    ``full_run=True`` is the native nightly plan's full-population option and
    is always declared for a scheduled run, so the plan's effect scope is the
    global ``shadow`` scope, never a slice. ``input_mode`` is always
    ``"snapshot"`` and ``snapshot_scope`` is always ``"shadow"`` (Cutover
    PR-7b): the shadow nightly's plan always reads a pinned snapshot of the
    ``shadow`` scope, never the legacy live store directly.
    ``expected_shadow_snapshot_id`` is the exact snapshot id ``_submit_plan``'s
    ``ensure_snapshot_fn`` call just verified is fresh for ``as_of`` (``None``
    for any caller outside that path, e.g. a direct `ops plan` invocation) --
    threaded through as ``args.expected_snapshot_id``, not yet read by
    ``cli._plan_command``/``pin_snapshot_inputs`` (PR-7b-3 adds that CAS check;
    this slice only threads the value through). ``expected_population`` is the
    operator's population document when present (``full_population`` derived
    the universe from the same file); absent, the plan still carries the
    full-run declaration and the refusal is ``ops submit``'s to make.
    ``input_manifest`` and ``year_start``/``year_end`` are derived fresh from
    ``as_of`` on every call (the manifest is captured and written per-``as_of``
    rather than read from one fixed filename, and the year span mirrors
    legacy's own scoring horizon) rather than being fixed constants.
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
    year_start, year_end = _derive_years(as_of)
    # issue #104: capture this call's own manifest only when there is a universe to capture
    # against -- with none, there is nothing for capture_inputs.capture to enumerate, and the
    # plan carries input_manifest=None exactly as it did before this change in that case.
    manifest_path = (_capture_input_manifest(root, as_of, universe, context, year_start, year_end)
                     if universe else None)
    plan_args = argparse.Namespace(
        command="plan", kind="nightly", as_of=as_of, mode="shadow", spec=None,
        no_ledger=False,
        input_manifest=manifest_path,
        expected_population=population,
        tickers=",".join(universe), context_tickers=",".join(context),
        full_run=bool(full_run), year_start=year_start, year_end=year_end,
        input_mode="snapshot", snapshot_scope="shadow", refresh_mode="legacy",
        refresh_plan=None, expected_snapshot_id=expected_shadow_snapshot_id)
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
    no-op resubmission), then polled until every job is terminal via
    :func:`_drive_jobs_to_terminal` (shared with
    ``_ensure_shadow_snapshot``'s own drive-to-terminal step, Cutover PR-7b, so
    there is exactly one polling loop in this module). ``serve`` is bounded by
    ``_serve_deadline``/``DEFAULT_SERVE_DEADLINE_ET`` (an absolute same-day ET
    cutoff, not a duration) and this returns ``"timed_out"`` if that deadline
    fires before every job is terminal, so a wedged job never blocks
    indefinitely. The final status is ``completed`` only when every job
    succeeded, else ``failed``.
    """
    from engine.v2.ops import cli
    from engine.v2.ops.bootstrap import open_catalog
    from engine.v2.ops.profiles import DEFAULT_POLICY
    from engine.v2.ops.stages import registry
    from engine.v2.ops.supervisor import Service

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
        return _drive_jobs_to_terminal(service, conn, job_ids, _serve_deadline(clock))
    finally:
        conn.close()


def _drive_jobs_to_terminal(service, conn, job_ids,
                            deadline_at) -> ServeOutcome:
    """Poll ``service`` until every job in ``job_ids`` is terminal or
    ``deadline_at`` fires. Returns ``"timed_out"`` on ``serve``'s own
    ``"deadline_exceeded"``, else ``"completed"`` when every job succeeded,
    else ``"failed"``. Shared by ``_default_serve`` and
    ``_ensure_shadow_snapshot`` (Cutover PR-7b) so there is exactly one
    polling loop in this module.
    """
    from engine.v2.ops.supervisor import serve

    outcome = serve(service, until=lambda: _jobs_terminal(conn, job_ids),
                    deadline_at=deadline_at)
    if outcome == "deadline_exceeded":
        return "timed_out"
    return "completed" if _jobs_succeeded(conn, job_ids) else "failed"


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


def _default_plan_import(root, expected_head_snapshot_id, expected_head_generation):
    """The production plan-import: the same ``engine.v2.data.import_snapshot.
    plan_import`` call ``ops snapshot plan-import`` makes (``cli.
    snapshot_command``), scoped to ``"shadow"`` -- the production default for
    ``_ensure_shadow_snapshot``'s ``plan_import_fn`` seam (Cutover PR-7b).
    """
    from engine.v2.data.import_snapshot import plan_import

    return plan_import(source_root=root, scope="shadow",
                       expected_head_snapshot_id=expected_head_snapshot_id,
                       expected_head_generation=expected_head_generation)


def _default_submit_import(root, conn, plan, idempotency_key, clock) -> str:
    """Publish and submit ``plan`` through the real ``ops snapshot plan-
    import``/``submit`` path (``cli.snapshot_command``'s own two calls,
    ``save_import_plan`` then ``submit_import``) -- the production default for
    ``_ensure_shadow_snapshot``'s ``submit_import_fn`` seam (Cutover PR-7b).
    Returns the new job's id.
    """
    from engine.v2.foundation import ArtifactStore
    from engine.v2.ops.snapshot_import import save_import_plan, submit_import
    from engine.v2.ops.stages import registry
    from engine.v2.ops.submission import NamespacePolicy

    store = ArtifactStore(_ops_root(root))
    plan_ref = save_import_plan(conn, store, plan, clock=clock)
    policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
    receipt = submit_import(conn, store, plan_ref.artifact_id, registry=registry(),
                            policy=policy, clock=clock, idempotency_key=idempotency_key,
                            repo_root=repo_root())
    return receipt.job_id


def _resulting_head_snapshot_id(root, conn, job_id) -> str:
    """Read ``SnapshotImportReceipt.resulting_head_snapshot_id`` off a
    succeeded ``snapshot_import`` job's own committed
    ``"snapshot_import_receipt"`` output artifact -- selected by NAME, never
    ``get_job().output_refs[0]`` (that list is ordered by artifact_id, not by
    output name, and a succeeded attempt can publish more than one named
    output -- e.g. the worker's own inspections output alongside the
    coordinator's receipt, ``snapshot_promotion.snapshot_import_effect``).
    Never a fresh ``data_snapshot_heads`` read either, which may already
    differ by the time this runs (Cutover PR-7b: an operator could run ``ops
    snapshot submit``/``promote`` by hand in between). Raises the typed,
    non-retryable ``INPUT_CHANGED`` ``OpsError`` if no such output exists, or
    if it exists but carries no usable ``resulting_head_snapshot_id`` --
    never returns ``None``, so a caller's ``"ready"`` outcome always carries
    a real snapshot id.
    """
    from engine.v2.foundation import ArtifactStore
    from engine.v2.ops.checkpoints import artifact

    row = conn.execute(
        "SELECT ao.artifact_id FROM attempt_outputs ao "
        "JOIN attempts a ON a.attempt_id = ao.attempt_id "
        "WHERE a.job_id = ? AND a.state = 'succeeded' AND ao.name = 'snapshot_import_receipt' "
        "ORDER BY a.attempt_number DESC LIMIT 1",
        (job_id,)).fetchone()
    if row is None:
        raise fail("INPUT_CHANGED",
                   "the shadow snapshot import succeeded with no receipt artifact",
                   details={"job_id": job_id})
    store = ArtifactStore(_ops_root(root))
    ref = artifact(conn, store, row["artifact_id"])
    document = json.loads(store.read_verified(ref))
    snapshot_id = document.get("resulting_head_snapshot_id") if isinstance(document, dict) else None
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise fail("INPUT_CHANGED",
                   "the shadow snapshot import receipt carries no resulting snapshot id",
                   details={"job_id": job_id})
    return snapshot_id


def _default_serve_snapshot_import(root, conn, job_id, clock,
                                   deadline_at) -> tuple[str, str | None]:
    """Drive ONE ``snapshot_import`` job to terminal and interpret the
    outcome -- the production default for ``_ensure_shadow_snapshot``'s
    ``serve_fn`` seam (Cutover PR-7b). Reuses :func:`_drive_jobs_to_terminal`,
    the SAME polling loop ``_default_serve`` uses, so there is exactly one
    ``supervisor.serve`` call site in this module. Returns ``("ready",
    snapshot_id)`` once the job succeeds (reading ``resulting_head_
    snapshot_id`` off its own committed receipt) or ``("timed_out", None)``
    if ``deadline_at`` fires first; raises the typed, non-retryable
    ``INPUT_CHANGED`` ``OpsError`` on any other terminal outcome.
    """
    from engine.v2.ops.profiles import DEFAULT_POLICY
    from engine.v2.ops.stages import registry
    from engine.v2.ops.supervisor import Service

    ops_root = _ops_root(root)
    service = Service(conn, ops_root, registry(), DEFAULT_POLICY, clock=clock,
                      code_source=repo_root())
    outcome = _drive_jobs_to_terminal(service, conn, (job_id,), deadline_at)
    if outcome == "timed_out":
        return "timed_out", None
    if outcome != "completed":
        raise fail("INPUT_CHANGED",
                   "the shadow snapshot import ended without succeeding",
                   details={"job_id": job_id})
    return "ready", _resulting_head_snapshot_id(root, conn, job_id)


def _shadow_import_job_identity(as_of: str, attempt: int) -> tuple[str, str]:
    """The ``attempt``-namespaced idempotency key and its job id.

    ``attempt`` (the caller's dedicated ``snapshot_attempt`` counter, never
    ``error_count``) namespaces ``f"shadow_snapshot_import:{as_of}:{attempt}"``:
    a fresh call after a prior TERMINAL failure of this specific job must mint
    a genuinely new key, because ``submission._insert_or_match`` matches an
    existing row under an unchanged key regardless of that row's own state --
    retrying under the SAME key would either re-match the dead row forever or
    risk an ``IDEMPOTENCY_CONFLICT`` if the legacy store moved since. A call
    that re-enters for the SAME ``attempt`` (crash-then-resume, or a prior
    ``"not_yet"``/``"timed_out"`` from :func:`_ensure_shadow_snapshot`) finds
    any existing row under that SAME key FIRST -- a plain, cheap catalog
    lookup, before anything else runs -- and never calls
    ``plan_import``/``submit_import`` again for it. (Cutover PR-7b.)
    """
    from engine.v2.ops.submission import job_id_for

    key = f"shadow_snapshot_import:{as_of}:{attempt}"
    return key, job_id_for("shadow", key)


def _shadow_import_existing_outcome(root, conn, job_id, state, clock, serve_fn):
    """Decides the outcome of an ALREADY-EXISTING ``snapshot_import`` row found
    under this exact ``attempt`` key (Cutover PR-7b): a ``succeeded`` row
    returns ``("ready", ...)`` immediately, reading ``snapshot_id`` off that
    job's own committed receipt (never a fresh ``data_snapshot_heads`` read,
    which may already differ); a non-terminal row (``queued``, ``running``,
    ``retry_wait``, ...) is reattached to and waited on, WITHOUT calling
    ``plan_import``/``submit_import`` again; a row that is terminal but not
    ``succeeded`` raises the typed, non-retryable ``INPUT_CHANGED``
    ``OpsError`` immediately instead of resubmitting.
    """
    if state == "succeeded":
        return "ready", _resulting_head_snapshot_id(root, conn, job_id)
    if state in TERMINAL_JOB_STATES:
        raise fail("INPUT_CHANGED",
                   "the shadow snapshot import ended without succeeding",
                   details={"job_id": job_id, "state": state})
    return serve_fn(root, conn, job_id, clock, _serve_deadline(clock))


def _ensure_shadow_snapshot(root: Path, as_of: str, clock, attempt: int, *,
                            plan_import_fn=None, submit_import_fn=None,
                            serve_fn=None) -> tuple[str, str | None]:
    """Cutover PR-7b (design: ``ARCHITECTURE.md`` "Cutover PR-7b's input
    sourcing" / "``nightly_trigger.py`` (Cutover PR-7b design)"). Commits (or
    reattaches to) the ``as_of`` session's ``shadow``-scope snapshot BEFORE any
    plan is built -- this is ``_submit_plan``'s default ``ensure_snapshot_fn``,
    called inside the same ``if plan_ref is None:`` guard, immediately before
    ``plan_fn`` (Cutover PR-7b-2).

    Returns ``("ready", snapshot_id)`` once a ``shadow``-scope snapshot is
    confirmed fresh for ``as_of`` and committed (or was already committed by
    an earlier call under this exact ``attempt``); ``("not_yet", None)`` when
    the legacy store has not caught up to ``as_of`` yet (the legacy input
    manifest's own ``selected_session`` does not match -- no attempt consumed,
    retry with the SAME ``attempt``); ``("timed_out", None)`` when the import
    job has not reached terminal before its deadline (it keeps running under
    its own lease; a later call with the SAME ``attempt`` reattaches to it); or
    raises the typed, non-retryable ``INPUT_CHANGED`` ``OpsError`` when the
    import job reaches a terminal, non-``succeeded`` state under this exact
    ``attempt`` -- never resubmitted under the SAME ``attempt``, only under a
    genuinely NEW one (the caller's ``TriggerReceipt.snapshot_attempt``, bumped
    only on this raise).

    The key/job-id rule is :func:`_shadow_import_job_identity`'s and the
    existing row's own state decides everything in
    :func:`_shadow_import_existing_outcome`; a row's absence falls through to a
    fresh plan-import. The catalog connection, the cheap existence check and
    the ``data_snapshot_heads`` CAS-pair read are this function's own direct
    reads, never behind a seam -- the same "cheap, catalog-only identity check,
    done first" pattern ``nightly.submit_native_score_batch_shadow_if_ready``
    already uses for a different job kind (Cutover PR-7a).

    ``plan_import_fn`` (``=None``, defaulting to :func:`_default_plan_import`)
    wraps ``engine.v2.data.import_snapshot.plan_import``: ``(root, expected_
    head_snapshot_id, expected_head_generation) -> ImportPlan``. ``submit_
    import_fn`` (``=None``, defaulting to :func:`_default_submit_import`)
    wraps ``snapshot_import.save_import_plan`` + ``submit_import``: ``(root,
    conn, plan, idempotency_key, clock) -> str`` (the new job's id).
    ``serve_fn`` (``=None``, defaulting to
    :func:`_default_serve_snapshot_import`) drives ONE job to terminal and
    interprets the outcome: ``(root, conn, job_id, clock, deadline_at) ->
    tuple[str, str | None]`` -- ``("ready", snapshot_id)`` or ``("timed_out",
    None)``, or it raises -- the SAME three-way contract this function itself
    has.
    """
    from engine.v2.foundation import ensure_directory
    from engine.v2.ops.bootstrap import open_catalog
    from engine.v2.ops.submission import get_job

    plan_import_fn = plan_import_fn or _default_plan_import
    submit_import_fn = submit_import_fn or _default_submit_import
    serve_fn = serve_fn or _default_serve_snapshot_import

    key, job_id = _shadow_import_job_identity(as_of, attempt)
    ops_root = _ops_root(root)
    ensure_directory(ops_root)
    conn = open_catalog(ops_root / "catalog.sqlite", clock=clock)
    try:
        existing = conn.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if existing is not None:
            return _shadow_import_existing_outcome(root, conn, job_id,
                                                   get_job(conn, job_id).state, clock, serve_fn)
        head_row = conn.execute(
            "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'"
        ).fetchone()
        expected_head_snapshot_id = head_row["snapshot_id"] if head_row is not None else None
        expected_head_generation = head_row["generation"] if head_row is not None else 0
        plan = plan_import_fn(root, expected_head_snapshot_id, expected_head_generation)
        if plan.legacy_input_manifest.selected_session != as_of:
            return "not_yet", None
        new_job_id = submit_import_fn(root, conn, plan, key, clock)
        return serve_fn(root, conn, new_job_id, clock, _serve_deadline(clock))
    finally:
        conn.close()


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
