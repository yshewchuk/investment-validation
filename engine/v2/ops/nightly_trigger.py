"""Schedule the native shadow nightly behind an ORATS-finality gate.

Slice 12's scheduled path is deliberately narrow. The legacy nightly keeps
its own crontab line and this module never touches it; what is scheduled here
is the parallel shadow-mode qualification DAG that already exists
(``ops plan nightly`` + ``ops submit``), gated on two facts:

* the as-of session is published at the provider -- a single lightweight
  ORATS market-wide probe (the summaries/cores pair legacy nightly step 1
  already makes), never a refresh job; and
* no legacy nightly holds ``<repo>/reports/.nightly.lock`` -- a non-blocking
  ``fcntl.flock`` probe, released again before submission -- so only one
  heavy job is ever in flight.

Everything is pure decision logic plus a thin CLI: the finality provider and
the submit call are injected, so tests drive every branch without a network
or a catalog. The state file (``reports/phase6/nightly_trigger/<as_of>.json``)
is the idempotency record: once ``submitted`` or ``missed`` for a date, the
receipt is returned unchanged forever -- never a second submit, never a
resurrected missed date. ``ops submit``'s own nightly idempotency (stage
identity from the plan document) is untouched: the trigger's per-date state
is bookkeeping for the retry window, not a second job-dedup layer.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable
from zoneinfo import ZoneInfo

from engine.v2.foundation import SystemClock, format_timestamp
from engine.v2.ops.errors import OpsError, fail

__all__ = [
    "TriggerReceipt",
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
TERMINAL_STATUSES = frozenset({"submitted", "missed"})
FAILURE_STATUSES = frozenset({"missed", "error"})
STATUSES = ("submitted", "not_yet", "missed", "already_submitted", "busy_legacy", "error")
STATE_DIR = ("reports", "phase6", "nightly_trigger")
#: Optional operator documents a scheduled run needs to be submittable at all:
#: the ``capture-inputs`` manifest and the planned ``ticker|strategy|event_date``
#: population. Absent, the plan still records the native default (empty)
#: population and ``ops submit`` refuses it, exactly like any other unplanned
#: nightly -- the trigger records that refusal as ``error`` rather than hiding it.
QUALIFICATION_INPUT_MANIFEST = "input_manifest.json"
QUALIFICATION_POPULATION = "expected_population.json"

FinalityProvider = Callable[[str, Iterable[str]], "tuple[bool, str]"]
SubmitCallable = Callable[..., str]


@dataclass(frozen=True)
class TriggerReceipt:
    schema_version: str = "nightly_trigger.v1.0"
    as_of: str = ""
    status: str = ""
    detail: str = ""
    checked_at: str = ""
    plan_ref: str | None = None


def repo_root() -> Path:
    """The code checkout's repo root, the way v2 ops code resolves it."""
    return Path(__file__).resolve().parents[3]


def legacy_lock_path() -> Path:
    """``<repo>/reports/.nightly.lock`` -- the legacy nightly's own lock file."""
    return repo_root() / "reports" / ".nightly.lock"


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
    return TriggerReceipt(
        schema_version=str(document.get("schema_version", TriggerReceipt.schema_version)),
        as_of=as_of, status=status, detail=str(document.get("detail", "")),
        checked_at=str(document.get("checked_at", "")),
        plan_ref=plan_ref if isinstance(plan_ref, str) else None)


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


def _legacy_free(path: Path) -> bool:
    """True when the legacy nightly lock can be taken; released before return.

    Non-blocking ``fcntl.flock`` on the file the legacy nightly itself holds,
    so the probe observes an in-flight heavy run. The handle is closed (which
    releases the lock) before the caller submits. An unopenable lock file is
    treated as held: fail closed, the next timer tick retries.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
    except OSError:
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    finally:
        handle.close()
    return True


def _boundary(value: str):
    try:
        return datetime.strptime(value, "%H:%M").time()
    except (TypeError, ValueError):
        raise fail("INVALID_REQUEST",
                   "ET window boundaries must be HH:MM") from None


def _validate_as_of(as_of: str) -> None:
    try:
        date.fromisoformat(as_of)
    except (TypeError, ValueError):
        raise fail("INVALID_REQUEST", "as_of must be an ISO date") from None


def _receipt(clock, as_of: str, status: str, detail: str,
             plan_ref: str | None = None) -> TriggerReceipt:
    return TriggerReceipt(as_of=as_of, status=status, detail=detail,
                          checked_at=format_timestamp(clock.now()), plan_ref=plan_ref)


def _record(root: Path, receipt: TriggerReceipt) -> TriggerReceipt:
    write_state(root, receipt)
    return receipt


def run_trigger(root: Path, as_of: str, *, tickers, context_tickers,
                deadline_et: str, window_start_et: str, provider=None,
                clock=None, submit_fn: SubmitCallable | None = None) -> TriggerReceipt:
    """The whole decision: state -> window -> deadline -> finality -> lock -> submit.

    ``tickers``/``context_tickers`` are the plan's watchlist and historical
    evidence universe. ``submit_fn`` and ``provider`` are injected seams; the
    production defaults are the real in-process plan/submit and the native
    ORATS probe.
    """
    clock = clock or SystemClock()
    root = Path(root)
    _validate_as_of(as_of)
    prior = load_state(root, as_of)
    if prior is not None and prior.status in TERMINAL_STATUSES:
        return prior
    now_et = clock.now().astimezone(ET).time()
    if now_et < _boundary(window_start_et):
        return _receipt(clock, as_of, "not_yet", "before the retry window opens")
    if now_et > _boundary(deadline_et):
        return _record(root, _receipt(
            clock, as_of, "missed", "the retry window closed before the session was final"))
    is_final, detail = probe_finality(as_of, tickers, provider=provider)
    if not is_final:
        return _record(root, _receipt(clock, as_of, "not_yet", detail))
    if not _legacy_free(legacy_lock_path()):
        return _record(root, _receipt(
            clock, as_of, "busy_legacy", "the legacy nightly lock is held; retrying next tick"))
    submit = submit_fn or _default_submit
    try:
        plan_ref = submit(root=root, as_of=as_of, tickers=tuple(tickers),
                          context_tickers=tuple(context_tickers), clock=clock)
    except (OpsError, OSError, ValueError, TypeError) as exc:
        detail = (f"{exc.code}: {exc.problem.message}" if isinstance(exc, OpsError)
                  else type(exc).__name__)
        return _record(root, _receipt(clock, as_of, "error", detail))
    return _record(root, _receipt(clock, as_of, "submitted",
                                  "submitted the shadow nightly plan", plan_ref=plan_ref))


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


def _scheduled_universe(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The scheduled plan's universe, with no ticker-list argument anywhere.

    The operator's qualified full population, when present, IS the default
    universe: one ticker per ``ticker|strategy|event_date`` key, used for both
    the watchlist and the historical-evidence context. Without that document
    the native plan's own default (empty) universe is used, exactly as an
    unconfigured ``ops plan nightly`` would.
    """
    document = _read_document(_qualification_path(root, QUALIFICATION_POPULATION))
    if not isinstance(document, list):
        return (), ()
    tickers = tuple(sorted({str(key).split("|")[0] for key in document
                            if isinstance(key, str) and "|" in key}))
    return tickers, tickers


def _default_submit(root: Path, as_of: str, tickers, context_tickers, clock) -> str:
    """The production submit: the real CLI plan/submit, in-process.

    ``cli._plan_command``/``cli._submit_command`` are the same functions the
    operator CLI dispatches to, called here under one catalog connection so no
    subprocess edge is introduced (import layers forbid one outside the
    executor/adapter pair). The ops root is the CLI's own default,
    ``<root>/data/operations``. The qualification documents beside the state
    files supply the frozen input manifest and planned population when the
    operator has captured them; their absence is refused by ``ops submit``
    itself and recorded as an ``error`` receipt, never bypassed here.
    """
    from engine.v2.foundation import ensure_directory
    from engine.v2.ops import cli
    from engine.v2.ops.bootstrap import open_catalog

    ops_root = Path(root) / "data" / "operations"
    ensure_directory(ops_root)
    plan_args = argparse.Namespace(
        command="plan", kind="nightly", as_of=as_of, mode="shadow", spec=None,
        no_ledger=False,
        input_manifest=_qualification_path(root, QUALIFICATION_INPUT_MANIFEST),
        expected_population=_qualification_path(root, QUALIFICATION_POPULATION),
        tickers=",".join(tickers), context_tickers=",".join(context_tickers),
        full_run=False, year_start=2024, year_end=2026, input_mode="legacy",
        snapshot_scope=None, refresh_mode="legacy", refresh_plan=None)
    conn = open_catalog(ops_root / "catalog.sqlite", clock=clock)
    try:
        planned = cli._plan_command(plan_args, ops_root, conn, clock)
        cli._submit_command(
            argparse.Namespace(plan=planned["plan_ref"],
                               idempotency_key="nightly-" + as_of),
            ops_root, conn, clock)
    finally:
        conn.close()
    return planned["plan_ref"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--as-of", default=None,
                        help="YYYY-MM-DD; defaults to today in America/New_York")
    parser.add_argument("--root", default=".",
                        help="the root the state file lives under (default: .)")
    args = parser.parse_args(argv)
    clock = SystemClock()
    root = Path(args.root).resolve()
    as_of = args.as_of or clock.now().astimezone(ET).date().isoformat()
    tickers, context_tickers = _scheduled_universe(root)
    try:
        receipt = run_trigger(root, as_of, tickers=tickers, context_tickers=context_tickers,
                              deadline_et=DEFAULT_DEADLINE_ET,
                              window_start_et=DEFAULT_WINDOW_START_ET, clock=clock)
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
