"""Focused acceptance tests for engine.v2.foundation.market_calendar.

Pure stdlib layer-0.5 code: hand-built canonical session strings and fixed
2019-2024 dates only — no real data, no network, no legacy imports.
"""
from __future__ import annotations

import ast
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.foundation.market_calendar import (  # noqa: E402
    CalendarInputError,
    CalendarSessions,
    build_calendar_sessions,
    planned_exit_date,
)

MODULE_PATH = ROOT / "engine" / "v2" / "foundation" / "market_calendar.py"

OBS_MAR = (
    "2024-03-08", "2024-03-11", "2024-03-12",
    "2024-03-13", "2024-03-14", "2024-03-15",
)
POST_STRATEGIES = ("STR-THRU", "DYN-SV", "TWIN-P", "TWIN-P5", "CND-PS",
                   "BFLY-P", "BFLY-P5", "RAMP7", "CTR5")


def _key(strategy: object, event: object, session: object) -> SimpleNamespace:
    return SimpleNamespace(strategy=strategy, event_date=event, session=session)


@pytest.fixture(scope="module")
def calendar() -> CalendarSessions:
    return build_calendar_sessions(OBS_MAR, event_through="2024-03-15")


# --------------------------------------------------------------------------
# build_calendar_sessions: R1 refusals, R6 determinism, projection rules
# --------------------------------------------------------------------------


def test_build_deduplicates_and_sorts() -> None:
    scrambled = ("2024-03-13", "2024-03-11", "2024-03-15", "2024-03-08",
                 "2024-03-13", "2024-03-14", "2024-03-12", "2024-03-11")
    assert (build_calendar_sessions(scrambled, event_through="2024-03-15").days
            == build_calendar_sessions(OBS_MAR, event_through="2024-03-15").days)


def test_projection_extends_and_appends_one_tail_session(calendar: CalendarSessions) -> None:
    assert calendar.days == OBS_MAR + ("2024-03-18",)
    assert calendar.observed_through == "2024-03-15"


def test_event_through_below_observed_max_appends_no_extra_session() -> None:
    built = build_calendar_sessions(OBS_MAR, event_through="2024-03-12")
    assert built.days == OBS_MAR  # 2024-03-13 is already observed, no 2024-03-18 tail
    assert built.observed_through == "2024-03-15"


def test_sparse_source_includes_first_post_through_session() -> None:
    built = build_calendar_sessions(("2024-03-08", "2024-03-15"), event_through="2024-03-10")
    assert built.days == ("2024-03-08", "2024-03-11", "2024-03-15")
    assert built.observed_through == "2024-03-15"


def test_observed_through_never_moves_under_projection() -> None:
    built = build_calendar_sessions(OBS_MAR, event_through="2024-04-10")
    assert built.observed_through == "2024-03-15"
    assert built.days[0] == "2024-03-08"
    assert built.days[-1] > "2024-04-10"


def test_projection_skips_good_friday() -> None:
    built = build_calendar_sessions(("2024-03-28",), event_through="2024-04-01")
    assert built.days == ("2024-03-28", "2024-04-01", "2024-04-02")


def test_thanksgiving_christmas_endpoints() -> None:
    nov = build_calendar_sessions(("2024-11-26", "2024-11-27"), event_through="2024-11-29")
    assert nov.days == ("2024-11-26", "2024-11-27", "2024-11-29", "2024-12-02")
    dec = build_calendar_sessions(("2024-12-20",), event_through="2024-12-26")
    assert dec.days == ("2024-12-20", "2024-12-23", "2024-12-24",
                        "2024-12-26", "2024-12-27")


def test_juneteenth_rule() -> None:
    early = build_calendar_sessions(("2019-06-18",), event_through="2019-06-20")
    assert "2019-06-19" in early.days  # not a market holiday before 2022
    late = build_calendar_sessions(("2022-06-17",), event_through="2022-06-21")
    assert late.days == ("2022-06-17", "2022-06-21", "2022-06-22")  # Jun 19 Sun -> Jun 20
    with pytest.raises(CalendarInputError) as excinfo:
        build_calendar_sessions(("2023-06-19",), event_through="2023-06-21")
    assert excinfo.value.code == "INVALID_REQUEST"


def test_new_year_observance_rules() -> None:
    sunday = build_calendar_sessions(("2022-12-30",), event_through="2023-01-03")
    assert sunday.days == ("2022-12-30", "2023-01-03", "2023-01-04")  # Jan 1 Sun -> Jan 2
    saturday = build_calendar_sessions(("2021-12-31",), event_through="2022-01-03")
    assert saturday.days == ("2021-12-31", "2022-01-03", "2022-01-04")  # no Friday call


INVALID_OBSERVED = [
    (),
    ("2024-03-16",),
    ("2024-12-25",),
    ("2024-3-13",),
    ("20240313",),
    ("2024-03-13T00:00",),
    ("2024-02-30",),
    (date(2024, 3, 13),),
]


@pytest.mark.parametrize("observed", INVALID_OBSERVED)
def test_invalid_observed_sessions_refused(observed: tuple) -> None:
    with pytest.raises(CalendarInputError) as excinfo:
        build_calendar_sessions(observed, event_through="2024-03-15")
    assert excinfo.value.code == "INVALID_REQUEST"


INVALID_THROUGH = [
    datetime(2024, 3, 20, tzinfo=timezone.utc),
    "2024-3-20",
    "2024-03-20 ",
    20240320,
    None,
]


@pytest.mark.parametrize("bad", INVALID_THROUGH)
def test_invalid_event_through_refused(bad: object) -> None:
    with pytest.raises(CalendarInputError) as excinfo:
        build_calendar_sessions(OBS_MAR, event_through=bad)
    assert excinfo.value.code == "INVALID_REQUEST"


def test_repeated_calls_are_deterministic(calendar: CalendarSessions) -> None:
    again = build_calendar_sessions(OBS_MAR, event_through="2024-03-15")
    assert again == calendar
    assert again.days == calendar.days
    assert again.observed_through == calendar.observed_through
    key = _key("BFLY-P5", "2024-03-13", "AMC")
    separate = build_calendar_sessions(tuple(sorted(OBS_MAR)), event_through="2024-03-15")
    assert separate.days == calendar.days
    assert planned_exit_date(key, again) == planned_exit_date(key, separate) == "2024-03-14"


# --------------------------------------------------------------------------
# planned_exit_date: session-aware anchors and the fixed exit policy
# --------------------------------------------------------------------------


def test_session_day_anchors(calendar: CalendarSessions) -> None:
    assert planned_exit_date(_key("STR-RUNUP", "2024-03-13", "BMO"), calendar) == "2024-03-12"
    assert planned_exit_date(_key("STR-THRU", "2024-03-13", "BMO"), calendar) == "2024-03-13"
    assert planned_exit_date(_key("STR-RUNUP", date(2024, 3, 13), "AMC"), calendar) == "2024-03-13"
    naive = datetime(2024, 3, 13, 16, 30)
    assert planned_exit_date(_key("STR-THRU", naive, "AMC"), calendar) == "2024-03-14"


@pytest.mark.parametrize("strategy", POST_STRATEGIES)
def test_post_print_dispatch(calendar: CalendarSessions, strategy: str) -> None:
    assert planned_exit_date(_key(strategy, "2024-03-13", "AMC"), calendar) == "2024-03-14"


def test_weekend_event_date_anchors(calendar: CalendarSessions) -> None:
    assert planned_exit_date(_key("STR-RUNUP", "2024-03-16", "BMO"), calendar) == "2024-03-15"
    assert planned_exit_date(_key("STR-RUNUP", "2024-03-16", "AMC"), calendar) == "2024-03-15"
    assert planned_exit_date(_key("STR-THRU", "2024-03-16", "BMO"), calendar) == "2024-03-18"
    assert planned_exit_date(_key("CTR5", "2024-03-16", "AMC"), calendar) == "2024-03-18"


def test_holiday_event_date_anchors() -> None:
    nov = build_calendar_sessions(("2024-11-26", "2024-11-27"), event_through="2024-11-29")
    assert planned_exit_date(_key("STR-RUNUP", "2024-11-28", "AMC"), nov) == "2024-11-27"
    assert planned_exit_date(_key("TWIN-P", "2024-11-28", "BMO"), nov) == "2024-11-29"


def test_projected_future_exit() -> None:
    future = build_calendar_sessions(OBS_MAR, event_through="2024-03-25")
    assert future.observed_through == "2024-03-15"
    assert future.days[-1] == "2024-03-26"
    assert planned_exit_date(_key("STR-THRU", "2024-03-20", "BMO"), future) == "2024-03-20"
    assert planned_exit_date(_key("STR-THRU", "2024-03-20", "AMC"), future) == "2024-03-21"
    assert planned_exit_date(_key("STR-RUNUP", "2024-03-18", "BMO"), future) == "2024-03-15"


def test_insufficient_anchor_coverage(calendar: CalendarSessions) -> None:
    with pytest.raises(CalendarInputError) as excinfo:
        planned_exit_date(_key("STR-RUNUP", "2024-03-08", "BMO"), calendar)
    assert excinfo.value.code == "INVALID_REQUEST"
    truncated = CalendarSessions(("2024-03-08", "2024-03-11"), "2024-03-11")
    with pytest.raises(CalendarInputError) as excinfo:
        planned_exit_date(_key("TWIN-P", "2024-03-11", "AMC"), truncated)
    assert excinfo.value.code == "INVALID_REQUEST"


def test_final_session_endpoint_events_keep_their_anchors(calendar: CalendarSessions) -> None:
    final = calendar.days[-1]
    assert final == "2024-03-18"
    assert planned_exit_date(_key("STR-RUNUP", final, "BMO"), calendar) == "2024-03-15"
    assert planned_exit_date(_key("STR-RUNUP", final, "AMC"), calendar) == final


@pytest.mark.parametrize("session", ["BMO", "AMC"])
def test_event_strictly_after_final_session_refused(calendar: CalendarSessions, session: str) -> None:
    with pytest.raises(CalendarInputError) as excinfo:
        planned_exit_date(_key("STR-RUNUP", "2024-03-19", session), calendar)
    assert excinfo.value.code == "INVALID_REQUEST"


def test_empty_calendar_refused() -> None:
    empty = CalendarSessions((), "2024-03-15")
    with pytest.raises(CalendarInputError) as excinfo:
        planned_exit_date(_key("STR-RUNUP", "2024-03-13", "AMC"), empty)
    assert excinfo.value.code == "INVALID_REQUEST"
    with pytest.raises(CalendarInputError) as excinfo:
        planned_exit_date(_key("STR-RUNUP", "2024-03-13", "BMO"), empty)
    assert excinfo.value.code == "INVALID_REQUEST"


INVALID_KEYS = [
    _key("CAL-P", "2024-03-13", "AMC"),
    _key("STR-RUNUP", "2024-03-13", "MOC"),
    _key("STR-RUNUP", "2024-03-13", None),
    _key("STR-RUNUP", "2024-3-13", "AMC"),
    _key("STR-RUNUP", datetime(2024, 3, 13, 9, 30, tzinfo=timezone.utc), "AMC"),
    _key("STR-RUNUP", 1234, "AMC"),
    _key([], "2024-03-13", "AMC"),
]


@pytest.mark.parametrize("key", INVALID_KEYS)
def test_invalid_key_refused(key: SimpleNamespace, calendar: CalendarSessions) -> None:
    with pytest.raises(CalendarInputError) as excinfo:
        planned_exit_date(key, calendar)
    assert excinfo.value.code == "INVALID_REQUEST"


def test_absent_structural_field_refused(calendar: CalendarSessions) -> None:
    no_session = SimpleNamespace(strategy="STR-RUNUP", event_date="2024-03-13")
    with pytest.raises(CalendarInputError) as excinfo:
        planned_exit_date(no_session, calendar)
    assert excinfo.value.code == "INVALID_REQUEST"
    broken = CalendarSessions(("2024-3-13",), "2024-03-13")
    with pytest.raises(CalendarInputError) as excinfo:
        planned_exit_date(_key("STR-THRU", "2024-03-13", "BMO"), broken)
    assert excinfo.value.code == "INVALID_REQUEST"


def test_error_envelope() -> None:
    assert issubclass(CalendarInputError, ValueError)
    assert CalendarInputError("x").code == "INVALID_REQUEST"


# --------------------------------------------------------------------------
# module isolation: stdlib-only imports, no legacy / provider / data reach
# --------------------------------------------------------------------------


def test_module_imports_are_stdlib_only() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert len(source.splitlines()) < 200
    tree = ast.parse(source, filename=str(MODULE_PATH))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.add((node.module or "").split(".")[0])
    assert roots <= sys.stdlib_module_names
    assert not roots & {"engine", "pandas", "numpy"}
