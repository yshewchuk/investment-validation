"""EXP-190 Q2b folds: synthetic fixtures only (no store, no network, no data/)."""
import copy
from datetime import date

import pytest

from engine.v2.data.errors import DataError
from engine.v2.ops.errors import OpsError
from experiments.size_quantile_folds import annual_folds


def _row(event_id, ticker, event_date, label_end, ceiling=None):
    return {"event_id": event_id, "ticker": ticker, "event_date": event_date, "label_end": label_end,
            "ceiling": event_date if ceiling is None else ceiling}


ROWS = [
    _row("e20a", "T1", "2020-03-10", "2020-04-10"),
    _row("e20b", "T2", "2020-09-15", "2020-10-15"),
    _row("e21a", "T1", "2021-02-10", "2021-03-10"),
    _row("e21b", "T3", "2021-06-20", "2021-07-20"),
    _row("e22a", "T2", "2022-04-05", "2022-05-05"),
    _row("e22b", "T4", "2022-08-25", "2022-09-25"),
    _row("e23a", "T5", "2023-03-15", "2023-04-15"),
    _row("e23b", "T1", "2023-07-01", "2023-08-01"),
    _row("e23c", "T6", "2023-10-20", "2023-11-20"),
    _row("e24a", "T3", "2024-02-14", "2024-03-14"),
    _row("e24b", "T7", "2024-05-30", "2024-06-30"),
    _row("e24c", "T2", "2024-11-11", "2024-12-11"),
    _row("e25a", "T4", "2025-01-20", "2025-02-20"),
    _row("e25b", "T5", "2025-06-02", "2025-07-02"),
    _row("e25c", "T8", "2025-09-09", "2025-10-09"),
]
ALL_IDS = [r["event_id"] for r in ROWS]
EVAL_BEFORE = "2026-01-01"
FULL_WINDOW = {"ticker": "T3", "start": date.min.isoformat(), "end": date.max.isoformat()}

T23 = ["e20a", "e20b", "e21a", "e21b", "e22a", "e22b"]
T24 = ["e20a", "e20b", "e21a", "e21b", "e22a", "e22b", "e23a", "e23b", "e23c"]
T25 = ["e20a", "e20b", "e21a", "e21b", "e22a", "e22b", "e23a", "e23b", "e23c", "e24a", "e24b", "e24c"]


def _fold(year, ids):
    return {"year": year, "training_candidate_event_ids": list(ids)}


def _plan(t23=T23, t24=T24, t25=T25):
    return [{"outer": _fold("2025", t25), "inner": [_fold("2023", t23), _fold("2024", t24)]}]


def _sealed(rows=ROWS, plan=None, windows=(), admitted=None, evaluation_labels_before=EVAL_BEFORE):
    targets = [{"event_id": r["event_id"], "ticker": r["ticker"], "event_date": r["event_date"],
                "label_end": r["label_end"]} for r in rows]
    return {
        "context": {"target_events": targets, "eligibility_plan": {"random_windows": list(windows)}},
        "fold_events": [{"event_id": r["event_id"], "ticker": r["ticker"], "event_date": r["event_date"],
                         "forecasts": [{"source_session_ceiling": r["ceiling"]}]} for r in rows],
        "events": [{"event_id": r["event_id"]} for r in rows if admitted is None or r["event_id"] in admitted],
        "evaluation_labels_before": evaluation_labels_before,
        "plan": {"folds": copy.deepcopy(_plan() if plan is None else plan)},
    }


def _edit(rows, event_id, **changes):
    return [{**r, **changes} if r["event_id"] == event_id else r for r in rows]


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def test_earlier_year_fold_has_exact_training_evaluation_and_cutoff():
    assert annual_folds(_sealed(), ["2022"]) == {"2022": {
        "year": "2022", "training_labels_before": "2022-01-01",
        "training_event_ids": ["e20a", "e20b", "e21a", "e21b"], "evaluation_event_ids": ["e22a", "e22b"]}}


def test_cutoff_is_the_earliest_test_ceiling_when_it_falls_before_jan_first():
    rows = _edit(ROWS, "e22a", ceiling="2021-06-01")
    assert annual_folds(_sealed(rows=rows), ["2022"]) == {"2022": {
        "year": "2022", "training_labels_before": "2021-06-01",
        "training_event_ids": ["e20a", "e20b", "e21a"], "evaluation_event_ids": ["e22a", "e22b"]}}


def test_label_at_or_after_cutoff_and_the_day_before_are_both_dropped():
    rows = _edit(_edit(ROWS, "e23b", label_end="2024-01-01"), "e23c", label_end="2023-12-31")
    plan = _plan(t24=["e20a", "e20b", "e21a", "e21b", "e22a", "e22b", "e23a"])
    out = annual_folds(_sealed(rows=rows, plan=plan), ["2024"])
    assert out["2024"]["training_labels_before"] == "2024-01-01"
    assert out["2024"]["training_event_ids"] == ["e20a", "e20b", "e21a", "e21b", "e22a", "e22b", "e23a"]


def test_embargo_ticker_removes_from_training_and_evaluation():
    plan = _plan(t23=["e20a", "e20b", "e21a", "e22a", "e22b"],
                 t24=["e20a", "e20b", "e21a", "e22a", "e22b", "e23a", "e23b", "e23c"],
                 t25=["e20a", "e20b", "e21a", "e22a", "e22b", "e23a", "e23b", "e23c", "e24b", "e24c"])
    out = annual_folds(_sealed(plan=plan, windows=[FULL_WINDOW]), ["2024"])
    assert out["2024"]["training_event_ids"] == ["e20a", "e20b", "e21a", "e22a", "e22b", "e23a", "e23b", "e23c"]
    assert out["2024"]["evaluation_event_ids"] == ["e24b", "e24c"]


def test_partial_window_embargoes_nothing():
    partial = {"ticker": "T3", "start": "2024-01-01", "end": "2024-12-31"}
    out = annual_folds(_sealed(windows=[partial]), ["2024"])
    assert out["2024"]["training_event_ids"] == T24
    assert out["2024"]["evaluation_event_ids"] == ["e24a", "e24b", "e24c"]


def test_evaluation_drops_a_label_not_available_before_evaluation_labels_before():
    out = annual_folds(_sealed(evaluation_labels_before="2024-06-01"), ["2024"])
    assert out["2024"]["evaluation_event_ids"] == ["e24a"]


def test_evaluation_next_day_boundary_drops_at_label_end_plus_one_and_keeps_at_plus_two():
    # e24c has label_end 2024-12-11 (L); available_before needs label_end + 1 day < cutoff.
    dropped = annual_folds(_sealed(evaluation_labels_before="2024-12-12"), ["2024"])
    kept = annual_folds(_sealed(evaluation_labels_before="2024-12-13"), ["2024"])
    assert dropped["2024"]["evaluation_event_ids"] == ["e24a", "e24b"]
    assert kept["2024"]["evaluation_event_ids"] == ["e24a", "e24b", "e24c"]


def test_stderr_year_line_reports_nonzero_drops(capsys):
    # e23a (a 2024 training candidate) and e24c (a 2024 evaluation event) are not admitted.
    admitted = [i for i in ALL_IDS if i not in ("e23a", "e24c")]
    out = annual_folds(_sealed(admitted=admitted), ["2024"])
    assert out["2024"]["training_event_ids"] == [i for i in T24 if i != "e23a"]
    assert out["2024"]["evaluation_event_ids"] == ["e24a", "e24b"]
    err = capsys.readouterr().err
    assert ("[quantile_folds] year=2024 cutoff=2024-01-01 training=8 evaluation=2 "
            "dropped_training=1 dropped_evaluation=1") in err


def test_admission_removes_the_event_but_it_still_counts_for_the_cutoff():
    rows = _edit(ROWS, "e24b", ceiling="2023-12-15")
    admitted = [i for i in ALL_IDS if i != "e24b"]
    out = annual_folds(_sealed(rows=rows, admitted=admitted), ["2024", "2025"])
    assert out["2024"] == {"year": "2024", "training_labels_before": "2023-12-15", "training_event_ids": T24,
                           "evaluation_event_ids": ["e24a", "e24c"]}
    assert out["2025"]["training_event_ids"] == [i for i in T25 if i != "e24b"]
    assert out["2025"]["evaluation_event_ids"] == ["e25a", "e25b", "e25c"]


YEAR_BREAKS = [
    pytest.param({"plan": [{"outer": _fold("2025", T25), "inner": [_fold("2023", T23)]}]}, id="sealed_year_missing"),
    pytest.param({"plan": [{"outer": _fold("2025", T25), "inner": [_fold("2023", T23), _fold("2024", T24),
                                                                  _fold("2024", T24)]}]}, id="repeated_year"),
    pytest.param({"plan": [{"outer": _fold("2025", T25), "inner": [_fold("2023", T23), _fold("2024", T24),
                                                                  _fold("2022", T23)]}]}, id="extra_year"),
    pytest.param({"plan": [{"outer": _fold("2023", T23), "inner": [_fold("2023", T23)]},
                           {"outer": _fold("2024", T24), "inner": [_fold("2023", T23)]},
                           {"outer": _fold("2025", T25), "inner": [_fold("2023", T23), _fold("2024", T24)]}]},
                 id="three_outers_with_inners"),
]
PARITY_BREAKS = [
    pytest.param({"plan": _plan(t24=[i for i in T24 if i != "e22b"])}, id="missing_id"),
    pytest.param({"plan": _plan(t25=T25 + ["e25a"])}, id="extra_id"),
    pytest.param({"plan": _plan(t24=["e20a", "e20b", "e21b", "e21a", "e22a", "e22b", "e23a", "e23b", "e23c"])},
                 id="reordered"),
    pytest.param({"rows": [r for r in ROWS if r["event_date"][:4] != "2025"]}, id="sealed_year_without_events"),
    *YEAR_BREAKS,
]


@pytest.mark.parametrize("change", PARITY_BREAKS)
def test_parity_refusals(change):
    with pytest.raises(OpsError) as exc:
        annual_folds(_sealed(**change), ["2024"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "fold_parity_mismatch")
    message = exc.value.problem.message
    assert message.startswith("fold_parity_mismatch: ")
    assert not any(event_id in message for event_id in ALL_IDS)


@pytest.mark.parametrize("change", YEAR_BREAKS)
def test_year_set_refusals_name_the_sealed_fold_years(change):
    with pytest.raises(OpsError) as exc:
        annual_folds(_sealed(**change), ["2024"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "fold_parity_mismatch")
    assert "sealed fold years" in exc.value.problem.message


def test_inner_folds_listed_in_reverse_year_order_still_pass():
    plan = [{"outer": _fold("2025", T25), "inner": [_fold("2024", T24), _fold("2023", T23)]}]
    out = annual_folds(_sealed(plan=plan), ["2024"])
    assert out["2024"]["training_event_ids"] == T24


@pytest.mark.parametrize("plan", [
    pytest.param([{"outer": _fold("2023", T23)}, {"outer": _fold("2024", T24)}, {"outer": _fold("2025", T25)}],
                 id="old_shape_outers_without_inner"),
    pytest.param([{"inner": [_fold("2023", T23), _fold("2024", T24)]}], id="group_missing_outer"),
])
def test_malformed_group_refuses_sealed_malformed(plan):
    with pytest.raises(OpsError) as exc:
        annual_folds(_sealed(plan=plan), ["2024"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "sealed_malformed")


def test_parity_refuses_even_when_only_an_earlier_year_is_requested():
    sealed = _sealed(plan=_plan(t24=[i for i in T24 if i != "e22b"]))
    with pytest.raises(OpsError) as exc:
        annual_folds(sealed, ["2022"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "fold_parity_mismatch")


def test_sealed_year_request_returns_the_sealed_training_ids():
    out = annual_folds(_sealed(), ["2023"])
    assert out["2023"]["training_labels_before"] == "2023-01-01"
    assert out["2023"]["training_event_ids"] == T23
    assert out["2023"]["evaluation_event_ids"] == ["e23a", "e23b", "e23c"]


@pytest.mark.parametrize("years", [[], "2024", 2024, ["2026"], ["2024", "2023"], ["2023", "2023"], ["202x"]])
def test_invalid_years_refuse(years):
    with pytest.raises(OpsError) as exc:
        annual_folds(_sealed(), years)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "years_invalid")


def test_year_without_events_is_population_collapsed():
    with pytest.raises(DataError) as exc:
        annual_folds(_sealed(), ["2019"])
    assert exc.value.code == "POPULATION_COLLAPSED"


def test_year_without_earlier_events_is_population_collapsed():
    with pytest.raises(DataError) as exc:
        annual_folds(_sealed(), ["2020"])
    assert exc.value.code == "POPULATION_COLLAPSED"


def test_training_emptied_by_the_filter_is_population_collapsed():
    rows = _edit(_edit(ROWS, "e20a", label_end="2020-12-31"), "e20b", label_end="2020-12-31")
    with pytest.raises(DataError) as exc:
        annual_folds(_sealed(rows=rows), ["2021"])
    assert exc.value.code == "POPULATION_COLLAPSED"


def test_evaluation_emptied_by_the_filter_is_population_collapsed():
    with pytest.raises(DataError) as exc:
        annual_folds(_sealed(evaluation_labels_before="2000-01-01"), ["2024"])
    assert exc.value.code == "POPULATION_COLLAPSED"


def test_missing_fold_events_refuses_sealed_malformed():
    sealed = _sealed()
    del sealed["fold_events"]
    with pytest.raises(OpsError) as exc:
        annual_folds(sealed, ["2024"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "sealed_malformed")
    assert exc.value.problem.message.startswith("sealed_malformed: ")
    assert "KeyError" in exc.value.problem.message


def test_missing_label_end_refuses_sealed_malformed():
    sealed = _sealed()
    del sealed["context"]["target_events"][0]["label_end"]
    with pytest.raises(OpsError) as exc:
        annual_folds(sealed, ["2024"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "sealed_malformed")


def test_stdout_is_empty_and_progress_lines_are_on_stderr(capsys):
    annual_folds(_sealed(), ["2023", "2024", "2025"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[quantile_folds] parity=ok sealed_years=2023,2024,2025" in captured.err
    assert ("[quantile_folds] year=2024 cutoff=2024-01-01 training=9 evaluation=3 "
            "dropped_training=0 dropped_evaluation=0") in captured.err
