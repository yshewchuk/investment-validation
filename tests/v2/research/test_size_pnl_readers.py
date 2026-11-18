"""EXP-188 S2b: admitted-key readers. Synthetic fixtures only; no store, no repository."""
import numpy as np
import pandas as pd
import pytest

from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.ops.errors import OpsError
from experiments import size_pnl_readers as rd
from experiments.size_pnl_inputs import LEAD

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
DENIED = "HOLDOUT_ACCESS_DENIED"
AS_OF = "2025-06"


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _ids(n, *, want):
    """First n canonical ids e0, e1, ... whose random-holdout membership is ``want``."""
    found = [f"e{i}" for i in range(10_000) if ExperimentHoldouts.random_membership(f"e{i}") is want]
    return found[:n]


def _admitted(rows):
    frame = pd.DataFrame(rows, columns=["event_id", "event_date"])
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    return rd.Admitted({"events": frame.sort_values("event_id").reset_index(drop=True), "excluded": {}})


def _load(universe, calls=None):
    """Fake shared reader: drops classified events and records holdout_exclusions like the real one."""
    def load(repository, snapshot, *, as_of_month, purpose, event_ids, max_object_bytes):
        if calls is not None:
            calls.append({"event_ids": event_ids, "purpose": purpose, "max_object_bytes": max_object_bytes})
        labels = [ExperimentHoldouts(as_of_month).classify(eid, day) for eid, day in universe]
        frame = pd.DataFrame(universe, columns=["event_id", "event_date"])
        frame = frame[[not lab for lab in labels]].reset_index(drop=True)
        frame.attrs["holdout_exclusions"] = [
            {"event_id": eid, "memberships": sorted(lab)} for (eid, _), lab in zip(universe, labels) if lab]
        return frame
    return load


PLAIN = _ids(5, want=False)
HOLD = _ids(1, want=True)[0]
UNIVERSE = [
    (PLAIN[0], "2023-05-01"), (PLAIN[1], "2024-06-01"), (PLAIN[2], "2024-09-01"),
    (PLAIN[3], "2025-03-01"),  # rolling window before the as-of month: dropped
    (PLAIN[4], "2025-08-01"),  # after the as-of month: ambiguous, dropped
    (HOLD, "2025-03-01"),      # random and rolling: counted under both labels
]
LABEL_IDS = ["a1", "a2", "a3", "a4", "a5"]
ADM = _admitted(list(zip(LABEL_IDS, ["2023-03-01", "2023-07-01", "2024-02-01", "2024-03-15", "2025-02-01"])))


def _row(i):
    return {"abs_move": 1.0 + i, "crush_pct_iv30": -2.0 - i, "pred_abs_move": 0.5 + i,
            "pred_iv_crush_30": -1.0, "horizon_sessions": 1}


TABLE = {eid: _row(i) for i, eid in enumerate(LABEL_IDS + PLAIN + [HOLD])}


def _spy(table, extra=()):
    """Fetch stub that records each requested batch and answers only from ``table``."""
    log = []

    def fetch(columns, ids):
        log.append(list(ids))
        rows = [{"event_id": eid, **{c: table[eid][c] for c in columns}} for eid in ids if eid in table]
        rows += [{"event_id": eid, **{c: 0.0 for c in columns}} for eid in extra]
        return pd.DataFrame(rows, columns=["event_id", *columns])
    return fetch, log


def test_admit_drops_classified_events_and_counts_by_label():
    calls = []
    out = rd.admit_events("repo", "snap", AS_OF, max_object_bytes=4096, load=_load(UNIVERSE, calls))
    events = out["events"]
    assert list(events["event_id"]) == sorted(PLAIN[:3])
    assert pd.api.types.is_datetime64_any_dtype(events["event_date"])
    assert not {HOLD, PLAIN[3], PLAIN[4]} & set(events["event_id"])
    assert out["excluded"] == {"ambiguous": 1, "random": 1, "rolling": 2}
    assert list(out["excluded"]) == sorted(out["excluded"])
    assert calls == [{"event_ids": None, "purpose": "selection", "max_object_bytes": 4096}]


@pytest.mark.parametrize("frame, reason", [
    (pd.DataFrame([UNIVERSE[0], UNIVERSE[0]], columns=["event_id", "event_date"]), "duplicate_event_id"),
    (pd.DataFrame({"event_id": ["x1"]}), "admission_malformed"),
    (pd.DataFrame({"event_date": ["2023-01-01"]}), "admission_malformed"),
])
def test_admit_refuses_malformed_populations(frame, reason):
    with pytest.raises(OpsError) as exc:
        rd.admit_events("repo", "snap", AS_OF, max_object_bytes=4096, load=lambda *a, **k: frame)
    assert _refusal(exc) == (CONFLICT, reason)


def test_read_labels_window_and_month_reads_only_admitted_ids():
    fetch, log = _spy(TABLE)
    out = rd.read_labels(fetch, ADM, start="2023-01-01", end="2025-01-01")
    assert list(out.index) == ["a1", "a2", "a3", "a4"]
    assert list(out.columns) == ["abs_move", "crush_pct_iv30", "event_date"]
    assert out.loc["a3", "abs_move"] == 3.0
    assert log == [["a1", "a2", "a3", "a4"]]
    rd.read_labels(fetch, ADM, start="2024-03-01", end="2024-04-01")
    assert log[-1] == ["a4"]


def test_dropped_holdouts_never_reach_fetch_and_explicit_ones_refuse():
    fetch, log = _spy(TABLE)
    rd.read_labels(fetch, rd.admit_events("repo", "snap", AS_OF, max_object_bytes=4096, load=_load(UNIVERSE)))
    assert {eid for batch in log for eid in batch} == set(PLAIN[:3])
    fetch, log = _spy(TABLE)
    with pytest.raises(OpsError) as exc:
        rd.read_labels(fetch, ADM, event_ids=["a1", HOLD])
    assert _refusal(exc) == (DENIED, "key_not_admitted")
    assert log == []


def test_plain_dict_admission_refuses_from_read_labels():
    plain = {"events": ADM["events"], "excluded": {}}
    fetch, log = _spy(TABLE)
    with pytest.raises(OpsError) as exc:
        rd.read_labels(fetch, plain)
    assert _refusal(exc) == (DENIED, "admission_not_issued")
    assert log == []


def test_fetch_returning_unrequested_key_refuses():
    fetch, _ = _spy(TABLE, extra=["zz"])
    with pytest.raises(OpsError) as exc:
        rd.read_labels(fetch, ADM)
    assert _refusal(exc) == (DENIED, "unrequested_key_returned")


def test_fetch_returning_repeated_key_refuses():
    def fetch(columns, ids):
        row = {"event_id": ids[0], **{c: 1.0 for c in columns}}
        return pd.DataFrame([row, row], columns=["event_id", *columns])
    with pytest.raises(OpsError) as exc:
        rd.read_labels(fetch, ADM)
    assert _refusal(exc) == (CONFLICT, "duplicate_event_id")


def test_window_excludes_2025_and_nan_labels_dropped_not_imputed(capsys):
    table = {**TABLE, "a2": {**TABLE["a2"], "abs_move": np.nan}}
    fetch, log = _spy(table)
    out = rd.read_labels(fetch, ADM, start="2023-01-01", end="2025-01-01")
    assert all("a5" not in batch for batch in log)
    assert list(out.index) == ["a1", "a3", "a4"]
    assert "label_missing=1" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_label_refuses_label_not_finite(bad):
    table = {**TABLE, "a2": {**TABLE["a2"], "abs_move": bad}}
    fetch, _ = _spy(table)
    with pytest.raises(OpsError) as exc:
        rd.read_labels(fetch, ADM)
    assert _refusal(exc) == (CONFLICT, "label_not_finite")


def test_string_label_refuses_label_not_finite():
    table = {**TABLE, "a1": {**TABLE["a1"], "crush_pct_iv30": "abc"}}
    fetch, _ = _spy(table)
    with pytest.raises(OpsError) as exc:
        rd.read_labels(fetch, ADM)
    assert _refusal(exc) == (CONFLICT, "label_not_finite")


def test_nan_and_none_labels_dropped_as_missing_with_float_columns(capsys):
    table = {**TABLE, "a1": {**TABLE["a1"], "abs_move": np.nan},
             "a3": {**TABLE["a3"], "crush_pct_iv30": None}}
    fetch, _ = _spy(table)
    out = rd.read_labels(fetch, ADM)
    assert list(out.index) == ["a2", "a4", "a5"]
    assert out["abs_move"].dtype == float and out["crush_pct_iv30"].dtype == float
    assert "label_missing=2" in capsys.readouterr().err


PROD_ADM = _admitted([("p0", "2018-06-01"), ("p1", "2023-02-01"), ("p2", "2023-06-01"), ("p3", "2024-01-10")])
PROD_TABLE = {"p0": _row(0), "p1": {**_row(1), "pred_abs_move": 0.6, "horizon_sessions": 1},
              "p2": {**_row(2), "pred_abs_move": 0.7, "pred_iv_crush_30": -2.0},
              "p3": {**_row(3), "pred_abs_move": 0.8}}


def _production(table, *, table_hash="h1", expected="h1", **kwargs):
    fetch, _ = _spy(table)
    return rd.read_production(fetch, PROD_ADM, expected_hash=expected, table_hash=table_hash, **kwargs)


def test_read_production_happy_path_start_filter_and_hash_binding():
    out = _production(PROD_TABLE, start="2023-01-01")
    frame = out["frame"]
    assert list(frame.index) == ["p1", "p2", "p3"]
    assert list(frame.columns) == ["event_date", "pred_move", "pred_crush"]
    assert frame.loc["p2", "pred_move"] == 0.7 and frame.loc["p2", "pred_crush"] == -2.0
    moved = _production(PROD_TABLE, table_hash="h2", expected="h2", start="2023-01-01")
    assert moved["binding"] != out["binding"]


def test_production_binding_changes_when_only_admitted_event_date_changes():
    base = _production(PROD_TABLE, start="2023-01-01")
    moved_adm = _admitted([("p0", "2018-06-01"), ("p1", "2023-02-01"), ("p2", "2023-06-15"), ("p3", "2024-01-10")])
    fetch, _ = _spy(PROD_TABLE)
    moved = rd.read_production(fetch, moved_adm, expected_hash="h1", table_hash="h1", start="2023-01-01")
    assert list(moved["frame"].index) == list(base["frame"].index) == ["p1", "p2", "p3"]
    for col in ("pred_move", "pred_crush"):
        assert list(moved["frame"][col]) == list(base["frame"][col])
    assert moved["frame"].loc["p2", "event_date"] != base["frame"].loc["p2", "event_date"]
    assert moved["binding"] != base["binding"]


def test_production_binding_covers_prediction_values():
    base = _production(PROD_TABLE, start="2023-01-01")
    again = _production(PROD_TABLE, start="2023-01-01")
    assert again["binding"] == base["binding"]
    changed = {**PROD_TABLE, "p2": {**PROD_TABLE["p2"], "pred_abs_move": 0.9}}
    assert _production(changed, start="2023-01-01")["binding"] != base["binding"]


def test_label_window_end_exclusive_start_inclusive_on_event_date():
    fetch, log = _spy(TABLE)
    out = rd.read_labels(fetch, ADM, start="2023-07-01", end="2024-02-01")
    assert list(out.index) == ["a2"]
    assert log == [["a2"]]


@pytest.mark.parametrize("table, kwargs, reason", [
    (PROD_TABLE, {"table_hash": "other"}, "production_hash_mismatch"),
    ({**PROD_TABLE, "p2": {**PROD_TABLE["p2"], "horizon_sessions": 3}}, {}, "lead_unavailable"),
    ({**PROD_TABLE, "p1": {**PROD_TABLE["p1"], "pred_abs_move": float("nan")}}, {},
     "production_series_failed"),
    (PROD_TABLE, {"start": "2030-01-01"}, "production_series_failed"),
])
def test_read_production_refusals(table, kwargs, reason):
    with pytest.raises(OpsError) as exc:
        _production(table, **kwargs)
    assert _refusal(exc) == (CONFLICT, reason)


def test_missing_production_row_logged_not_imputed(capsys):
    out = _production({k: v for k, v in PROD_TABLE.items() if k != "p2"}, start="2023-01-01")
    assert list(out["frame"].index) == ["p1", "p3"]
    assert "production_missing=1" in capsys.readouterr().err


MT_ADM = _admitted([("e1", "2023-02-01"), ("e2", "2023-06-01"), ("e3", "2024-02-01"),
                    ("e4", "2024-05-01"), ("e5", "2024-09-01")])
PROD = {"frame": pd.DataFrame({"event_date": pd.to_datetime(["2023-02-01", "2023-06-01", "2024-02-01", "2024-09-01"]),
                               "pred_move": [0.1, 0.2, 0.3, 0.5],
                               "pred_crush": [-1.0, -1.5, -2.0, -2.5]},
                              index=pd.Index(["e1", "e2", "e3", "e5"], name="event_id")),
        "binding": "pb"}


def _bound(years):
    """Hand-built bound candidate in the S2a shape; years maps year -> [(event_id, pred), ...]."""
    return {"status": "bound", "binding": "cand", "years": {
        str(y): {"pred": np.array([p for _, p in rows], dtype=float),
                 "event_ids": np.array([e for e, _ in rows], dtype=object), "binding": f"b{y}"}
        for y, rows in years.items()}}


def _model(mid, kind, source="v14"):
    return {"model_id": mid, "source": source, "candidate_id": mid, "target_transform": "none",
            "kind": kind, "roles": [mid]}


def _tables(bound, models, **kwargs):
    return rd.model_tables(bound, {"models": models}, MT_ADM, PROD, **kwargs)


def test_candidate_concatenates_inner_years():
    out = _tables({"m": _bound({2023: [("e1", 1.0), ("e2", 2.0)], 2024: [("e3", 3.0)]})}, [_model("m", "candidate")])
    assert list(out["m"]["table"].index) == ["e1", "e2", "e3"]
    assert list(out["m"]["table"]["pred_move"]) == [1.0, 2.0, 3.0]
    assert out["m"]["status"] == "ready" and out["m"]["binding"] != "cand" and isinstance(out["m"]["binding"], str)


def test_blend_is_equal_weight_mean_of_components():
    blend = {"status": "bound", "kind": "blend", "binding": "blend", "components": {
        "c1": _bound({2023: [("e1", 2.0), ("e2", 2.0)]}), "c2": _bound({2023: [("e1", 4.0), ("e2", 4.0)]})}}
    out = _tables({"b": blend}, [_model("b", "blend")])
    assert list(out["b"]["table"]["pred_move"]) == [3.0, 3.0]


def test_duplicate_across_years_and_mismatched_components_refuse():
    with pytest.raises(OpsError) as exc:
        _tables({"m": _bound({2023: [("e1", 1.0)], 2024: [("e1", 2.0)]})}, [_model("m", "candidate")])
    assert _refusal(exc) == (CONFLICT, "duplicate_event_id")
    blend = {"status": "bound", "kind": "blend", "binding": "blend", "components": {
        "c1": _bound({2023: [("e1", 2.0), ("e2", 2.0)]}), "c2": _bound({2023: [("e1", 4.0)]})}}
    with pytest.raises(OpsError) as exc:
        _tables({"b": blend}, [_model("b", "blend")])
    assert _refusal(exc) == (CONFLICT, "component_ids_differ")


def test_excluded_deferred_and_production_models_pass_through():
    bound = {"x": {"status": "excluded", "reason": "missing"},
             "y": {"status": "deferred", "reason": "production_bound_by_s2b"},
             "p": {"status": "deferred", "reason": "production_bound_by_s2b"}}
    out = _tables(bound, [_model("x", "candidate"), _model("y", "candidate"),
                          _model("p", "production", source="production")])
    assert out["x"] == {"status": "excluded", "reason": "missing"}
    assert out["y"] == {"status": "excluded", "reason": "production_bound_by_s2b"}
    assert out["p"]["table"].equals(PROD["frame"]) and out["p"]["binding"] != "pb"


def test_no_crush_rows_dropped_and_counted_and_outside_ids_refused():
    out = _tables({"m": _bound({2023: [("e1", 1.0), ("e4", 4.0)]})}, [_model("m", "candidate")])
    assert list(out["m"]["table"].index) == ["e1"]
    assert out["m"]["dropped"] == {"no_crush_forecast": 1}
    with pytest.raises(OpsError) as exc:
        _tables({"m": _bound({2023: [("e1", 1.0), ("zz", 2.0)]})}, [_model("m", "candidate")])
    assert _refusal(exc) == (CONFLICT, "event_ids_outside_population")


def test_production_frame_with_unadmitted_id_refuses():
    frame = PROD["frame"].rename(index={"e5": "zz"})
    bad_prod = {"frame": frame, "binding": PROD["binding"]}
    with pytest.raises(OpsError) as exc:
        rd.model_tables({}, {"models": [_model("p", "production", source="production")]}, MT_ADM, bad_prod)
    assert _refusal(exc) == (CONFLICT, "event_ids_outside_population")


def test_model_tables_refuses_non_finite_production_prediction():
    frame = PROD["frame"].assign(pred_move=[0.1, float("nan"), 0.3, 0.5])
    bad_prod = {"frame": frame, "binding": PROD["binding"]}
    with pytest.raises(OpsError) as exc:
        rd.model_tables({}, {"models": [_model("p", "production", source="production")]}, MT_ADM, bad_prod)
    assert _refusal(exc) == (CONFLICT, "production_series_failed")


def test_model_tables_refuses_production_frame_missing_column():
    frame = PROD["frame"].drop(columns=["pred_crush"])
    bad_prod = {"frame": frame, "binding": PROD["binding"]}
    with pytest.raises(OpsError) as exc:
        rd.model_tables({}, {"models": [_model("p", "production", source="production")]}, MT_ADM, bad_prod)
    assert _refusal(exc) == (CONFLICT, "production_series_failed")


def _via_model_tables(frame):
    """Production frame through model_tables; returns the production table."""
    roster = {"models": [_model("p", "production", source="production")]}
    return rd.model_tables({}, roster, MT_ADM, {"frame": frame, "binding": "pb"})["p"]["table"]


def _via_read_production(frame):
    """Production frame served through the fetch path of read_production; returns its frame."""
    def fetch(columns, ids):
        rows = [{"event_id": eid, "pred_abs_move": move, "pred_iv_crush_30": crush, "horizon_sessions": LEAD}
                for eid, move, crush in zip(frame.index, frame["pred_move"], frame["pred_crush"])]
        return pd.DataFrame(rows, columns=["event_id", *columns])
    return rd.read_production(fetch, MT_ADM, expected_hash="h1", table_hash="h1")["frame"]


ENTRY_PATHS = [_via_model_tables, _via_read_production]


@pytest.mark.parametrize("run", ENTRY_PATHS)
def test_production_path_refuses_empty_frame(run):
    with pytest.raises(OpsError) as exc:
        run(PROD["frame"].iloc[:0])
    assert _refusal(exc) == (CONFLICT, "production_series_failed")


@pytest.mark.parametrize("run", ENTRY_PATHS)
def test_production_path_refuses_duplicate_index(run):
    with pytest.raises(OpsError) as exc:
        run(PROD["frame"].iloc[[0, 0, 1, 2]])
    assert _refusal(exc) == (CONFLICT, "duplicate_event_id")


@pytest.mark.parametrize("run", ENTRY_PATHS)
def test_production_path_refuses_string_predictions(run):
    with pytest.raises(OpsError) as exc:
        run(PROD["frame"].assign(pred_move=["abc", "0.2", "0.3", "0.5"]))
    assert _refusal(exc) == (CONFLICT, "production_series_failed")


@pytest.mark.parametrize("run", ENTRY_PATHS)
def test_production_path_returns_numeric_strings_as_floats(run):
    frame = PROD["frame"].assign(pred_move=["0.1", "0.2", "0.3", "0.5"], pred_crush=["-1.0", "-1.5", "-2.0", "-2.5"])
    out = run(frame)
    assert list(out["pred_move"]) == [0.1, 0.2, 0.3, 0.5]
    assert list(out["pred_crush"]) == [-1.0, -1.5, -2.0, -2.5]
    assert out["pred_move"].dtype == float and out["pred_crush"].dtype == float


def test_model_tables_replaces_supplied_event_date_with_admitted_date():
    frame = PROD["frame"].assign(event_date=pd.to_datetime(["2030-01-01"] * 4))
    table = _via_model_tables(frame)
    assert list(table["event_date"]) == list(PROD["frame"]["event_date"])
    assert table.loc["e2", "event_date"] == pd.Timestamp("2023-06-01")


def test_model_tables_windowing_uses_admitted_date_not_supplied_date():
    frame = PROD["frame"].assign(event_date=pd.to_datetime(["2030-01-01"] * 4))
    roster = {"models": [_model("p", "production", source="production")]}
    out = rd.model_tables({}, roster, MT_ADM, {"frame": frame, "binding": "pb"},
                          start="2023-03-01", end="2024-01-01")
    assert list(out["p"]["table"].index) == ["e2"]


def test_start_and_end_filters_applied():
    out = _tables({"m": _bound({2023: [("e1", 1.0), ("e2", 2.0), ("e3", 3.0)]})},
                  [_model("m", "candidate")], start="2023-03-01", end="2024-01-01")
    assert list(out["m"]["table"].index) == ["e2"]


def test_production_model_window_excludes_pre_start_rows():
    out = _tables({}, [_model("p", "production", source="production")], start="2023-03-01")
    assert list(out["p"]["table"].index) == ["e2", "e3", "e5"]


def test_same_candidate_different_windows_give_different_bindings():
    bound = {"m": _bound({2023: [("e1", 1.0), ("e2", 2.0), ("e3", 3.0)]})}
    full = _tables(bound, [_model("m", "candidate")])
    narrow = _tables(bound, [_model("m", "candidate")], start="2023-03-01", end="2024-01-01")
    assert full["m"]["binding"] != narrow["m"]["binding"]


def test_changed_production_crush_changes_candidate_binding_with_same_ids_and_preds():
    bound = {"m": _bound({2023: [("e1", 1.0), ("e2", 2.0), ("e3", 3.0)]})}
    base = _tables(bound, [_model("m", "candidate")])
    crush = PROD["frame"]["pred_crush"].copy()
    crush.iloc[1] = -9.0  # e2's production crush forecast changes
    changed_prod = {"frame": PROD["frame"].assign(pred_crush=crush), "binding": PROD["binding"]}
    changed = rd.model_tables(bound, {"models": [_model("m", "candidate")]}, MT_ADM, changed_prod)
    assert list(changed["m"]["table"].index) == list(base["m"]["table"].index)
    assert list(changed["m"]["table"]["pred_move"]) == list(base["m"]["table"]["pred_move"])
    assert list(changed["m"]["table"]["pred_crush"]) != list(base["m"]["table"]["pred_crush"])
    assert changed["m"]["binding"] != base["m"]["binding"]
