"""EXP-190 Q3c arm S: fits, inner folds, monthly z_q, binding counts, receipts and admission. Synthetic only (no data/,
no store); the regression is faked except in the end-to-end test."""
import copy
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import lib
from experiments import size_quantile_arm_s as arm_s
from experiments.label_availability import available_before, printed_before
from experiments.size_quantile_prereg import load_spec

SPEC, _ = load_spec()
FIXED = SPEC["grid"]["fixed"]
YEARS = ["2023", "2024", "2025"]
DICT = "sha256:" + "a" * 64
BINDING = "sha256:" + "b" * 64
OTHER = "sha256:" + "c" * 64
CONFIG = {"2023": [7, 300], "2024": [15, 1000], "2025": [15, 1000]}
EVENTS = [
    ("e20a", "T1", "2020-03-10", "2020-04-10"), ("e20b", "T2", "2020-09-15", "2020-10-15"),
    ("e21a", "T1", "2021-02-10", "2021-03-10"), ("e21b", "T3", "2021-06-20", "2021-07-20"),
    ("e22a", "T2", "2022-04-05", "2022-05-05"), ("e22b", "T4", "2022-08-25", "2022-09-25"),
    ("e23a", "T5", "2023-03-15", "2023-04-15"), ("e23b", "T1", "2023-07-01", "2023-08-01"),
    ("e23c", "T6", "2023-10-20", "2023-11-20"), ("e24a", "T3", "2024-02-14", "2024-03-14"),
    ("e24b", "T7", "2024-05-30", "2024-06-30"), ("e24c", "T2", "2024-11-11", "2024-12-11"),
    ("e25a", "T4", "2025-01-20", "2025-02-20"), ("e25b", "T5", "2025-06-02", "2025-07-02"),
    ("e25c", "T8", "2025-09-09", "2025-10-09"),
]
IDS = [e[0] for e in EVENTS]
DATE = {e: d for e, _, d, _ in EVENTS}
LABEL = {e: 1.0 + i % 5 for i, e in enumerate(IDS)}
T23 = ["e20a", "e20b", "e21a", "e21b", "e22a", "e22b"]
E23, E24, E25 = ["e23a", "e23b", "e23c"], ["e24a", "e24b", "e24c"], ["e25a", "e25b", "e25c"]
T24, T25 = T23 + E23, T23 + E23 + E24
EVAL = {"2023": E23, "2024": E24, "2025": E25}
LATE = {"dates": {"e23a": "2023-03-01", "e24a": "2024-03-01"}, "ends": {"e24a": "2024-05-15"}}


def _spec(**calibration):
    spec = copy.deepcopy(SPEC)
    spec["calibration"].update(calibration)
    return spec


def _outer_folds():
    return [{"outer": {"year": y, "training_candidate_event_ids": ids}}
            for y, ids in (("2023", T23), ("2024", T24), ("2025", T25))]


def _candidates():
    return [{"candidate_id": f"h{leaves}_{leaf}", "feature_recipe": {"subset": "HMOSXLP"},
             "recipe": {"schema_version": "magnitude_ridge_hgb_recipe.v1", "model": "history_hgb",
                        "estimator": {"params": {**FIXED, "max_leaf_nodes": leaves, "min_samples_leaf": leaf,
                                                 "loss": "quantile", "quantile": 0.5}}}}
            for leaves in (7, 15) for leaf in (300, 1000)]


def _sealed(binding=BINDING, dates=None, ends=None):
    dates, ends = dates or {}, ends or {}
    events = [(e, t, dates.get(e, d), ends.get(e, end)) for e, t, d, end in EVENTS]
    return {
        "context": {"target_events": [{"event_id": e, "ticker": t, "event_date": d, "label_end": end}
                                      for e, t, d, end in events],
                    "eligibility_plan": {"random_windows": []}},
        "fold_events": [{"event_id": e, "ticker": t, "event_date": d, "forecasts": [{"source_session_ceiling": d}]}
                        for e, t, d, _ in events],
        "events": [{"event_id": e} for e in IDS],
        "evaluation_labels_before": "2026-01-01",
        "plan": {"plan_id": "P1", "candidates": _candidates(), "folds": _outer_folds()},
        "rows": [{"event_id": e, "f_a": float(i), "f_b": i / 10, "f_empty": None} for i, e in enumerate(IDS)],
        "model_columns": ["f_a", "f_b", "f_empty"],
        "dictionary_hash": DICT,
        "binding": binding,
    }


def _sign(receipt):
    """Set the content hash over every other field, as the walk's receipts carry it."""
    receipt["content_hash"] = content_hash({k: v for k, v in receipt.items() if k != "content_hash"})
    return receipt


def _receipt(year, spec, event_ids=None, configuration=None):
    return _sign({"inputs": {"year": year, "sealed_binding": BINDING, "spec_hash": lib.spec_hash(spec)},
                  "event_ids": list(event_ids or EVAL[year]), "configuration": list(configuration or CONFIG[year])})


def _run(sealed, spec, years, out):
    return arm_s.fit_arm_s(sealed, spec, {y: _receipt(y, spec) for y in years}, out)


def _ids(matrix):
    return [IDS[int(v)] for v in np.asarray(matrix)[:, 0]]


def _mean(ids):
    return sum(LABEL[e] for e in ids) / len(ids)


def _reads(calls):
    return [(c["ids"], c["before"]) for c in calls["labels"]]


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _snapshot(out):
    return {p.name: p.read_bytes() for p in out.iterdir()}


def _out_of_memory():
    raise MemoryError


def _availability(avail, ends=None):
    """The test's own reading of each label's native availability: the day after its label end, overridden by avail.
    `ends` maps event id to label end; it defaults to the module's EVENTS."""
    if ends is None:
        ends = {e: end for e, _, _, end in EVENTS}
    days = {e: (date.fromisoformat(end) + timedelta(days=1)).isoformat() for e, end in ends.items()}
    return {**days, **avail}


def _install(monkeypatch, out, labels=LABEL, fit=None, real=False, avail=None):
    """Fake the label reader (labels available before the cutoff, by label end and native availability) and, unless
    real, the regression fit. `avail` overrides native availability per event id."""
    calls = {"labels": [], "fits": []}

    def read_labels(sealed, event_ids, labels_before):
        ends = {e["event_id"]: e["label_end"] for e in sealed["context"]["target_events"]}
        days = _availability(avail or {}, ends)
        kept = [e for e in event_ids if available_before(ends[e], labels_before)
                and printed_before(days[e], labels_before)]
        files = sorted(p.name for p in out.glob("arm_s_fit_*.json")) if out.is_dir() else []
        calls["labels"].append({"ids": list(event_ids), "before": labels_before, "returned": kept, "files": files})
        return pd.DataFrame({"event_id": kept, "absolute_move_pct": [labels[e] for e in kept],
                             "target_available_on": [days[e] for e in kept]})

    def fake_fit(recipe, train_x, y, eval_x, *, columns):
        calls["fits"].append({"recipe": recipe, "ids": _ids(train_x), "y": [float(v) for v in y],
                              "eval": _ids(eval_x)})
        if fit is not None:
            fit()
        return {"predictions": [float(np.mean(y))] * len(eval_x)}

    monkeypatch.setattr(arm_s, "training_labels", read_labels)
    if not real:
        monkeypatch.setattr(arm_s, "fit_regression_predictions", fake_fit)
    return calls


def _fixed_fit(monkeypatch, median, scale):
    """Replace each year's model fit with fixed per-event values; the z_q arithmetic is what is under test."""
    def fake(spec, sealed, config, ids, x, y, eval_x, targets, available):
        names = _ids(eval_x)
        return np.array([median.get(e, 0.0) for e in names]), np.array([scale.get(e, 1.0) for e in names])

    monkeypatch.setattr(arm_s, "_fit", fake)


@pytest.mark.parametrize("year", YEARS)
def test_median_and_scale_fits_use_absolute_error_at_the_years_own_configuration(monkeypatch, tmp_path, year):
    calls = _install(monkeypatch, tmp_path / "arm")
    _run(_sealed(), _spec(), [year], tmp_path / "arm")
    assert len(calls["fits"]) >= 2
    for call in calls["fits"]:
        params = call["recipe"]["estimator"]["params"]
        assert params["loss"] == "absolute_error" and params.get("quantile") is None
        assert [params["max_leaf_nodes"], params["min_samples_leaf"]] == CONFIG[year]


def test_inner_folds_fit_only_earlier_rows_labelled_before_the_held_out_year(monkeypatch, tmp_path):
    calls = _install(monkeypatch, tmp_path / "arm")
    _run(_sealed(ends={"e20b": "2021-06-01"}), _spec(), ["2024"], tmp_path / "arm")
    inner, scale, median = calls["fits"][:3], calls["fits"][3], calls["fits"][4]
    # e20b's label ends after 2021-01-01: absent from the 2021 inner fit, present from the 2022 inner fit on.
    assert [c["ids"] for c in inner] == [["e20a"], ["e20a", "e20b", "e21a", "e21b"], T23]
    # The earliest year (2020) gets no residual; each held row's target is |y - out-of-fold median|, never in-sample.
    assert scale["ids"] == ["e21a", "e21b", "e22a", "e22b", "e23a", "e23b", "e23c"]
    expect = [abs(LABEL[e] - _mean(["e20a"])) for e in ["e21a", "e21b"]]
    expect += [abs(LABEL[e] - _mean(["e20a", "e20b", "e21a", "e21b"])) for e in ["e22a", "e22b"]]
    expect += [abs(LABEL[e] - _mean(T23)) for e in E23]
    assert scale["y"] == pytest.approx(expect)
    assert scale["eval"] == E24
    assert median["ids"] == T24 and median["eval"] == E24
    assert median["y"] == [LABEL[e] for e in T24]


def _inner_fit(calls, held):
    """Training ids of the one fake fit that predicts the inner held-out year whose events are `held`."""
    return next(c["ids"] for c in calls["fits"] if c["eval"] == held)


def test_inner_fit_drops_a_label_available_in_the_held_out_year_until_the_next_inner_year(monkeypatch, tmp_path):
    # e22b's label ends 2022-12-05 (before 2023-01-01) but its native availability is 2023-02-10: the 2023 inner
    # fit must not see it, the 2024 inner fit must, and it stays a held-out row of its own year (2022).
    calls = _install(monkeypatch, tmp_path / "arm", avail={"e22b": "2023-02-10"})
    _run(_sealed(ends={"e22b": "2022-12-05"}), _spec(min_events=1), ["2025"], tmp_path / "arm")
    assert _inner_fit(calls, E23) == ["e20a", "e20b", "e21a", "e21b", "e22a"]
    assert _inner_fit(calls, E24) == T24
    assert [c["eval"] for c in calls["fits"] if "e22b" in c["eval"]] == [["e22a", "e22b"]]


def test_inner_fit_excludes_availability_on_jan_first_and_keeps_dec_thirty_first(monkeypatch, tmp_path):
    calls = _install(monkeypatch, tmp_path / "arm", avail={"e21a": "2023-01-01", "e21b": "2022-12-31"})
    _run(_sealed(), _spec(min_events=1), ["2025"], tmp_path / "arm")
    assert _inner_fit(calls, E23) == ["e20a", "e20b", "e21b", "e22a", "e22b"]
    assert _inner_fit(calls, E24) == T24


def _training_hash(monkeypatch, out, avail):
    calls = _install(monkeypatch, out, avail=avail)
    result = _run(_sealed(), _spec(min_events=1), ["2024"], out)
    return result["2024"]["receipt"]["inputs"]["training_availability_hash"], calls["labels"][0]["returned"]


def test_receipt_binds_each_training_labels_availability_in_training_order(monkeypatch, tmp_path):
    avail = {"e21a": "2023-01-02"}
    digest, kept = _training_hash(monkeypatch, tmp_path / "a", avail)
    days = _availability(avail)
    assert kept == T24
    assert digest == content_hash([[e, days[e]] for e in T24])
    changed, _ = _training_hash(monkeypatch, tmp_path / "b", {"e21a": "2023-01-03"})
    assert changed != digest


def test_no_out_of_fold_row_is_fit_failed_and_later_years_still_run(monkeypatch, tmp_path):
    late = {e: "2022-06-30" for e in T23}
    _install(monkeypatch, tmp_path / "arm")
    out = _run(_sealed(ends=late), _spec(), YEARS, tmp_path / "arm")
    assert out["2023"]["status"] == "fit_failed"
    assert out["2023"]["receipt"]["failure"] == "no out-of-fold residual"
    assert out["2023"]["scored"] == {} and out["2023"]["unscored"] == {}
    assert out["2024"]["status"] == out["2025"]["status"] == "ok"
    assert (tmp_path / "arm" / "arm_s_fit_2023.json").exists()


def test_first_year_is_entirely_unscored_and_reads_only_its_training_labels(monkeypatch, tmp_path):
    calls = _install(monkeypatch, tmp_path / "arm")
    out = _run(_sealed(), _spec(min_events=1), ["2023"], tmp_path / "arm")
    assert out["2023"]["scored"] == {}
    assert out["2023"]["unscored"] == dict.fromkeys(E23, "scale_burn_in")
    assert _reads(calls) == [(T23, "2023-01-01")]


def test_donor_window_is_the_twelve_months_before_each_month_start(monkeypatch, tmp_path):
    calls = _install(monkeypatch, tmp_path / "arm")
    _run(_sealed(**LATE), _spec(min_events=1), ["2023", "2024"], tmp_path / "arm")
    # e23a sits on the lower edge (2023-03-01); e24a on the upper edge of the 2024-03 window and is not its donor.
    assert _reads(calls) == [
        (T23, "2023-01-01"), (T24, "2024-01-01"),
        (["e23a", "e23b", "e23c"], "2024-03-01"),
        (["e23b", "e23c", "e24a"], "2024-05-01"),
        (["e24a", "e24b"], "2024-11-01"),
    ]
    assert calls["labels"][3]["returned"] == ["e23b", "e23c"]


def test_a_month_below_min_events_after_labels_burns_in_and_one_without_donors_reads_nothing(monkeypatch, tmp_path):
    calls = _install(monkeypatch, tmp_path / "arm")
    out = _run(_sealed(**LATE), _spec(min_events=3), ["2023", "2024"], tmp_path / "arm")
    assert _reads(calls)[2:] == [(["e23a", "e23b", "e23c"], "2024-03-01"), (["e23b", "e23c", "e24a"], "2024-05-01")]
    assert set(out["2024"]["scored"]) == {"e24a"}
    assert out["2024"]["unscored"] == {"e24b": "scale_burn_in", "e24c": "scale_burn_in"}


def test_exactly_min_events_labelled_donors_scores_the_month(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path / "arm")
    out = _run(_sealed(**LATE), _spec(min_events=2), ["2023", "2024"], tmp_path / "arm")
    assert set(out["2024"]["scored"]) == {"e24a", "e24b", "e24c"}
    assert out["2024"]["unscored"] == {}


def test_monthly_z_q_floors_scales_in_standardising_and_floors_the_quantiles(monkeypatch, tmp_path):
    median = {"e23a": -1.0, "e23b": 0.5, "e23c": 2.0, "e24a": -2.0}
    scale = {"e23a": 0.2, "e23b": -0.3, "e23c": 1.0, "e24a": 0.1}
    _fixed_fit(monkeypatch, median, scale)
    _install(monkeypatch, tmp_path / "arm")
    out = _run(_sealed(ends={"e24a": "2024-05-15"}), _spec(min_events=3), ["2023", "2024"], tmp_path / "arm")
    # Donors E23 (labels 2, 3, 4): residuals (y - median) / max(scale, 0.5) = 6, 5, 2. Their linear quantiles at the
    # spec levels (0.1 .. 0.9) are worked by hand below; e24a has scale 0.1, floored to 0.5 for its own raw.
    z = [2.6, 3.5, 4.4, 5.0, 5.2, 5.5, 5.8]
    raw = [-2.0 + 0.5 * v for v in z]
    assert set(out["2024"]["scored"]) == {"e24a"}
    assert out["2024"]["scored"]["e24a"]["raw"] == pytest.approx(raw)
    assert out["2024"]["scored"]["e24a"]["q"] == pytest.approx([max(v, 0.0) for v in raw])
    assert out["2024"]["unscored"] == {"e24b": "scale_burn_in", "e24c": "scale_burn_in"}
    assert out["2024"]["floor"] == {"events": 1, "events_bound": 1, "donors": 3, "donors_bound": 2}


def test_binding_counts_are_strict_and_donors_count_once_across_months(monkeypatch, tmp_path):
    _fixed_fit(monkeypatch, {}, {"e23a": 0.5, "e23b": -0.3, "e24a": 0.1})
    _install(monkeypatch, tmp_path / "arm")
    out = _run(_sealed(), _spec(min_events=1), ["2023", "2024"], tmp_path / "arm")
    # e23a at exactly 0.5 is not bound; e23b and e24a are. Donors used: E23 (2024-02), e23b and e23c and e24a
    # (2024-05), e24a and e24b (2024-11): five distinct.
    assert out["2024"]["floor"] == {"events": 3, "events_bound": 1, "donors": 5, "donors_bound": 2}
    assert out["2023"]["floor"] == {"events": 0, "events_bound": 0, "donors": 0, "donors_bound": 0}


def test_each_year_receipt_is_on_disk_before_any_read_of_its_events(monkeypatch, tmp_path):
    calls = _install(monkeypatch, tmp_path / "arm")
    _run(_sealed(), _spec(min_events=1), YEARS, tmp_path / "arm")
    assert any(set(E24) & set(c["ids"]) for c in calls["labels"])
    for call in calls["labels"]:
        for year in YEARS:
            if set(EVAL[year]) & set(call["ids"]):
                assert f"arm_s_fit_{year}.json" in call["files"], (year, call["before"])


def test_no_label_read_includes_an_event_dated_on_or_after_its_read_date(monkeypatch, tmp_path):
    calls = _install(monkeypatch, tmp_path / "arm")
    _run(_sealed(), _spec(min_events=1), YEARS, tmp_path / "arm")
    for call in calls["labels"]:
        assert all(DATE[e] < call["before"] for e in call["ids"]), call["before"]


def test_replay_refits_nothing_and_returns_identical_output(monkeypatch, tmp_path):
    out = tmp_path / "arm"
    _install(monkeypatch, out)
    first = _run(_sealed(), _spec(min_events=1), YEARS, out)
    before = _snapshot(out)
    calls = _install(monkeypatch, out)
    assert _run(_sealed(), _spec(min_events=1), YEARS, out) == first
    assert calls["fits"] == []
    assert _snapshot(out) == before


def test_changed_training_label_refuses_without_refit(monkeypatch, tmp_path):
    out = tmp_path / "arm"
    _install(monkeypatch, out)
    _run(_sealed(), _spec(min_events=1), YEARS, out)
    before = _snapshot(out)
    calls = _install(monkeypatch, out, labels={**LABEL, "e20a": 9.0})
    with pytest.raises(OpsError) as exc:
        _run(_sealed(), _spec(min_events=1), YEARS, out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "fit_receipt_changed")
    assert calls["fits"] == [] and _snapshot(out) == before


def _edit_raw(path):
    document = json.loads(path.read_text())
    document["raw"]["S"][0][0] = 99.0
    path.write_text(json.dumps(document, sort_keys=True))


CORRUPTIONS = {"raw_edited": _edit_raw, "invalid_json": lambda path: path.write_text("{not json")}


@pytest.mark.parametrize("corrupt", list(CORRUPTIONS))
def test_altered_receipt_refuses_as_unreadable_without_refit(monkeypatch, tmp_path, corrupt):
    out = tmp_path / "arm"
    _install(monkeypatch, out)
    _run(_sealed(), _spec(min_events=1), YEARS, out)
    CORRUPTIONS[corrupt](out / "arm_s_fit_2023.json")
    calls = _install(monkeypatch, out)
    with pytest.raises(OpsError) as exc:
        _run(_sealed(), _spec(min_events=1), YEARS, out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "fit_receipt_unreadable")
    assert calls["fits"] == []


@pytest.mark.parametrize("case", ["empty", "gap", "binding", "spec_hash", "event_ids"])
def test_admission_refuses_mismatched_raw_receipts_before_any_read_or_fit(monkeypatch, tmp_path, case):
    out, spec = tmp_path / "arm", _spec(min_events=1)
    receipts = {y: _receipt(y, spec) for y in YEARS}
    if case == "empty":
        receipts = {}
    elif case == "gap":
        del receipts["2024"]
    elif case == "binding":
        receipts["2024"]["inputs"]["sealed_binding"] = OTHER
        _sign(receipts["2024"])
    elif case == "spec_hash":
        receipts["2024"]["inputs"]["spec_hash"] = OTHER
        _sign(receipts["2024"])
    else:
        receipts["2024"]["event_ids"] = list(reversed(E24))
        _sign(receipts["2024"])
    calls = _install(monkeypatch, out)
    with pytest.raises(OpsError) as exc:
        arm_s.fit_arm_s(_sealed(), spec, receipts, out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "raw_receipt_mismatch")
    assert calls == {"labels": [], "fits": []} and not out.exists()


@pytest.mark.parametrize("edit", ["configuration", "added_key"])
def test_receipt_edited_after_hashing_is_a_mismatch_before_any_read_or_fit(monkeypatch, tmp_path, edit):
    out, spec = tmp_path / "arm", _spec(min_events=1)
    receipts = {y: _receipt(y, spec) for y in YEARS}
    if edit == "configuration":
        receipts["2024"]["configuration"] = [7, 300]
    else:
        receipts["2024"]["note"] = "added after hashing"
    calls = _install(monkeypatch, out)
    with pytest.raises(OpsError) as exc:
        arm_s.fit_arm_s(_sealed(), spec, receipts, out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "raw_receipt_mismatch")
    assert calls == {"labels": [], "fits": []} and not out.exists()


def test_receipts_round_tripped_through_json_are_admitted(monkeypatch, tmp_path):
    spec, out = _spec(min_events=1), tmp_path / "arm"
    _install(monkeypatch, out)
    receipts = json.loads(json.dumps({y: _receipt(y, spec) for y in YEARS}))
    results = arm_s.fit_arm_s(_sealed(), spec, receipts, out)
    assert [results[y]["status"] for y in YEARS] == ["ok"] * len(YEARS)


def test_plan_grid_mismatch_refuses_before_any_label_read_or_fit(monkeypatch, tmp_path):
    out, sealed = tmp_path / "arm", _sealed()
    sealed["plan"]["candidates"] = [c for c in sealed["plan"]["candidates"] if c["candidate_id"] != "h15_1000"]
    calls = _install(monkeypatch, out)
    with pytest.raises(OpsError) as exc:
        _run(sealed, _spec(min_events=1), YEARS, out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "v14_grid_mismatch")
    assert calls == {"labels": [], "fits": []} and not out.exists()


def _duplicated(sealed):
    sealed["rows"].append(dict(sealed["rows"][0]))
    return sealed


def test_duplicate_lead_one_row_refuses_on_a_fresh_run_with_no_fit(monkeypatch, tmp_path):
    out = tmp_path / "arm"
    calls = _install(monkeypatch, out)
    with pytest.raises(OpsError) as exc:
        _run(_duplicated(_sealed()), _spec(min_events=1), ["2023"], out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "panel_row_duplicate")
    assert calls["fits"] == [] and not out.exists()


def test_duplicate_lead_one_row_refuses_on_a_replay_with_no_fit_and_no_write(monkeypatch, tmp_path):
    out = tmp_path / "arm"
    _install(monkeypatch, out)
    _run(_sealed(), _spec(min_events=1), ["2023"], out)
    before = _snapshot(out)
    calls = _install(monkeypatch, out)
    with pytest.raises(OpsError) as exc:
        _run(_duplicated(_sealed()), _spec(min_events=1), ["2023"], out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "panel_row_duplicate")
    assert calls["fits"] == [] and _snapshot(out) == before


def test_infinite_feature_refuses_before_any_fit_and_writes_nothing(monkeypatch, tmp_path):
    sealed, out = _sealed(), tmp_path / "arm"
    sealed["rows"][0]["f_a"] = float("inf")
    calls = _install(monkeypatch, out)
    with pytest.raises(OpsError) as exc:
        _run(sealed, _spec(min_events=1), ["2023"], out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "feature_infinite")
    assert calls["fits"] == [] and not out.exists()


def test_memory_error_refuses_resource_limit_and_writes_nothing_for_that_year(monkeypatch, tmp_path):
    out = tmp_path / "arm"
    _install(monkeypatch, out, fit=_out_of_memory)
    with pytest.raises(OpsError) as exc:
        _run(_sealed(), _spec(min_events=1), ["2023"], out)
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not (out / "arm_s_fit_2023.json").exists()


def test_real_fit_end_to_end_on_a_synthetic_panel_gives_finite_outputs(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    events, rows, labels = [], [], {}
    for year in range(2019, 2024):
        for i in range(150):
            event_id, date = f"r{year}_{i:03d}", f"{year}-{1 + i % 12:02d}-15"
            b = float(rng.normal())
            events.append({"event_id": event_id, "ticker": "T", "event_date": date, "label_end": date})
            rows.append({"event_id": event_id, "f_a": float(rng.normal()), "f_b": b, "f_empty": None})
            labels[event_id] = 1.0 + abs(b) + 0.5 * float(rng.random())
    sealed = {"context": {"target_events": events}, "rows": rows, "model_columns": ["f_a", "f_b", "f_empty"],
              "dictionary_hash": DICT, "binding": BINDING, "plan": {"plan_id": "P1", "candidates": _candidates()}}
    ids = {y: [e["event_id"] for e in events if e["event_date"][:4] == y] for y in ("2022", "2023")}

    def folds(_sealed, years):
        return {y: {"training_labels_before": f"{y}-01-01", "evaluation_event_ids": ids[y],
                    "training_event_ids": [e["event_id"] for e in events if e["event_date"] < f"{y}-01-01"]}
                for y in years}

    monkeypatch.setattr(arm_s, "annual_folds", folds)
    _install(monkeypatch, tmp_path / "arm", labels=labels, real=True)
    spec = _spec(min_events=20)
    receipts = {y: _receipt(y, spec, event_ids=ids[y], configuration=[7, 300]) for y in ids}
    out = arm_s.fit_arm_s(sealed, spec, receipts, tmp_path / "arm")
    assert out["2022"]["status"] == out["2023"]["status"] == "ok"
    for year in ids:
        assert np.isfinite(np.asarray(out[year]["receipt"]["raw"]["S"], float)).all()
    assert out["2023"]["scored"]
    for outcome in out["2023"]["scored"].values():
        assert np.isfinite(outcome["raw"]).all() and np.isfinite(outcome["q"]).all()
