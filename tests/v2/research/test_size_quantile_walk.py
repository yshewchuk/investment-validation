"""EXP-190 Q3a part 2: walk-forward and fit receipts on synthetic folds; fake labels and fits (no data/, no store)."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sklearn

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import lib
from experiments import size_quantile_fits as fits
from experiments import size_quantile_walk as walk
from experiments.size_quantile_prereg import load_spec

SPEC, _ = load_spec()
LEVELS = list(SPEC["levels"])
FIXED = SPEC["grid"]["fixed"]
YEARS = ["2023", "2024", "2025"]
DICT = "sha256:" + "a" * 64
BINDING = "sha256:" + "b" * 64
OTHER = "sha256:" + "c" * 64
KEYS = ["7x300", "7x1000", "15x300", "15x1000"]
# Constant predictions per configuration: 15x1000 wins selection in every later year, 7x300 never does.
PRED = {"7x300": 0.0, "7x1000": 1.0, "15x300": 1.5, "15x1000": 3.0}
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
LABEL = {e: 1.0 + i % 5 for i, e in enumerate(IDS)}
T23 = ["e20a", "e20b", "e21a", "e21b", "e22a", "e22b"]
E23, E24, E25 = ["e23a", "e23b", "e23c"], ["e24a", "e24b", "e24c"], ["e25a", "e25b", "e25c"]
T24, T25 = T23 + E23, T23 + E23 + E24
EXPECT = {"2023": (T23, E23, "2023-01-01"), "2024": (T24, E24, "2024-01-01"), "2025": (T25, E25, "2025-01-01")}
SELECTION = {"2023": [], "2024": E23, "2025": E23 + E24}
CONFIG = {"2023": [7, 300], "2024": [15, 1000], "2025": [15, 1000]}


def _outer_folds():
    return [{"outer": {"year": y, "training_candidate_event_ids": ids}}
            for y, ids in (("2023", T23), ("2024", T24), ("2025", T25))]


def _candidates():
    return [{"candidate_id": f"h{leaves}_{leaf}", "feature_recipe": {"subset": "HMOSXLP"},
             "recipe": {"schema_version": "magnitude_ridge_hgb_recipe.v1", "model": "history_hgb",
                        "estimator": {"params": {**FIXED, "max_leaf_nodes": leaves, "min_samples_leaf": leaf,
                                                 "loss": "quantile", "quantile": 0.5}}}}
            for leaves in (7, 15) for leaf in (300, 1000)]


def _sealed(binding=BINDING, drop_row=None):
    return {
        "context": {"target_events": [{"event_id": e, "ticker": t, "event_date": d, "label_end": end}
                                      for e, t, d, end in EVENTS],
                    "eligibility_plan": {"random_windows": []}},
        "fold_events": [{"event_id": e, "ticker": t, "event_date": d, "forecasts": [{"source_session_ceiling": d}]}
                        for e, t, d, _ in EVENTS],
        "events": [{"event_id": e} for e in IDS],
        "evaluation_labels_before": "2026-01-01",
        "plan": {"plan_id": "P1", "candidates": _candidates(), "folds": _outer_folds()},
        "rows": [{"event_id": e, "f_a": float(i), "f_b": i / 10, "f_empty": None}
                 for i, e in enumerate(IDS) if e != drop_row],
        "model_columns": ["f_a", "f_b", "f_empty"],
        "dictionary_hash": DICT,
        "binding": binding,
    }


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _snapshot(out):
    return {p.name: p.read_bytes() for p in out.iterdir()}


def _pinball(labels, prediction):
    """Mean over events and levels of the pinball loss, written from its definition (not the production helper)."""
    total = 0.0
    for y in labels:
        for a in LEVELS:
            total += a * (y - prediction) if y >= prediction else (1 - a) * (prediction - y)
    return total / (len(labels) * len(LEVELS))


def _verifies(receipt):
    return content_hash({k: v for k, v in receipt.items() if k != "content_hash"}) == receipt["content_hash"]


def _out_of_memory(*_args):
    raise MemoryError


def _install(monkeypatch, out, labels=LABEL, fit=None):
    """Fake the label reader and the fit regression; record every call with the files present at that moment."""
    calls = {"labels": [], "fits": []}

    def read_labels(sealed, event_ids, labels_before):
        existing = sorted(p.name for p in out.glob("fit_*.json")) if out.is_dir() else []
        calls["labels"].append({"ids": list(event_ids), "before": labels_before, "existing": existing})
        return pd.DataFrame({"event_id": list(event_ids), "absolute_move_pct": [labels[e] for e in event_ids]})

    def fake_fit(recipe, train_x, y, eval_x, *, columns):
        params = recipe["estimator"]["params"]
        key = f"{params['max_leaf_nodes']}x{params['min_samples_leaf']}"
        calls["fits"].append({"key": key, "y": [float(v) for v in y], "train_x": np.array(train_x),
                              "n": len(eval_x)})
        if fit is not None:
            return fit(key, len(eval_x), calls["labels"][-1]["ids"])
        return {"predictions": [PRED[key]] * len(eval_x)}

    monkeypatch.setattr(walk, "training_labels", read_labels)
    monkeypatch.setattr(fits, "fit_regression_predictions", fake_fit)
    return calls


def _rewrite(path, **changes):
    """Edit a receipt's body and recompute its content hash, so only the edit (not the hash) is what changed."""
    document = json.loads(path.read_text())
    document.pop("content_hash")
    document.update(changes)
    path.write_text(json.dumps({**document, "content_hash": content_hash(document)}, sort_keys=True))


def _edit_raw(path):
    document = json.loads(path.read_text())
    document["raw"]["7x300"][0][0] = 99.0
    path.write_text(json.dumps(document, sort_keys=True))


def _drop_hash(path):
    document = json.loads(path.read_text())
    path.write_text(json.dumps({k: v for k, v in document.items() if k != "content_hash"}))


def _directory(path):
    path.unlink()
    path.mkdir()


CORRUPTIONS = {
    "raw_edited": _edit_raw,
    "invalid_json": lambda path: path.write_text("{not json"),
    "json_list": lambda path: path.write_text("[]"),
    "missing_hash": _drop_hash,
    "directory": _directory,
}


def _boom_in_2024(key, n, training_ids):
    if set(training_ids) == set(T24):
        raise ValueError("boom")
    return {"predictions": [PRED[key]] * n}


def test_three_years_chain_receipts_and_see_only_their_own_labels(monkeypatch, tmp_path):
    out = tmp_path / "walk"
    calls = _install(monkeypatch, out)
    sealed = _sealed()
    receipts = walk.fit_walk_forward(sealed, SPEC, YEARS, out)
    assert list(receipts) == YEARS
    assert [(c["ids"], c["before"]) for c in calls["labels"]] == [
        (T23, "2023-01-01"),
        (E23, "2024-01-01"), (T24, "2024-01-01"),
        (E23 + E24, "2025-01-01"), (T25, "2025-01-01"),
    ]
    for index, year in enumerate(YEARS):
        train, evaluation, cutoff = EXPECT[year]
        receipt = receipts[year]
        assert receipt["schema_version"] == walk.SCHEMA
        assert receipt["status"] == "ok" and receipt["failure"] is None
        assert receipt["levels"] == LEVELS
        assert receipt["configuration"] == CONFIG[year]
        expected = {k: _pinball([LABEL[e] for e in SELECTION[year]], PRED[k]) for k in KEYS} if SELECTION[year] else {}
        assert receipt["selection_losses"] == {k: pytest.approx(v) for k, v in expected.items()}
        assert receipt["arm_id"] == fits.arm_id(SPEC, sealed, "Q_raw", tuple(CONFIG[year]))
        assert receipt["event_ids"] == evaluation
        assert set(receipt["raw"]) == set(KEYS)
        for key in KEYS:
            assert np.asarray(receipt["raw"][key]).shape == (len(evaluation), 7)
            assert np.allclose(receipt["raw"][key], PRED[key])
        assert receipt["inputs"] == {
            "sealed_binding": BINDING, "spec_hash": lib.spec_hash(SPEC), "sklearn_version": sklearn.__version__,
            "year": year, "training_labels_before": cutoff,
            "training_labels_hash": content_hash([[e, LABEL[e]] for e in train]),
            "selection_labels_hash": content_hash([[e, LABEL[e]] for e in SELECTION[year]]),
            "evaluation_event_ids_hash": content_hash(evaluation),
            "prior_receipt_hashes": [receipts[y]["content_hash"] for y in YEARS[:index]],
        }
        assert _verifies(receipt)
    for index, year in enumerate(YEARS):
        block = calls["fits"][28 * index: 28 * (index + 1)]
        train, evaluation, _ = EXPECT[year]
        assert len(block) == 28
        for call in block:
            assert call["y"] == [LABEL[e] for e in train]
            assert call["train_x"][:, 0].tolist() == [float(IDS.index(e)) for e in train]
            assert call["n"] == len(evaluation)


def test_each_year_is_on_disk_before_a_later_year_reads_its_labels(monkeypatch, tmp_path):
    out = tmp_path / "walk"
    calls = _install(monkeypatch, out)
    walk.fit_walk_forward(_sealed(), SPEC, YEARS, out)
    assert any(set(E23) & set(c["ids"]) for c in calls["labels"])
    for call in calls["labels"]:
        for year in YEARS:
            if set(EXPECT[year][1]) & set(call["ids"]):
                assert f"fit_{year}.json" in call["existing"], (year, call["before"])


def test_replay_refits_nothing_and_rewrites_nothing(monkeypatch, tmp_path):
    out = tmp_path / "walk"
    _install(monkeypatch, out)
    first = walk.fit_walk_forward(_sealed(), SPEC, YEARS, out)
    before = _snapshot(out)
    calls = _install(monkeypatch, out)
    assert walk.fit_walk_forward(_sealed(), SPEC, YEARS, out) == first
    assert calls["fits"] == []
    assert _snapshot(out) == before
    assert sorted(before) == [f"fit_{y}.json" for y in YEARS]


def test_extending_a_run_replays_the_shared_years_and_fits_only_the_new_one(monkeypatch, tmp_path):
    out = tmp_path / "walk"
    _install(monkeypatch, out)
    short = walk.fit_walk_forward(_sealed(), SPEC, YEARS[:2], out)
    before = _snapshot(out)
    calls = _install(monkeypatch, out)
    full = walk.fit_walk_forward(_sealed(), SPEC, YEARS, out)
    assert list(full) == YEARS and full["2023"] == short["2023"] and full["2024"] == short["2024"]
    assert len(calls["fits"]) == 28
    assert all(c["y"] == [LABEL[e] for e in T25] for c in calls["fits"])
    after = _snapshot(out)
    assert all(after[name] == data for name, data in before.items())
    assert sorted(after) == [f"fit_{y}.json" for y in YEARS]


@pytest.mark.parametrize("change", ["training_label", "selection_label", "binding", "sklearn", "earlier_receipt"])
def test_changed_inputs_refuse_without_refit(monkeypatch, tmp_path, change):
    out = tmp_path / "walk"
    _install(monkeypatch, out)
    walk.fit_walk_forward(_sealed(), SPEC, YEARS, out)
    labels, binding = dict(LABEL), BINDING
    if change == "training_label":
        labels["e20a"] = 9.0
    elif change == "selection_label":
        labels["e23a"] = 9.0
    elif change == "binding":
        binding = OTHER
    elif change == "sklearn":
        monkeypatch.setattr(sklearn, "__version__", "0.0.0-test")
    else:
        _rewrite(out / "fit_2023.json", arm_id="changed")
    before = _snapshot(out)
    calls = _install(monkeypatch, out, labels=labels)
    with pytest.raises(OpsError) as exc:
        walk.fit_walk_forward(_sealed(binding=binding), SPEC, YEARS, out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "fit_receipt_changed")
    assert calls["fits"] == []
    assert _snapshot(out) == before


@pytest.mark.parametrize("corrupt", list(CORRUPTIONS))
def test_altered_receipt_refuses_as_unreadable_without_refit(monkeypatch, tmp_path, corrupt):
    out = tmp_path / "walk"
    _install(monkeypatch, out)
    walk.fit_walk_forward(_sealed(), SPEC, YEARS, out)
    CORRUPTIONS[corrupt](out / "fit_2023.json")
    calls = _install(monkeypatch, out)
    with pytest.raises(OpsError) as exc:
        walk.fit_walk_forward(_sealed(), SPEC, YEARS, out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "fit_receipt_unreadable")
    assert calls["fits"] == []


def test_fit_failure_stops_the_walk_and_replays_without_refit(monkeypatch, tmp_path, capsys):
    out = tmp_path / "walk"
    calls = _install(monkeypatch, out, fit=_boom_in_2024)
    receipts = walk.fit_walk_forward(_sealed(), SPEC, YEARS, out)
    assert list(receipts) == ["2023", "2024"]
    failed = receipts["2024"]
    assert failed["status"] == "fit_failed" and "boom" in failed["failure"] and failed["raw"] == {}
    assert "fit_failed: boom" in capsys.readouterr().err
    assert all(c["before"] != "2025-01-01" for c in calls["labels"])
    assert not (out / "fit_2025.json").exists()
    again_calls = _install(monkeypatch, out)
    assert walk.fit_walk_forward(_sealed(), SPEC, YEARS, out) == receipts
    assert again_calls["fits"] == []
    assert all(c["before"] != "2025-01-01" for c in again_calls["labels"])
    assert not (out / "fit_2025.json").exists()


@pytest.mark.parametrize("where", ["fit", "labels"])
def test_memory_error_refuses_resource_limit_with_nothing_written(monkeypatch, tmp_path, where):
    out = tmp_path / "walk"
    _install(monkeypatch, out, fit=_out_of_memory if where == "fit" else None)
    if where == "labels":
        monkeypatch.setattr(walk, "training_labels", _out_of_memory)
    with pytest.raises(OpsError) as exc:
        walk.fit_walk_forward(_sealed(), SPEC, YEARS, out)
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not out.exists()


def test_oversize_receipt_refuses_before_touching_disk(monkeypatch, tmp_path):
    out = tmp_path / "walk"
    _install(monkeypatch, out)
    monkeypatch.setattr(walk, "MAX_RECEIPT_BYTES", 100)
    with pytest.raises(OpsError) as exc:
        walk.fit_walk_forward(_sealed(), SPEC, YEARS, out)
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not out.exists()


def test_decode_memory_error_refuses_before_any_disk_write(monkeypatch, tmp_path):
    out = tmp_path / "walk"
    _install(monkeypatch, out)
    monkeypatch.setattr(walk, "json", SimpleNamespace(dumps=json.dumps, loads=_out_of_memory))
    with pytest.raises(OpsError) as exc:
        walk.fit_walk_forward(_sealed(), SPEC, ["2023"], out)
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not out.exists()


@pytest.mark.parametrize("other_binding, outcome",[(BINDING, "replay"), (OTHER, "refuse")])
def test_link_race_returns_the_existing_receipt_or_refuses(monkeypatch, tmp_path, other_binding, outcome):
    other = tmp_path / "other"
    _install(monkeypatch, other)
    walk.fit_walk_forward(_sealed(binding=other_binding), SPEC, ["2023"], other)
    written = (other / "fit_2023.json").read_bytes()
    out = tmp_path / "walk"

    def racing_link(_src, dst):
        dst.write_bytes(written)
        raise FileExistsError(str(dst))

    _install(monkeypatch, out)
    monkeypatch.setattr(walk.os, "link", racing_link)
    if outcome == "replay":
        assert walk.fit_walk_forward(_sealed(), SPEC, ["2023"], out)["2023"] == json.loads(written)
    else:
        with pytest.raises(OpsError) as exc:
            walk.fit_walk_forward(_sealed(), SPEC, ["2023"], out)
        assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "fit_receipt_changed")
    assert sorted(p.name for p in out.iterdir()) == ["fit_2023.json"]


def test_non_contiguous_years_refuse_before_any_label_read_or_fit(monkeypatch, tmp_path):
    out = tmp_path / "walk"
    calls = _install(monkeypatch, out)
    with pytest.raises(OpsError) as exc:
        walk.fit_walk_forward(_sealed(), SPEC, ["2022", "2024"], out)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "years_not_contiguous")
    assert calls == {"labels": [], "fits": []}
    assert not out.exists()


def test_grid_mismatch_refuses_before_annual_folds_run(monkeypatch, tmp_path):
    sealed = _sealed()
    sealed["plan"]["candidates"] = [c for c in sealed["plan"]["candidates"] if c["candidate_id"] != "h15_1000"]

    def folds_must_not_run(*_args):
        raise AssertionError("annual_folds ran before bind_grid refused")

    monkeypatch.setattr(walk, "annual_folds", folds_must_not_run)
    with pytest.raises(OpsError) as exc:
        walk.fit_walk_forward(sealed, SPEC, YEARS, tmp_path / "walk")
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "v14_grid_mismatch")


def test_fit_path_refusal_propagates_and_writes_no_receipt(monkeypatch, tmp_path):
    out = tmp_path / "walk"
    _install(monkeypatch, out)
    with pytest.raises(OpsError) as exc:
        walk.fit_walk_forward(_sealed(drop_row="e23a"), SPEC, YEARS, out)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "panel_row_missing")
    assert not (out / "fit_2023.json").exists()
