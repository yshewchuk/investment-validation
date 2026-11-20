"""EXP-188 SY1: synthetic-predictor generator, guards and tuning guard. Synthetic fixtures only; no repository."""
import ast
import copy
import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml

from engine.v2.ops.errors import OpsError
from experiments import size_pnl_readers as readers
from experiments import size_pnl_synthetic as syn
from experiments import size_pnl_tuning as tuning
from experiments.size_pnl_shortvol import Refusal
from experiments.size_pnl_tuning import SYNTHETIC_PREFIX

ROOT = pathlib.Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "experiments" / "EXP-188_size_decision_pnl_rescore" / "spec.yaml"
with open(SPEC_PATH, encoding="utf-8") as fh:
    SPEC = yaml.safe_load(fh)
ARM = syn.arm_from_spec(SPEC)
OUT_OF_SCOPE = "SYNTHETIC_OUT_OF_SCOPE"
CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
DENIED = "HOLDOUT_ACCESS_DENIED"


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _events(n, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    return pd.DataFrame({"event_id": [f"ev{i:05d}" for i in range(n)], "event_date": pd.to_datetime(days)})


def _labels(events, move):
    """Label frame indexed by event_id with float abs_move, crush_pct_iv30 and event_date."""
    return pd.DataFrame({"abs_move": np.asarray(move, dtype=float), "crush_pct_iv30": -2.0,
                         "event_date": pd.to_datetime(events["event_date"]).to_numpy()},
                        index=pd.Index(events["event_id"], name="event_id"))


def _admitted(events):
    return readers.Admitted({"events": events.copy(), "excluded": {"random_3pct": 1}})


def _production(events, crush=-1.0):
    frame = pd.DataFrame({"pred_move": 1.0, "pred_crush": crush,
                          "event_date": pd.to_datetime(events["event_date"]).to_numpy()},
                         index=pd.Index(events["event_id"], name="event_id"))
    return readers.Production({"frame": frame, "binding": "pb", "table_hash": "t"})


def _world(n=300, seed=0):
    events = _events(n, seed)
    move = np.random.default_rng(seed + 1).uniform(0.5, 5.0, n)
    return _admitted(events), _labels(events, move), _production(events)


def _by(out, family, **params):
    for entry in out.values():
        if entry["family"] == family and entry["params"] == params:
            return entry["table"]
    raise KeyError((family, params))


def _fetch_from(labels, calls=None):
    def fetch(columns, ids):
        if calls is not None:
            calls.append(list(ids))
        return labels.loc[list(ids), list(columns)].reset_index()
    return fetch


# 1. predictor list and ids
def test_default_arm_has_19_predictors_with_stable_prefixed_ids():
    preds = syn.predictors(ARM)
    assert len(preds) == 19
    ids = [syn.model_id(ARM, p) for p in preds]
    assert len(set(ids)) == 19 and all(i.startswith(SYNTHETIC_PREFIX) for i in ids)
    assert ids == [syn.model_id(ARM, p) for p in preds]
    other = {**ARM, "seed_tag": "exp188-synth-v2"}
    assert all(syn.model_id(other, p) != syn.model_id(ARM, p) for p in preds)


# 2. definitions
def test_definitions_on_a_300_event_fixture():
    admitted, labels, production = _world(300)
    out = syn.generate(admitted, labels, production, ARM)
    y = labels["abs_move"]
    oracle = _by(out, "oracle")
    np.testing.assert_allclose(oracle["pred_move"], y.loc[oracle.index])
    z_ref = None
    for s in ARM["spread_ladder"]:
        med = _by(out, "median_noise", s=s)
        mean = _by(out, "mean_noise", s=s)
        reg = _by(out, "regression_direction", s=s)
        yy = y.loc[med.index].to_numpy()
        z_med = np.log(med["pred_move"].to_numpy() / yy) / s
        np.testing.assert_allclose(np.log(mean["pred_move"].to_numpy() / yy) / s + s / 2, z_med, atol=1e-9)
        np.testing.assert_allclose(np.log(reg["pred_move"].to_numpy() / yy) / s - s / 2, z_med, atol=1e-9)
        np.testing.assert_allclose(mean["pred_move"], med["pred_move"] * np.exp(-s * s / 2), rtol=1e-9)
        np.testing.assert_allclose(reg["pred_move"], med["pred_move"] * np.exp(s * s / 2), rtol=1e-9)
        if z_ref is None:
            z_ref = z_med
        np.testing.assert_allclose(z_med, z_ref, atol=1e-9)
    np.testing.assert_allclose(_by(out, "level_shift", shift=0.1)["pred_move"], 1.1 * y.loc[oracle.index])
    np.testing.assert_allclose(_by(out, "level_shift", shift=-0.1)["pred_move"], 0.9 * y.loc[oracle.index])


def _expanding_prior(labels):
    dates = pd.to_datetime(labels["event_date"])
    starts = dates.dt.to_period("Q").dt.start_time
    prior = {}
    for eid, start in zip(labels.index, starts):
        before = labels.loc[dates < start, "abs_move"]
        prior[eid] = before.mean() if len(before) else np.nan
    return pd.Series(prior)


def test_shrinkage_matches_expanding_prior_and_constant_is_per_quarter():
    admitted, labels, production = _world(300)
    out = syn.generate(admitted, labels, production, ARM)
    prior = _expanding_prior(labels)
    y = labels["abs_move"]
    half = _by(out, "shrinkage", L=0.5)
    np.testing.assert_allclose(half["pred_move"], 0.5 * y.loc[half.index] + 0.5 * prior.loc[half.index])
    null = _by(out, "shrinkage", L=1.0)
    quarters = pd.to_datetime(null["event_date"]).dt.to_period("Q")
    assert (null.groupby(quarters)["pred_move"].nunique() == 1).all()
    first = pd.to_datetime(labels["event_date"]) < pd.Timestamp("2023-04-01")
    for L in ARM["shrinkage_L"]:
        entry = next(e for e in out.values() if e["family"] == "shrinkage" and e["params"] == {"L": L})
        assert not set(entry["table"].index) & set(labels.index[first.to_numpy()])
        assert entry["dropped"]["undetermined_mean"] == int(first.sum())
        assert len(entry["table"]) + sum(entry["dropped"].values()) == len(labels)


# 3. statistics
def test_large_sample_spread_and_mean_unbiasedness():
    admitted, labels, production = _world(20000, seed=3)
    out = syn.generate(admitted, labels, production, ARM)
    y = labels["abs_move"].loc[_by(out, "median_noise", s=0.4).index].to_numpy()
    med = _by(out, "median_noise", s=0.4)["pred_move"].to_numpy()
    mean = _by(out, "mean_noise", s=0.4)["pred_move"].to_numpy()
    assert abs(np.log(med / y).std() - 0.4) < 0.02
    assert abs(np.mean(mean / y) - 1.0) < 0.02


# 4. metrics
@pytest.mark.parametrize("family, params", [("oracle", {}), ("median_noise", {"s": 0.6})])
def test_metrics_match_numpy_recomputation(family, params):
    admitted, labels, production = _world(300)
    out = syn.generate(admitted, labels, production, ARM)
    entry = next(e for e in out.values() if e["family"] == family and e["params"] == params)
    table = entry["table"]
    err = table["pred_move"].to_numpy() - labels.loc[table.index, "abs_move"].to_numpy()
    assert entry["metrics"]["n"] == len(table)
    assert entry["metrics"]["rmse"] == pytest.approx(float(np.sqrt(np.mean(err ** 2))))
    assert entry["metrics"]["mae"] == pytest.approx(float(np.mean(np.abs(err))))
    if family == "oracle":
        assert entry["metrics"]["rmse"] == 0.0 and entry["metrics"]["mae"] == 0.0
    else:
        assert entry["metrics"]["rmse"] >= entry["metrics"]["mae"]


def test_hand_computed_fixed_fixture_metrics():
    events = pd.DataFrame({"event_id": ["fx0", "fx1", "fx2"],
                           "event_date": pd.to_datetime(["2023-01-10", "2023-01-20", "2023-02-15"])})
    labels = _labels(events, [2.0, 4.0, 6.0])
    out = syn.generate(_admitted(events), labels, _production(events), ARM)
    oracle = next(e for e in out.values() if e["family"] == "oracle")
    assert oracle["metrics"]["rmse"] == pytest.approx(0.0) and oracle["metrics"]["mae"] == pytest.approx(0.0)
    shift = next(e for e in out.values() if e["family"] == "level_shift" and e["params"] == {"shift": 0.1})
    np.testing.assert_allclose(shift["table"]["pred_move"].to_numpy(), [2.2, 4.4, 6.6])
    assert shift["metrics"]["n"] == 3
    assert shift["metrics"]["mae"] == pytest.approx(0.4)
    assert shift["metrics"]["rmse"] == pytest.approx(0.4320493798938574)


# 5. guards
def test_event_dated_2025_refuses_and_generates_nothing():
    events = _events(50)
    events.loc[0, "event_date"] = pd.Timestamp("2025-03-01")
    labels = _labels(events, np.ones(50))
    with pytest.raises(OpsError) as exc:
        syn.generate(_admitted(events), labels, _production(events), ARM)
    assert _refusal(exc) == (DENIED, OUT_OF_SCOPE)
    assert exc.value.problem.details["why"] == "date_out_of_scope"
    assert exc.value.problem.details["events"] == 1


def test_event_dated_2022_refuses():
    events = _events(50)
    events.loc[0, "event_date"] = pd.Timestamp("2022-12-31")
    labels = _labels(events, np.ones(50))
    with pytest.raises(OpsError) as exc:
        syn.generate(_admitted(events), labels, _production(events), ARM)
    assert _refusal(exc) == (DENIED, OUT_OF_SCOPE)
    assert exc.value.problem.details["why"] == "date_out_of_scope"


def test_event_dated_2024_12_31_afternoon_is_accepted():
    events = _events(50)
    events.loc[0, "event_date"] = pd.Timestamp("2024-12-31 15:30")
    labels = _labels(events, np.ones(50))
    out = syn.generate(_admitted(events), labels, _production(events), ARM)
    assert len(out) == 19


def test_label_id_not_admitted_refuses():
    admitted, labels, production = _world(40)
    labels = labels.rename(index={labels.index[0]: "zz"})
    with pytest.raises(OpsError) as exc:
        syn.generate(admitted, labels, production, ARM)
    assert _refusal(exc) == (DENIED, OUT_OF_SCOPE)
    assert exc.value.problem.details["why"] == "event_not_admitted"


def test_holdout_event_is_never_read_or_generated():
    admitted, labels, production = _world(40)
    calls = []
    with pytest.raises(OpsError) as exc:
        readers.read_labels(_fetch_from(labels, calls), admitted, event_ids=["h1"])
    assert _refusal(exc) == (DENIED, "key_not_admitted")
    assert calls == []
    read = readers.read_labels(_fetch_from(labels), admitted)
    out = syn.generate(admitted, read, production, ARM)
    for entry in out.values():
        assert "h1" not in entry["table"].index


# 6. non-issued inputs and malformed labels
def test_plain_dict_admission_refuses():
    admitted, labels, production = _world(20)
    plain = {"events": admitted["events"], "excluded": {}}
    with pytest.raises(OpsError) as exc:
        syn.generate(plain, labels, production, ARM)
    assert _refusal(exc) == (DENIED, "admission_not_issued")


def test_plain_dict_production_refuses():
    admitted, labels, production = _world(20)
    plain = {"frame": production["frame"], "binding": "pb", "table_hash": "t"}
    with pytest.raises(OpsError) as exc:
        syn.generate(admitted, labels, plain, ARM)
    assert _refusal(exc) == (CONFLICT, "production_not_issued")


@pytest.mark.parametrize("make", [
    lambda lab: lab.drop(columns=["abs_move"]),
    lambda lab: lab.iloc[:0],
    lambda lab: pd.concat([lab, lab.iloc[:1]]),
    lambda lab: lab.set_axis(range(len(lab))),
    lambda lab: None,
])
def test_malformed_labels_refuse(make):
    admitted, labels, production = _world(20)
    with pytest.raises(OpsError) as exc:
        syn.generate(admitted, make(labels), production, ARM)
    assert _refusal(exc) == (CONFLICT, "synthetic_labels_malformed")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_abs_move_label_refuses(bad):
    admitted, labels, production = _world(20)
    labels.iloc[0, labels.columns.get_loc("abs_move")] = bad
    with pytest.raises(OpsError) as exc:
        syn.generate(admitted, labels, production, ARM)
    assert _refusal(exc) == (CONFLICT, "synthetic_labels_malformed")


# 7. missing crush and zero labels
def test_missing_crush_dropped_and_counted_zero_label_kept():
    events = _events(60)
    move = np.linspace(1.0, 3.0, 60)
    move[0] = 0.0
    labels = _labels(events, move)
    production = _production(events)
    production["frame"].iloc[1:6, production["frame"].columns.get_loc("pred_crush")] = np.nan
    out = syn.generate(_admitted(events), labels, production, ARM)
    entry = next(e for e in out.values() if e["family"] == "median_noise" and e["params"] == {"s": 0.4})
    assert entry["dropped"]["no_crush_forecast"] == 5
    assert not set(labels.index[1:6]) & set(entry["table"].index)
    assert len(entry["table"]) + sum(entry["dropped"].values()) == len(labels)
    oracle = _by(out, "oracle")
    assert labels.index[0] in oracle.index and oracle.loc[labels.index[0], "pred_move"] == 0.0


# 8. arm_from_spec refusals
def _arm_with(**changes):
    spec = copy.deepcopy(SPEC)
    spec["synthetic_arm"].update(changes)
    return spec


@pytest.mark.parametrize("spec", [
    {k: v for k, v in SPEC.items() if k != "synthetic_arm"},
    _arm_with(spread_ladder=[]),
    _arm_with(families=["oracle", "bogus"]),
    _arm_with(inner_years=[True, 2024]),
    _arm_with(shrinkage_L=[0.0, 0.5]),
    _arm_with(shrinkage_L=[1.5]),
    _arm_with(level_shift=[-1.0, 0.1]),
    _arm_with(families=["oracle", "oracle"]),
])
def test_arm_from_spec_refuses_malformed_blocks(spec):
    with pytest.raises(OpsError) as exc:
        syn.arm_from_spec(spec)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "synthetic_arm_invalid")


# 9. tuning guard
def test_synthetic_model_cannot_be_tuned_or_receipted(tmp_path):
    mid = syn.model_id(ARM, syn.predictors(ARM)[1])
    with pytest.raises(Refusal) as exc:
        tuning.tune([], mid)
    assert exc.value.details["reason"] == "SYNTHETIC_IN_TUNING"
    path = tmp_path / "tuning_x.json"
    with pytest.raises(Refusal) as exc:
        tuning.write_receipt(path, {"stage": "tuning", "model_id": mid, "c": 1.0})
    assert exc.value.details["reason"] == "SYNTHETIC_IN_TUNING"
    assert not path.exists()


def test_real_model_ids_and_inputs_receipts_still_work(tmp_path):
    with pytest.raises(ValueError, match="items must not be empty"):
        tuning.tune([], "M1")
    out = tuning.write_receipt(tmp_path / "inputs.json", {"stage": "inputs"})
    assert out["stage"] == "inputs" and "content_hash" in out
    real = tuning.write_receipt(tmp_path / "tuning_m1.json", {"stage": "tuning", "model_id": "M1", "c": 1.0})
    assert real["model_id"] == "M1"


# 10. independence
ROSTER_MODULES = ("size_pnl_driver", "size_pnl_inputs", "size_pnl_readers", "size_pnl_pool",
                  "size_pnl_tuning", "size_pnl_report")


def _imported_names(source):
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            found |= {node.module or ""} | {alias.name for alias in node.names}
    return found


@pytest.mark.parametrize("name", ROSTER_MODULES)
def test_roster_modules_never_import_the_synthetic_module(name):
    source = (ROOT / "experiments" / f"{name}.py").read_text(encoding="utf-8")
    assert not any("size_pnl_synthetic" in n for n in _imported_names(source))
