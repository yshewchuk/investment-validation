"""EXP-188 Amendment 3: the G_PROD gate arm. Synthetic fixtures only; no repository, no real artifact."""
import hashlib
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from engine.v2.ops.errors import OpsError
from experiments import size_pnl_gate_fold as fold, size_pnl_gprod as gp, size_pnl_ladder as lad, size_pnl_straddle as sd

FEATURES = [f"x{i}" for i in range(45)] + list(gp.MODEL_COLS)
START = pd.Timestamp("2025-01-01")
EMPTY = pd.DataFrame({"event_date": pd.Series([], dtype=object), "edge": pd.Series([], dtype=float)})


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


# ---- chunk helpers (as in test_size_pnl_straddle.py) -----------------------

def pair(k, *, call_e=(2.0, 2.2), put_e=(1.8, 2.0), call_x=(1.0, 1.4, 1), put_x=(0.6, 1.0, 1)):
    """Put and call at strike ``k``: entry (bid, ask), exit (bid, ask, rung)."""
    return [(0, k, *put_e, *put_x), (1, k, *call_e, *call_x)]


def rec(eid, spot, legs):
    cols = list(zip(*legs))
    return {"event_id": eid, "expiry": np.datetime64("2024-03-15"), "spot_entry": spot, "spot_exit": spot,
            "dte_entry": 30, "dte_exit": 29, "right": cols[0], "strike": cols[1], "bid_e": cols[2],
            "ask_e": cols[3], "bid_x": cols[4], "ask_x": cols[5], "exit_rung": cols[6]}


def good(*eids):
    """Chunk with one healthy ATM straddle per event id (entry relative spread ~0.10)."""
    return lad._pack([rec(e, 100.0, pair(100)) for e in eids], {})


# ---- module helpers ---------------------------------------------------------

def ev(eid, edge, *, mcap=20e9, implied=5.0, date="2025-01-10"):
    return {"event_id": eid, "event_date": pd.Timestamp(date), "pred_abs_move": 5.0 + edge, "implied_move": implied,
            "mcap_usd": mcap}


def fold_payload():
    """A linear fold whose prediction is 0.05 * forecast_edge, so the registered threshold needs an edge of about 1.018."""
    X = np.random.default_rng(0).normal(size=(80, 50))
    y = 0.05 * X[:, FEATURES.index("forecast_edge")]
    model = LinearRegression().fit(X, y)
    return {"model": model, "features": FEATURES, "threshold": fold.REGISTERED_THRESHOLD, "seed": 1, "train_through": 2024,
            "recipe_hash": "r"}


def inputs_frame(ids):
    """Every feature column 0.0 (so the 45 production inputs are 0.0)."""
    return pd.DataFrame({"event_id": list(ids), **{feature: 0.0 for feature in FEATURES}})


def _sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def write_gate_files(tmp_path, payload=None, inputs=None):
    fold_path, inputs_path = tmp_path / "fold.joblib", tmp_path / "inputs.parquet"
    joblib.dump(fold_payload() if payload is None else payload, str(fold_path))
    (inputs_frame(["E1", "E2"]) if inputs is None else inputs).to_parquet(inputs_path)
    return {"gate_fold": {"path": str(fold_path), "sha256": _sha(fold_path)},
            "gate_inputs": {"path": str(inputs_path), "sha256": _sha(inputs_path)}}


def history(n=300):
    """A pool of ``n`` paired-error rows: move errors are N(0, 1.5) around the forecast; crush labels are constant."""
    index = pd.Index([f"H{i}" for i in range(n)])
    pred_move = 5.0 + (np.arange(n) % 7) * 0.3
    table = pd.DataFrame({"event_date": pd.Timestamp("2024-03-01") + pd.to_timedelta(np.arange(n) % 300, unit="D"),
                          "pred_move": pred_move, "pred_crush": -15.0}, index=index)
    rng = np.random.default_rng(3)
    labels = pd.DataFrame({"abs_move": pred_move + rng.normal(0, 1.5, n), "crush_pct_iv30": -15.0}, index=index)
    return table, labels


class Recorder:
    """A stub fold model: records the rows it is asked to predict and returns a constant."""
    def predict(self, X):
        self.seen = np.array(X)
        return np.full(len(X), 0.5)


def _ten_gate(tmp_path):
    """The gate for the decision tests: 'hi', 'lo', 'small' have healthy inputs; 'nan' has a NaN input."""
    inputs = inputs_frame(["hi", "lo", "nan", "small"])
    inputs.loc[inputs["event_id"] == "nan", "x3"] = np.nan
    return gp.load_gate(write_gate_files(tmp_path, inputs=inputs))


# ---- load_gate --------------------------------------------------------------

def test_load_gate_returns_none_without_entries():
    assert gp.load_gate({}) is None


@pytest.mark.parametrize("key", ["gate_fold", "gate_inputs"])
def test_load_gate_refuses_one_entry_only(key):
    with pytest.raises(OpsError) as exc:
        gp.load_gate({key: {"path": "x", "sha256": "0" * 64}})
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "gate_config_incomplete")


def test_load_gate_loads_a_verified_pair(tmp_path):
    binding = write_gate_files(tmp_path, inputs=inputs_frame(["E1", "E2"]))
    gate = gp.load_gate(binding)
    assert gate.threshold == fold.REGISTERED_THRESHOLD
    assert gate.features == tuple(FEATURES)
    assert list(gate.inputs.index) == ["E1", "E2"]
    assert gp.receipt_body(gate) == {"rule": "G_PROD", "fold_sha256": binding["gate_fold"]["sha256"],
                                     "inputs_sha256": binding["gate_inputs"]["sha256"], "threshold": fold.REGISTERED_THRESHOLD}


@pytest.mark.parametrize("key, what", [("gate_fold", "fold"), ("gate_inputs", "inputs")])
def test_load_gate_refuses_a_sha_mismatch_before_parsing(tmp_path, key, what):
    binding = write_gate_files(tmp_path)
    with open(binding[key]["path"], "wb") as handle:
        handle.write(b"not a joblib or parquet file")
    with pytest.raises(OpsError) as exc:
        gp.load_gate(binding)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", f"gate_{what}_sha_mismatch")


def test_load_gate_refuses_a_missing_file_and_a_symlink(tmp_path):
    binding = write_gate_files(tmp_path)
    fold_path = binding["gate_fold"]["path"]
    with open(fold_path, "rb") as handle:
        original = handle.read()
    os.remove(fold_path)
    with pytest.raises(OpsError) as exc:
        gp.load_gate(binding)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "gate_fold_missing")
    copy = tmp_path / "copy.joblib"
    copy.write_bytes(original)  # the same bytes, so the hash still matches; the link itself must be refused
    os.symlink(copy, fold_path)
    with pytest.raises(OpsError) as exc:
        gp.load_gate(binding)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "gate_fold_missing")


def test_load_gate_refuses_a_bad_binding():
    good_inputs = {"path": "i", "sha256": "0" * 64}
    with pytest.raises(OpsError) as exc:
        gp.load_gate({"gate_fold": {"path": "x"}, "gate_inputs": good_inputs})  # no sha256
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "gate_binding_invalid")
    with pytest.raises(OpsError) as exc:
        gp.load_gate({"gate_fold": {"path": "x", "sha256": "abc"}, "gate_inputs": good_inputs})  # wrong shape
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "gate_binding_invalid")


@pytest.mark.parametrize("payload", [
    pytest.param({**fold_payload(), "threshold": fold.REGISTERED_THRESHOLD + 0.01}, id="changed_threshold"),
    pytest.param({**fold_payload(), "train_through": 2025}, id="late_fold"),
    pytest.param({**fold_payload(), "features": [f for f in FEATURES if f != "forecast_edge"]}, id="no_forecast_edge"),
])
def test_load_gate_refuses_a_changed_threshold_or_late_fold(tmp_path, payload):
    with pytest.raises(OpsError) as exc:
        gp.load_gate(write_gate_files(tmp_path, payload=payload))
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "gate_fold_invalid")


@pytest.mark.parametrize("inputs", [
    pytest.param(inputs_frame(["E1", "E2"]).drop(columns=["x3"]), id="missing_feature"),
    pytest.param(pd.concat([inputs_frame(["E1"]), inputs_frame(["E1"])], ignore_index=True), id="duplicate_event_id"),
])
def test_load_gate_refuses_inputs_with_wrong_columns(tmp_path, inputs):
    with pytest.raises(OpsError) as exc:
        gp.load_gate(write_gate_files(tmp_path, inputs=inputs))
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "gate_inputs_invalid")


# ---- scoring ----------------------------------------------------------------

def test_scores_replace_only_the_five_model_columns():
    recorder = Recorder()
    inputs = pd.DataFrame({**{f"x{i}": 100.0 + i for i in range(45)}, **{c: 999.0 for c in gp.MODEL_COLS}},
                          index=pd.Index(["E1"], name="event_id"))
    gate = gp.Gate(recorder, tuple(FEATURES), fold.REGISTERED_THRESHOLD, inputs, "a", "b")
    table, labels = history()
    pool = gp.pool.pool_history(table, labels)
    pred, res = pool["pred_abs_move"].to_numpy(), pool["err_move"].to_numpy()
    scores = gp._scores(gate, pd.DataFrame([ev("E1", 2.0)]), pred, res)
    assert scores == {"E1": (0.5, None)}
    row = dict(zip(FEATURES, recorder.seen[0]))
    assert all(row[f"x{i}"] == 100.0 + i for i in range(45))
    assert row["pred_abs_move"] == 7.0 and row["forecast_edge"] == 2.0
    for col in ("pred_abs_move_p10", "pred_abs_move_p90"):
        assert np.isfinite(row[col]) and row[col] != 999.0
    assert row["pred_abs_move_p10"] <= row["pred_abs_move_p90"]
    assert np.isfinite(row["pred_abs_move_sd"]) and row["pred_abs_move_sd"] > 0 and row["pred_abs_move_sd"] != 999.0


# ---- month_decider ----------------------------------------------------------

def test_decisions_cover_taken_not_taken_filter_and_undetermined(tmp_path):
    gate = _ten_gate(tmp_path)
    events = [ev("hi", 3.0), ev("lo", 0.5), ev("nan", 3.0), ev("absent", 3.0), ev("small", 3.0, mcap=1e6)]
    chunk = good("hi", "lo", "nan", "absent", "small")
    table, labels = history()
    out = gp.month_decider(gate, table, labels, START)(pd.DataFrame(events), chunk).set_index("event_id")
    expected = {
        "hi": ("taken", None),
        "lo": ("not_taken", "below_gate_threshold"),
        "nan": ("undetermined", "gate_input_nan"),
        "absent": ("undetermined", "gate_input_missing"),
        "small": ("not_taken", "universe_filter"),  # the gate is not asked
    }
    for eid, (status, reason) in expected.items():
        assert out.loc[eid, "status"] == status
        if reason is None:
            assert pd.isna(out.loc[eid, "reason"])
        else:
            assert out.loc[eid, "reason"] == reason
    assert list(out.columns) == [c for c in sd.DECISION_COLS if c != "event_id"]


def test_thin_pool_is_undetermined_not_no(tmp_path):
    gate = _ten_gate(tmp_path)
    table, labels = history(n=10)
    out = gp.month_decider(gate, table, labels, START)(pd.DataFrame([ev("hi", 3.0)]), good("hi")).set_index("event_id")
    assert out.loc["hi", "status"] == "undetermined" and out.loc["hi", "reason"] == "thin_pool"


def test_pool_excludes_events_on_or_after_the_month_start(tmp_path):
    table, labels = history()
    late = [f"L{i}" for i in range(5)]
    late_rows = pd.DataFrame({"event_date": START + pd.to_timedelta(np.arange(5), unit="D"), "pred_move": 5.0, "pred_crush": -15.0},
                             index=late)
    table = pd.concat([table, late_rows])
    labels = pd.concat([labels, pd.DataFrame({"abs_move": 500.0, "crush_pct_iv30": -15.0}, index=late)])
    gate = gp.load_gate(write_gate_files(tmp_path, inputs=inputs_frame(["hi"])))
    out = gp.month_decider(gate, table, labels, START)(pd.DataFrame([ev("hi", 3.0)]), good("hi")).set_index("event_id")
    assert out.loc["hi", "status"] == "taken"
    before = gp.pool.pool_history(table[table["event_date"] < START], labels)
    assert before["err_move"].abs().max() < 50
    recorder = Recorder()
    probe = gp.Gate(recorder, tuple(FEATURES), fold.REGISTERED_THRESHOLD,
                    inputs_frame(["hi"]).set_index("event_id")[FEATURES].astype(float), "a", "b")
    gp._scores(probe, pd.DataFrame([ev("hi", 3.0)]), before["pred_abs_move"].to_numpy(), before["err_move"].to_numpy())
    assert dict(zip(FEATURES, recorder.seen[0]))["pred_abs_move_sd"] < 5


# ---- straddle contract ------------------------------------------------------

def test_straddle_decide_gate_contract():
    events = pd.DataFrame([ev("hi", 3.0)])
    with pytest.raises(ValueError, match="needs a gate callable"):
        sd.decide(events, EMPTY, "G_PROD", good("hi"))
    with pytest.raises(ValueError, match="unknown rule"):
        sd.decide(events, EMPTY, "R_L", good("hi"), gate=lambda e: ("taken", None))
    assert sd.RULES == ("R_L", "R_L2") and "G_PROD" not in sd.RULES


def test_missing_forecast_is_undetermined_before_the_gate_is_asked():
    out = sd.decide(pd.DataFrame([ev("u", 1.0, implied=float("nan"))]), EMPTY, "G_PROD", good("u"),
                    gate=lambda e: pytest.fail("gate must not be called"))
    assert out.loc[0, "status"] == "undetermined" and out.loc[0, "reason"] == "missing_forecast"
