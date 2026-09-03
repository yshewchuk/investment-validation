"""experiments/ — spec hashing, the append-only ledger, scaffolding, promotion.

The ledger is the multiple-testing record, so its append-only invariant is
tested hardest: a rewrite must be impossible through the API and detectable
when attempted out of band. Scaffolding is tested for the pre-registration
stamp and the EXP-101 floor that keeps the 0-50 range reserved.
"""
from __future__ import annotations

import pandas as pd
import pytest

from experiments import lib
from experiments import promote as promote_mod
from experiments.new_experiment import scaffold


class TestSpecHash:
    def test_matches_engine_definition(self):
        from engine.evaluate import spec_hash as engine_hash

        spec = {"id": "EXP-101", "primary_spec": {"back_dte": 20}}
        assert lib.spec_hash(spec) == engine_hash(spec)

    def test_slugify(self):
        assert lib.slugify("CAL-P exact-spec backtest!") == "cal_p_exact_spec_backtest"
        assert lib.slugify("???") == "experiment"


class TestExperimentIds:
    def test_parse_rejects_below_floor(self):
        with pytest.raises(ValueError):
            lib.parse_experiment_id("EXP-050")
        with pytest.raises(ValueError):
            lib.parse_experiment_id("EXP-099")
        with pytest.raises(ValueError):
            lib.parse_experiment_id("exp-101")  # wrong case
        assert lib.parse_experiment_id("EXP-101") == 101

    def test_next_number_starts_at_floor(self, tmp_path):
        assert lib.next_experiment_number(tmp_path / "empty") == 101

    def test_next_number_continues(self, tmp_path):
        root = tmp_path / "experiments"
        (root / "EXP-104_foo").mkdir(parents=True)
        (root / "EXP-101_bar").mkdir()
        assert lib.next_experiment_number(root) == 105


class TestLedger:
    ROW = {"id": "EXP-101", "spec_hash": "deadbeef", "date": "2026-08-30",
           "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "",
           "promoted": "False"}

    def test_append_read_roundtrip(self, tmp_path):
        path = tmp_path / "LEDGER.csv"
        lib.ledger_append([self.ROW], path=path)
        frame = lib.ledger_read(path)
        assert len(frame) == 1
        assert frame.iloc[0]["id"] == "EXP-101"

    def test_read_missing_returns_empty_with_columns(self, tmp_path):
        frame = lib.ledger_read(tmp_path / "nope.csv")
        assert frame.empty and list(frame.columns) == lib.LEDGER_COLUMNS

    def test_verify_append_accepts_growth(self):
        assert lib.verify_append(b"a", b"ab")
        assert lib.verify_append(b"", b"ab")
        assert lib.verify_append(b"ab", b"ab")

    def test_verify_append_rejects_rewrite(self):
        assert not lib.verify_append(b"abc", b"abX")
        assert not lib.verify_append(b"abc", b"ab")  # trim
        assert not lib.verify_append(b"abc", b"xbc")

    def test_missing_column_rejected(self, tmp_path):
        bad = {k: v for k, v in self.ROW.items() if k != "stage"}
        with pytest.raises(lib.LedgerError):
            lib.ledger_append([bad], path=tmp_path / "L.csv")

    def test_no_rewrite_api_exists(self):
        for forbidden in ("ledger_rewrite", "ledger_delete", "ledger_replace", "ledger_write"):
            assert not hasattr(lib, forbidden)

    def test_context_counts_specs(self, tmp_path):
        path = tmp_path / "L.csv"
        lib.ledger_append([self.ROW, {**self.ROW, "stage": "ran"}], path=path)
        lib.ledger_append([{**self.ROW, "id": "EXP-102", "spec_hash": "feedface"}], path=path)
        ctx = lib.ledger_context("deadbeef", path=path)
        assert ctx["specs_tried"] == 2
        assert ctx["this_spec_rows"] == 2


class TestScaffold:
    def test_creates_folder_and_stamps_preregistration(self, tmp_path):
        folder = scaffold("Probe title", "Probe hypothesis.", exp_id="EXP-101",
                          root=tmp_path, ledger_path=tmp_path / "LEDGER.csv")
        assert folder.name.startswith("EXP-101")
        spec = lib.load_spec(folder / "spec.yaml")
        assert spec["id"] == "EXP-101"
        assert spec["preregistered_at"], "scaffolder must stamp preregistered_at"
        assert (folder / "run.py").exists()
        assert (folder / "REPORT.md").exists()
        assert (folder / "results").is_dir()
        # PLANNED row landed in the ledger.
        frame = lib.ledger_read(tmp_path / "LEDGER.csv")
        assert (frame["stage"] == "planned").any()

    def test_refuses_duplicate_id(self, tmp_path):
        scaffold("A", "h.", exp_id="EXP-101", root=tmp_path,
                 ledger_path=tmp_path / "L.csv")
        with pytest.raises(FileExistsError):
            scaffold("B", "h.", exp_id="EXP-101", root=tmp_path,
                     ledger_path=tmp_path / "L.csv")

    def test_auto_numbering(self, tmp_path):
        folder = scaffold("Auto", "h.", root=tmp_path, ledger_path=tmp_path / "L.csv")
        assert folder.name.startswith("EXP-101")


class TestRegistryTable:
    def test_render_latest_state_per_id(self, tmp_path):
        from experiments import registry_table

        path = tmp_path / "L.csv"
        lib.ledger_append([{
            "id": "EXP-101", "spec_hash": "abc", "date": "2026-08-30",
            "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "",
            "promoted": "False",
        }], path=path)
        lib.ledger_append([{
            "id": "EXP-101", "spec_hash": "abc", "date": "2026-08-31",
            "stage": "ran", "oos_mean_mid": "0.02", "sharpe_trade": "0.7",
            "promoted": "False",
        }], path=path)
        table = registry_table.render_table(path)
        assert "| EXP-101 | ran |" in table
        assert "0.02" in table
        assert "planned" not in table  # latest row per id wins

    def test_render_empty(self, tmp_path):
        from experiments import registry_table

        assert "No experiments" in registry_table.render_table(tmp_path / "nope.csv")


class TestDecide:
    def _m(self, mean, sharpe, ploss, tail=True, regimes=None):
        return {"mean": mean, "sharpe_trade": sharpe,
                "mc": {"p_loss": ploss},
                "stress": {"regimes": regimes or {"2022": {"n": 40, "mean": 0.05}},
                           "tail_injection": {"available": True} if tail else {}}}

    def test_better_challenger_promotes(self):
        ok, reasons = promote_mod.decide(self._m(0.04, 1.5, 0.08),
                                         self._m(0.03, 1.2, 0.10), prereg_valid=True)
        assert ok, reasons

    def test_worse_mean_refused(self):
        ok, _ = promote_mod.decide(self._m(0.02, 1.5, 0.08),
                                   self._m(0.03, 1.2, 0.10), prereg_valid=True)
        assert not ok

    def test_worse_ploss_refused(self):
        ok, _ = promote_mod.decide(self._m(0.04, 1.5, 0.20),
                                   self._m(0.03, 1.2, 0.10), prereg_valid=True)
        assert not ok

    def test_unregistered_refused(self):
        ok, _ = promote_mod.decide(self._m(0.04, 1.5, 0.08),
                                   self._m(0.03, 1.2, 0.10), prereg_valid=False)
        assert not ok

    def test_short_leg_without_tail_refused(self):
        ok, _ = promote_mod.decide(self._m(0.04, 1.5, 0.08, tail=False),
                                   self._m(0.03, 1.2, 0.10), prereg_valid=True,
                                   short_leg=True)
        assert not ok

    def test_tie_keeps_champion(self):
        ok, _ = promote_mod.decide(self._m(0.03, 1.2, 0.10),
                                   self._m(0.03, 1.2, 0.10), prereg_valid=True)
        assert not ok


class TestDecideFullShape:
    def _results(self, mean, sharpe=1.0, p_loss=0.1, prereg=True, checklist_fails=0,
                 tail=None, brier=None):
        doc = {
            "headline": {"mean": mean, "sharpe_trade": sharpe, "n": 100},
            "mc": {"by_fraction": {"0.05": {"p_loss": p_loss}}},
            "stress": {"regimes": {"2022": {"n": 40, "mean": 0.05}},
                       "tail_injection": tail if tail is not None else {"available": True}},
            "preregistration": {"valid": prereg},
            "checklist_fails": checklist_fails,
        }
        if brier is not None:
            doc["calibration"] = {"brier_skill": brier}
        return doc

    def test_full_shape_better_promotes(self):
        ok, reasons = promote_mod.decide(self._results(0.05, sharpe=1.5),
                                         self._results(0.03, sharpe=1.0))
        assert ok, reasons

    def test_full_shape_worse_refused(self):
        ok, _ = promote_mod.decide(self._results(0.02), self._results(0.03))
        assert not ok

    def test_full_shape_worse_ploss_refused(self):
        ok, _ = promote_mod.decide(self._results(0.05, p_loss=0.3), self._results(0.03, p_loss=0.1))
        assert not ok

    def test_full_shape_checklist_fails_refused(self):
        ok, reasons = promote_mod.decide(self._results(0.05, checklist_fails=2),
                                         self._results(0.03))
        assert not ok
        assert any("(e)" in r for r in reasons)

    def test_brier_skill_rule(self):
        ok, reasons = promote_mod.decide(self._results(0.05, brier=-0.10),
                                         self._results(0.03))
        assert not ok
        assert any("(f)" in r and "FAIL" in r for r in reasons)
        ok, reasons = promote_mod.decide(self._results(0.05, sharpe=1.5, brier=0.01),
                                         self._results(0.03, sharpe=1.0))
        assert ok, reasons

    def test_short_leg_tail_must_be_available(self):
        ok, _ = promote_mod.decide(self._results(0.05, tail={"available": False}),
                                   self._results(0.03), short_leg=True)
        assert not ok

    def test_missing_mc_key_fails_closed(self):
        doc = self._results(0.05)
        doc["mc"] = {}
        ok, reasons = promote_mod.decide(doc, self._results(0.03))
        assert not ok
        assert any("(b)" in r and "FAIL" in r for r in reasons)


class TestChampionFromRegistry:
    def test_registry_baseline_fails_closed_on_missing_fields(self):
        from experiments.promote import champion_from_registry

        champ = champion_from_registry("gate_midfill_str_thru")
        assert champ["mean"] is not None  # eval block carries gated_mean_ret
        assert champ.get("sharpe_trade") is None  # registry cannot supply it
        challenger = {"headline": {"mean": champ["mean"] + 0.05, "sharpe_trade": 1.5},
                      "mc": {"by_fraction": {"0.05": {"p_loss": 0.05}}},
                      "stress": {"regimes": {}, "tail_injection": {"available": True}},
                      "preregistration": {"valid": True},
                      "checklist_fails": 0}
        ok, reasons = promote_mod.decide(challenger, champ)
        assert not ok, "promotion against an incomplete baseline must fail closed"
        assert any("(a)" in r for r in reasons)


class TestPromotionReport:
    def test_report_renders_through_engine_report(self, tmp_path):
        from experiments.promote import render_promotion_report

        spec = {"id": "EXP-101", "title": "probe", "hypothesis": "h.",
                "primary_spec": {"x": 1}}
        challenger = {"headline": {"mean": 0.05, "sharpe_trade": 1.5, "n": 100},
                      "mc": {"by_fraction": {"0.05": {"p_loss": 0.05}}}}
        champion = {"mean": 0.03, "sharpe_trade": 1.2, "mc": {"p_loss": 0.1}}
        reasons = ["PASS (a1) probe"]
        path = render_promotion_report("EXP-101", spec, challenger, champion,
                                       reasons, tmp_path)
        assert path.name == "REPORT.md"
        md = path.read_text()
        assert "Promotion report" in md
        assert "Decision:" in md
        assert "## Provenance" in md
        assert "data snapshot" in md


class TestTailShockDirection:
    """Which tail gets shocked is a property of the structure, not a default.

    CAL-P is short a put and is ruined by a fall; CND-P is short the move in
    BOTH directions and is ruined by either wing being blown through. Ranking a
    condor's shock set on the signed move would inject only the down tail and
    leave half its ruin cases untouched.
    """

    @staticmethod
    def _trades(moves):
        import json

        rows = []
        for i, move in enumerate(moves):
            legs = {
                "spot_entry": 100.0,
                "spot_exit": 100.0 * (1.0 + move),
                "exit": [{"name": "short_hi", "right": "P", "side": "sell",
                          "qty": 1.0, "strike": 105.0, "expiry": "2024-05-17",
                          "dte": 0, "bid": 5.0, "ask": 5.4}],
            }
            rows.append({
                "event_id": f"e{i}", "ticker": "T",
                "event_date": pd.Timestamp("2024-05-02"),
                "entry_date": pd.Timestamp("2024-05-01"),
                "exit_date": pd.Timestamp("2024-05-03"),
                "fill_alpha": 0.5, "entry_cost": 2.0, "exit_value": 2.0,
                "pnl": 0.0, "ret": 0.0, "legs": json.dumps(legs),
            })
        return pd.DataFrame(rows)

    def _shocked(self, out):
        return [i for i in range(len(out)) if out.loc[i, "exit_value"] != 2.0]

    def test_signed_shock_takes_the_worst_fall(self):
        from experiments.common import calp_tail_shock

        # index 1 is the biggest fall; index 2 is a bigger move, but upward.
        out = calp_tail_shock(self._trades([0.01, -0.12, 0.40, -0.01]),
                              worst_frac=0.25)
        assert self._shocked(out) == [1]

    def test_absolute_shock_takes_the_biggest_move_either_way(self):
        from experiments.common import abs_move_tail_shock

        out = abs_move_tail_shock(self._trades([0.01, -0.12, 0.40, -0.01]),
                                  worst_frac=0.25)
        assert self._shocked(out) == [2]

    def test_both_leave_the_quiet_events_alone(self):
        from experiments.common import abs_move_tail_shock, calp_tail_shock

        trades = self._trades([0.01, -0.12, 0.40, -0.01])
        for shock in (calp_tail_shock, abs_move_tail_shock):
            out = shock(trades, worst_frac=0.25)
            assert 0 not in self._shocked(out)
            assert 3 not in self._shocked(out)


class TestTrainedGateThreshold:
    """A gate with no champion picks its threshold on TRAIN rows, per fold.

    The threshold is part of the decision rule, so choosing it on the year being
    gated is the same leak as fitting the model on it: every walk-forward year
    would look like a selection the gate actually made.
    """

    @staticmethod
    def _dataset(n=1200, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        signal = rng.normal(size=n)
        return pd.DataFrame({
            "event_id": [f"e{i}" for i in range(n)],
            "event_date": pd.to_datetime("2018-01-01")
            + pd.to_timedelta(rng.integers(0, 2500, size=n), unit="D"),
            "f1": signal,
            "f2": rng.normal(size=n),
            "ret": 0.3 * signal + rng.normal(scale=0.2, size=n),
        })

    def test_threshold_comes_from_the_training_fold(self):
        from experiments.common import make_trained_gate

        data = self._dataset()
        gate, state = make_trained_gate("test_gate", data, ["f1", "f2"],
                                        top_fraction=0.2)
        train = data.iloc[:900]
        gate.fit(train)
        chosen = [s["threshold_chosen_on_train"] for s in state.stats
                  if "threshold_chosen_on_train" in s]
        assert len(chosen) == 1
        assert state.threshold == pytest.approx(chosen[0])

    def test_it_passes_about_the_registered_top_fraction(self):
        from experiments.common import make_trained_gate

        data = self._dataset()
        gate, state = make_trained_gate("test_gate", data, ["f1", "f2"],
                                        top_fraction=0.2)
        gate.fit(data)
        passed = gate.select(data)
        assert 0.12 < float(passed.mean()) < 0.30

    def test_a_fold_too_small_to_fit_selects_nothing(self):
        from experiments.common import make_trained_gate

        data = self._dataset()
        gate, state = make_trained_gate("test_gate", data, ["f1", "f2"],
                                        top_fraction=0.2)
        gate.fit(data.iloc[:50])
        assert state.threshold is None
        assert not gate.select(data).any()

    def test_a_champion_gate_and_a_trained_one_cannot_be_confused(self):
        from experiments.common import _RegisteredGateState

        with pytest.raises(ValueError, match="exactly one of"):
            _RegisteredGateState("x", ("f1",), None, threshold=0.1, top_fraction=0.2)


class TestScaffoldQuotesTheStrategy:
    """`--strategy "*"` must scaffold, not write a spec.yaml that cannot parse.

    `*` opens a YAML alias, so an unquoted programme-wide strategy made
    new_experiment.py create the folder and then die reading its own output —
    leaving a half-scaffolded experiment with no ledger row.
    """

    def test_a_wildcard_strategy_round_trips(self, tmp_path):
        folder = scaffold("wildcard strategy spec", "hypothesis text",
                          exp_id="EXP-901", strategy="*", root=tmp_path,
                          ledger_path=tmp_path / "LEDGER.csv")
        assert lib.load_spec(folder / "spec.yaml")["strategy"] == "*"

    def test_an_ordinary_strategy_still_round_trips(self, tmp_path):
        folder = scaffold("ordinary strategy spec", "hypothesis text",
                          exp_id="EXP-902", strategy="CND-P", root=tmp_path,
                          ledger_path=tmp_path / "LEDGER.csv")
        assert lib.load_spec(folder / "spec.yaml")["strategy"] == "CND-P"
