"""experiments/ — spec hashing, the append-only ledger, scaffolding, promotion.

The ledger is the multiple-testing record, so its append-only invariant is
tested hardest: a rewrite must be impossible through the API and detectable
when attempted out of band. Scaffolding is tested for the pre-registration
stamp and the EXP-101 floor that keeps the 0-50 range reserved.
"""
from __future__ import annotations

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
