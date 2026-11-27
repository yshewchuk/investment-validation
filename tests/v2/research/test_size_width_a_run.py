"""EXP-189 Amendment 2, slice 1: model A priced end to end (the driver under the A fill and its receipts, the refusals, the
selection, AFills' guard and binding, the unlisted exit contract, and run.py's --fill-variant). Synthetic only: the world of
test_size_width_f_run and test_size_width_f_quotes.

Independent oracle: model A's fills are computed here from each strike's quote (mid, bid, ask) by the slice's formula
(mid + c * half for a buy, mid - c * half for a sell, c = min(1, 0.21 + w * min(s / 0.50, 1))); nothing from the
experiments fair_value_*, size_width_* pricing code computes an expected value here."""
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments import fair_value_constants as fvk
from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as fvr
from experiments import size_pnl_receipts as receipts
from experiments import size_pnl_shortvol as sv
from experiments import size_width_a_fills as af
from experiments import size_width_driver as driver
from experiments import size_width_f_counts as fc
from experiments import size_width_f_fills as ff
from experiments import size_width_f_quotes as wf
from experiments import size_width_f_select as sel
from experiments import size_width_f_store as fstore
from experiments import size_width_report as report
from experiments.fair_value_legs import FairValueRefusal
from experiments.size_pnl_shortvol import Refusal
from experiments.size_pnl_straddle import EXIT_UNQUOTED
from experiments.size_width_fills import Skip
from tests.v2.research import test_size_pnl_driver as pnl
from tests.v2.research import test_size_width_f_quotes as qt
from tests.v2.research.test_size_width_f_run import (  # noqa: F401 (the fixtures are used by name)
    DATES, IDS, _f_run, _fill, _outer_chunk, _quote, _receipt, _setup, _traded, bounded, m0, stubbed)
from tests.v2.research.test_size_width_run import FAKE, _argv, _gates, _install, _run_main

DOC = fvk.constants_document(fvk.primary_constants(0.21, 0.0, 0.15, 0.10))  # base, spread, thin_size, thin_oi
EXPERIMENTS = Path(__file__).resolve().parents[3] / "experiments"
A_SOURCES = ("fair_value_apply.py", "fair_value_constants.py", "size_width_a_fills.py", "size_width_f_fills.py",
             "size_width_f_quotes.py", "size_pnl_shortvol.py", "size_width_fills.py", "size_width_pricing.py")
WEIGHT = {"primary": 0.0, "spread050": 0.50}  # the spread weight each variant sets (the fixtures' sizes and open interest are not thin)
MODEL = af.ModelA(DOC, DOC["constants_id"], "primary")


def _a_fill(pins):
    """The A fill dict: the F fill's shape, id A, and the pinned constants_id."""
    return {**_fill(pins), "id": "A", "constants_id": DOC["constants_id"]}


def _constants(tmp_path, monkeypatch):
    """Writes DOC to tmp_path/model_a_constants.json and points the module's CONSTANTS_PATH at it."""
    path = tmp_path / "model_a_constants.json"
    path.write_text(json.dumps(DOC), encoding="utf-8")
    monkeypatch.setattr(af, "CONSTANTS_PATH", path)
    return path


def _a_run(src, out, stage, fill, frames, pins, variant):
    return _f_run(src, out, stage, fill, sel.FInputs(pins, qt._reader(qt._base(frames)), "snap-1", variant))


def _a_fill_price(k, buy, variant):
    """Model A's fill of a put at strike k, from its quote alone (no thin flags: the fixture sizes are 50 and open interest 500)."""
    mid, bid, ask = _quote(k)
    half, s = (ask - bid) / 2, (ask - bid) / mid
    c = min(1.0, 0.21 + WEIGHT[variant] * min(s / 0.50, 1.0))
    return mid + c * half if buy else mid - c * half


def _a_oracle(traded, chunk, variant):
    """(entry debit, exit proceeds) of one structure's traded legs: a buy opens at its buy fill and a sell at its sell fill;
    the exit closes each leg at the other fill."""
    debit = proceeds = 0.0
    for leg in traded:
        k, qty, buy = float(chunk["strike"][leg["idx"]]), float(leg["qty"]), leg["side"] == "buy"
        debit += qty * (_a_fill_price(k, True, variant) if buy else -_a_fill_price(k, False, variant))
        proceeds += qty * (_a_fill_price(k, False, variant) if buy else -_a_fill_price(k, True, variant))
    return debit, proceeds


@pytest.mark.parametrize("variant", ["primary", "spread050"])
def test_the_a_run_prices_every_taken_arm_1_0_row_at_model_a_and_binds_its_receipts(tmp_path, monkeypatch, m0, stubbed, variant):
    frames, root, pins, src = _setup(tmp_path, monkeypatch)
    path = _constants(tmp_path, monkeypatch)
    out, fill = tmp_path / "out", _a_fill(pins)
    loaded = {}
    for stage in ("inner", "outer"):
        if stage == "outer" and variant == "spread050":
            path.write_text(json.dumps(DOC, indent=2), encoding="utf-8")  # the same constants, re-formatted: only the bytes change
        loaded[stage] = path.read_bytes()
        _a_run(src, out, stage, fill, frames, pins, variant)
    frozen = pd.read_parquet(out / "months" / "2025-02.parquet")
    taken = frozen[(frozen["status"] == "taken") & (frozen["arm_or_rule"] == "1.0")]
    assert len(taken) >= 1
    chunk, pos = _outer_chunk()
    for rec in taken.to_dict("records"):
        debit, proceeds = _a_oracle(_traded(chunk, pos, str(rec["event_id"]), rec["structure"]), chunk, variant)
        assert rec["entry_cost"] == pytest.approx(debit, rel=1e-9, abs=1e-12)
        assert rec["exit_value"] == pytest.approx(proceeds, rel=1e-9, abs=1e-12)
    receipt = {}
    for stage in ("inner", "outer"):
        receipt[stage] = receipts.load_receipt(out / "receipts" / f"inputs_{stage}.json")
        bound = receipt[stage]["fill_model"]
        assert bound["fill_model_id"] == "A" and bound["constants_id"] == DOC["constants_id"] and bound["variant"] == variant
        assert "constants_file_sha256" not in bound
        assert receipt[stage]["constants_file_sha256"] == hashlib.sha256(loaded[stage]).hexdigest()
        assert bound["model_a_source_sha256"] == {n: hashlib.sha256((EXPERIMENTS / n).read_bytes()).hexdigest() for n in A_SOURCES}
        assert bound["fvf_id"] == fill["fvf_id"] and bound["quote_snapshot_id"] == "snap-1"
    assert receipt["inner"]["inputs_hash"] == receipt["outer"]["inputs_hash"]
    if variant == "spread050":
        assert receipt["inner"]["constants_file_sha256"] != receipt["outer"]["constants_file_sha256"]
    assert (out / "receipts" / "inner_summary.json").exists() and (out / "receipts" / "outer_summary.json").exists()
    checked = 0
    for name in ("taken", "legs"):
        for part in sorted((out / "report_inputs" / "outer").glob(f"{name}_*.parquet")):
            rows = pd.read_parquet(part)
            assert (rows["constants_id"] == DOC["constants_id"]).all() and (rows["variant"] == variant).all()
            if name != "taken": continue
            for rec in rows.to_dict("records"):
                if rec["arm"] != "1.0" or rec["rung"] == EXIT_UNQUOTED or rec["k"] != 1.0: continue
                want = sum(leg["qty"] * (1.0 if leg["side"] == "buy" else -1.0) * _quote(float(chunk["strike"][leg["idx"]]))[0]
                           for leg in _traded(chunk, pos, str(rec["event_id"]), rec["structure"]))
                assert rec["exit_value_at_eff_mid"] == pytest.approx(want, rel=1e-9, abs=1e-12)
                checked += 1
    assert checked >= 1


def test_a_store_built_without_a_model_binds_nothing():
    assert fstore.Store(None, "inner", None).bind == {}


def test_the_two_variants_oracle_fills_differ_for_a_leg():
    chunk, pos = _outer_chunk()
    leg = _traded(chunk, pos, "E3", sv.NAMES[0])[0]
    k, buy = float(chunk["strike"][leg["idx"]]), leg["side"] == "buy"
    assert abs(_a_fill_price(k, buy, "primary") - _a_fill_price(k, buy, "spread050")) > 1e-9


def test_an_outer_under_another_variant_than_the_inners_refuses_and_writes_no_outer_receipts(tmp_path, monkeypatch, m0, stubbed):
    frames, root, pins, src = _setup(tmp_path, monkeypatch)
    _constants(tmp_path, monkeypatch)
    out = tmp_path / "out"
    _a_run(src, out, "inner", _a_fill(pins), frames, pins, "primary")
    with pytest.raises(Refusal) as caught:
        _a_run(src, out, "outer", _a_fill(pins), frames, pins, "spread050")
    assert caught.value.details["reason"] == "tuning_receipt_changed" and caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not (out / "receipts" / "inputs_outer.json").exists()
    assert not (out / "receipts" / "outer_summary.json").exists()


def test_the_report_refuses_an_a_run_and_writes_no_report(tmp_path, monkeypatch, m0, stubbed):
    frames, root, pins, src = _setup(tmp_path, monkeypatch)
    _constants(tmp_path, monkeypatch)
    out = tmp_path / "out"
    for stage in ("inner", "outer"):
        _a_run(src, out, stage, _a_fill(pins), frames, pins, "primary")
    with pytest.raises(Refusal) as caught:
        report.assemble(out, expected=src.expected)
    assert caught.value.details["reason"] == "tuning_receipt_changed" and caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not (out / "REPORT.md").exists()


def test_a_source_over_the_cap_refuses_input_too_large_before_any_quote_is_read(tmp_path, monkeypatch):
    _constants(tmp_path, monkeypatch)
    monkeypatch.setattr(af, "MAX_SOURCE_BYTES", 10)
    with pytest.raises(fvi.FairValueRefusal) as caught:
        af.load(DOC["constants_id"], "primary")
    assert caught.value.details["reason"] == "input_too_large" and caught.value.code == "RESOURCE_LIMIT_EXCEEDED"


def _unread(expiries):
    raise AssertionError("no quote may be read before the constants and the fill are admitted")


@pytest.mark.parametrize("case, reason", [
    ("absent", "constants_unreadable"), ("not_json", "constants_unreadable"), ("too_large", "input_too_large"),
    ("wrong_id", "constants_id_mismatch"), ("variant_x", "variant_unknown"),
    ("f_with_spread050", "fill_variant_unexpected"), ("no_constants_id", "fill_model_invalid")])
def test_open_run_refuses_model_a_inputs_before_any_quote_is_read(tmp_path, monkeypatch, case, reason):
    pins = qt._pins(qt._write(tmp_path, qt._base(qt._frames())))
    path = tmp_path / "model_a_constants.json"
    if case == "not_json":
        path.write_bytes(b"not json")
    elif case == "too_large":
        path.write_bytes(b"0" * (af.MAX_CONSTANTS_BYTES + 1))
    elif case != "absent":
        path.write_text(json.dumps(DOC), encoding="utf-8")
    monkeypatch.setattr(af, "CONSTANTS_PATH", path)
    fill = _a_fill(pins)
    if case == "wrong_id":
        fill["constants_id"] = "0" * 64
    if case == "no_constants_id":
        del fill["constants_id"]
    if case == "f_with_spread050":
        fill = _fill(pins)
    variant = {"variant_x": "x", "f_with_spread050": "spread050"}.get(case, "primary")
    with pytest.raises((fvi.FairValueRefusal, FairValueRefusal)) as caught:
        sel.open_run(fill, sel.FInputs(pins, _unread, "snap-1", variant), qt.DEFAULT_PLAN, "snap-1", root=tmp_path)
    assert caught.value.details["reason"] == reason
    assert caught.value.code == ("RESOURCE_LIMIT_EXCEEDED" if reason == "input_too_large" else "EXPERIMENT_IDENTITY_CONFLICT")


def _spec(*amendments):
    return {"id": "EXP-189", "amendments": list(amendments)}


def _refused(reason, call):
    with pytest.raises(fvi.FairValueRefusal) as caught:
        call()
    assert caught.value.details["reason"] == reason and caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


F_CARRIER = {"id": "F", "fvf_id": fvr.fvf_id()}
A_CARRIER = {"id": "A", "fvf_id": fvr.fvf_id(), "constants_id": "c" * 64}


def test_an_f_then_a_pair_of_carriers_selects_a():
    assert sel.fill_model(_spec({"id": "A1", "fill_model": F_CARRIER}, {"id": "A2", "fill_model": A_CARRIER})) == A_CARRIER


def test_an_a_then_f_pair_of_carriers_is_ambiguous():
    _refused("fill_model_ambiguous", lambda: sel.fill_model(_spec({"fill_model": A_CARRIER}, {"fill_model": F_CARRIER})))


def test_an_f_a_a_triple_of_carriers_is_ambiguous():
    _refused("fill_model_ambiguous", lambda: sel.fill_model(_spec({"fill_model": F_CARRIER}, {"fill_model": A_CARRIER},
                                                                  {"fill_model": A_CARRIER})))


def test_a_lone_f_carrier_still_selects_f():
    assert sel.fill_model(_spec({"fill_model": F_CARRIER})) == F_CARRIER


def _model_quotes(tmp_path, frames, model):
    """An F quote adapter over ``frames`` that prices and binds ``model`` (the pinned tables under tmp_path/root)."""
    base = qt._base(frames)
    root = qt._write(tmp_path, base)
    fq = wf.FQuotes(qt.DEFAULT_PLAN, qt._reader(base), tables_root=root, pins=qt._pins(root), snapshot_id="snap-1",
                    expected_snapshot_id="snap-1", vol_scale=0.01, rate_scale=0.01, model=model)
    return fq, qt._chunk(frames, qt.DEFAULT_PLAN)


def _structure_table(fq, chunk, exit_):
    """The positive-quantity legs of the default structure (event 0) as the adapter's leg table, with its binding columns."""
    legs = [leg for leg in qt._resolved(chunk, 0) if leg["qty"] > 0]
    return qt._legs_frame(fq, chunk, legs, at_exit=exit_)


@pytest.mark.parametrize("exit_", [False, True], ids=["entry", "exit"])
@pytest.mark.parametrize("column, value", [("constants_id", "0" * 64), ("variant", "spread050")], ids=["id", "variant"])
def test_a_leg_table_with_another_pinned_pair_refuses_legs_constants_mismatch(tmp_path, exit_, column, value):
    fq, chunk = _model_quotes(tmp_path, qt._frames(), MODEL)
    table = _structure_table(fq, chunk, exit_).assign(**{column: value})
    with pytest.raises(FairValueRefusal) as caught:
        af.AFills(fq, MODEL)._structure(table, exit_)
    assert caught.value.details["reason"] == "legs_constants_mismatch" and caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("exit_", [False, True], ids=["entry", "exit"])
def test_a_leg_table_without_the_binding_columns_refuses_legs_unbound(tmp_path, exit_):
    fq, chunk = _model_quotes(tmp_path, qt._frames(), MODEL)
    table = _structure_table(fq, chunk, exit_).drop(columns=["variant"])
    with pytest.raises(FairValueRefusal) as caught:
        af.AFills(fq, MODEL)._structure(table, exit_)
    assert caught.value.details["reason"] == "legs_unbound" and caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("exit_", [False, True], ids=["entry", "exit"])
@pytest.mark.parametrize("column", ["constants_id", "variant"])
def test_a_leg_table_with_a_null_binding_value_refuses_legs_unbound(tmp_path, exit_, column):
    fq, chunk = _model_quotes(tmp_path, qt._frames(), MODEL)
    table = _structure_table(fq, chunk, exit_)
    table[column] = table[column].astype(object)
    table.loc[table.index[0], column] = None  # the columns exist; one row's binding is missing
    with pytest.raises(FairValueRefusal) as caught:
        af.AFills(fq, MODEL)._structure(table, exit_)
    assert caught.value.details["reason"] == "legs_unbound" and caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_a_wide_quote_is_refused_as_f_fill_over_mid_and_is_a_skip_reason_only_of_model_a(tmp_path, monkeypatch):
    # Hand arithmetic (model A primary, c = 0.21, no thin flags; mid(k) = 1 + 0.01 (k - 100)^2; half = 0.9 on every quote,
    # so the lowest bid, at the 100 strike's mid of 1.0, is 0.1 >= 0):
    #   legs: buy PUT 110 (mid 2.0, quote 1.1 / 2.9), sell PUT 100 (mid 1.0, quote 0.1 / 1.9), both quantity 1.
    #   debit_mid = 2.0 - 1.0 = 1.0 > 0.
    #   debit_fair: the 110 put is worth ~9 or more (intrinsic 10 less discounting), above its ask 2.9, so V = 2.9; the
    #   ATM 100 put is ~2.75 at 25% vol (3.26 at zero rate, less the put's rate discount), above its ask 1.9, so V = 1.9.
    #   debit_fair = 2.9 - 1.9 = 1.0 > 0, so not_a_debit does not fire.
    #   fills: buy 110 at 2.0 + 0.21 * 0.9 = 2.189; sell 100 at 1.0 - 0.21 * 0.9 = 0.811; debit_F = 2.189 - 0.811 = 1.378.
    #   excess = debit_F - debit_mid = 0.378 = 0.42 * 0.9, and 0.378 > 0.25 * debit_mid = 0.25, so fill_over_mid fires.
    half = 0.9
    frames = qt._frames()
    wide = {(right, k): (qt._mid(k) - half, qt._mid(k) + half) for k in qt.STRIKES for right in ("P", "C")}
    frames[("AAA", pd.Timestamp(qt.ENTRY))] = qt._listing("AAA", qt.ENTRY, quotes=wide)
    monkeypatch.setattr(sv, "entry", lambda priced: (0.0, 0.0))  # the relative spread is not under test
    fq, chunk = _model_quotes(tmp_path, frames, MODEL)
    trade = SimpleNamespace(i=0, chunk=chunk, legs=[qt._leg(chunk, 0, "P", 110.0, side="buy"), qt._leg(chunk, 0, "P", 100.0)])
    fills = af.AFills(fq, MODEL)
    res = fills.entry_result(trade)
    assert res.debit_mid == pytest.approx(1.0)
    assert res.debit_fair == pytest.approx(1.0)
    assert res.debit_F - res.debit_mid == pytest.approx(0.42 * half)
    assert fills.entry(trade, None) == Skip("f_fill_over_mid")
    assert "f_fill_over_mid" in ff.SKIP_REASONS and "f_fill_over_mid" not in ff.ENTRY_REASONS


def test_an_f_fill_over_mid_is_an_f_skip_in_the_counts():
    cands = pd.DataFrame({"event_id": ["e1", "e2", "e3"], "reason": ["f_fill_over_mid", "f_fill_over_fair", "no_trade"]})
    assert fc._skipped(cands) == {"e1", "e2"}


def test_an_unlisted_exit_contract_carries_the_pins_and_is_unpriceable_under_model_a(tmp_path):
    frames = qt._frames()
    frames[("AAA", pd.Timestamp(qt.EXIT))] = qt._listing("AAA", qt.EXIT, skip=[("P", 100.0)])
    fq, chunk = _model_quotes(tmp_path, frames, MODEL)
    legs = [qt._leg(chunk, 0, "P", 100.0), qt._leg(chunk, 0, "P", 99.0, side="buy")]
    table = qt._legs_frame(fq, chunk, legs, at_exit=True)
    assert table.loc[0, "status"] == "unpriceable"
    assert (table["constants_id"] == DOC["constants_id"]).all() and (table["variant"] == "primary").all()
    assert af.AFills(fq, MODEL)._structure(table, True).outcome == "unpriceable"


def _pins_file(tmp_path):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"manifest_sha256": "0" * 64, "files": {"quotes/year=2024/month=02.parquet": "0" * 64}}),
                    encoding="utf-8")
    return str(path)


def _quotes_factory(monkeypatch, result, name="exp189_a_quotes"):
    module = types.ModuleType(name)
    module.make = lambda args: result
    monkeypatch.setitem(sys.modules, name, module)
    return f"{name}:make"


def test_run_py_refuses_a_fill_variant_without_f_pins(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    _gates(monkeypatch)
    assert run_py.main(_argv(tmp_path, "--fill-variant", "spread050", sources=FAKE)) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["refused"] == "INVALID_EXPERIMENT_SPEC" and refused["reason"] == "f_inputs_incomplete"


def test_run_py_hands_the_fill_variant_to_the_f_inputs(tmp_path, monkeypatch, capsys):
    run_py = _run_main()
    _gates(monkeypatch)
    _install(monkeypatch, lambda args: pnl.sources())
    seen = {}

    def run(src, stage, out_dir, **kwargs):
        seen.update(kwargs)
        return {"stage": stage, "arms": [], "months": 0}

    monkeypatch.setattr(driver, "run", run)
    quotes = _quotes_factory(monkeypatch, (qt._reader(qt._base(qt._frames())), "snap-1"))
    assert run_py.main(_argv(tmp_path, "--f-pins", _pins_file(tmp_path), "--f-quotes", quotes, "--fill-variant", "spread050",
                             sources=FAKE)) == 0
    assert seen["f_inputs"].variant == "spread050" and seen["f_inputs"].snapshot_id == "snap-1"
