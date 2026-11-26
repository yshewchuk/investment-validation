"""EXP-189 Amendment 1, slice W7d: F selection from the spec and the receipt binding.

Synthetic only: fixture spec dicts and tmp_path spec files, never the real spec.yaml selection, never data/."""
import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.fills import FillModel
from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as fvr
from experiments import size_pnl_receipts as receipts
from experiments import size_pnl_shortvol as sv
from experiments import size_width_deviation as deviation
from experiments import size_width_driver as driver
from experiments import size_width_f_fills, size_width_f_quotes
from experiments import size_width_f_select as sel
from experiments import size_width_pricing as seam
from experiments import size_width_prereg
from experiments.size_pnl_shortvol import Refusal
from tests.v2.research import test_size_width_f_quotes as wq
from tests.v2.research import test_size_width_prereg as ptest
from tests.v2.research.test_size_pnl_driver import OK, PLAN_FRAME, chunk, sources, use_chunks
from tests.v2.research.test_size_width_driver import bounded, m0, pricing  # noqa: F401  (autouse and stub fixtures)

UNITS = {"borrow30": "percent", "borrow2yr": "percent", "divAmt": "dollars_per_share"}
RULES = {"rate_out_of_range": "missing", "rate_empty_row": "rate_unknown"}
PIN_FILES = {"quotes/year=2024/month=01.parquet": "Q1", "rates/year=2024/month=01.parquet": "R1"}
MANIFEST = json.dumps({"units": UNITS}).encode("utf-8")  # the manifest.json bytes that the spec's reingest hash pins
MANIFEST_SHA = hashlib.sha256(MANIFEST).hexdigest()


def _write_manifest(root, blob=MANIFEST):
    """Write ``blob`` as ``<root>/tables/manifest.json`` and return the sha256 hex of its bytes."""
    (Path(root) / "tables").mkdir(parents=True, exist_ok=True)
    (Path(root) / "tables" / "manifest.json").write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def _fill(**changes):
    fill = {"id": "F", "fvf_id": fvr.fvf_id(), "carry_dividend_mode": sel.MODE, "rate_rules": dict(RULES),
            "vol_scale": 1.0, "rate_scale": 0.01,
            "reingest": {"path": "tables", "manifest_sha256": MANIFEST_SHA, "manifest_units": UNITS}}
    fill.update(changes)
    return fill


def _pins(**changes):
    pins = {"manifest_sha256": MANIFEST_SHA, "files": dict(PIN_FILES)}
    pins.update(changes)
    return pins


def _inputs(pins=None):
    return sel.FInputs(_pins() if pins is None else pins, lambda expiries: None, "snap-1")


def _refused(reason, call):
    with pytest.raises(fvi.FairValueRefusal) as caught:
        call()
    assert caught.value.details["reason"] == reason
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def _spec(*amendments):
    return {"id": "EXP-189", "amendments": list(amendments)}


@pytest.fixture
def f_pricing(monkeypatch, tmp_path):
    """F's load returns a copy of the driver's (already stubbed) PRICING, so the fixture stubs stay in force for F.
    The driver's open_run takes its default root from tmp_path, where the spec's manifest.json is written."""
    made = []
    _write_manifest(tmp_path)
    monkeypatch.setitem(sel.open_run.__kwdefaults__, "root", tmp_path)

    def load(fills=None):
        copy = types.SimpleNamespace(**{k: v for k, v in vars(driver.PRICING).items() if not k.startswith("__")})
        made.append((fills, copy))
        return copy

    monkeypatch.setattr(seam, "load", load)
    return made


def _f_receipt():
    return {"spec_hash": "SPEC", "spec_hash_checked": True, "fill_model": _fill()}


# ---- 1. fill_model -------------------------------------------------------------------

def test_fill_model_is_none_without_amendments_or_without_the_key():
    assert sel.fill_model({}) is None
    assert sel.fill_model({"amendments": []}) is None
    assert sel.fill_model(_spec({"id": "A1", "note": "x"}, "plain")) is None


def test_fill_model_returns_the_one_f_entry_among_others():
    f = {"id": "F", "x": 1}
    assert sel.fill_model(_spec({"id": "A0"}, {"id": "A1", "fill_model": f})) == f


def test_fill_model_with_another_id_refuses_fill_model_id():
    _refused("fill_model_id", lambda: sel.fill_model(_spec({"id": "A1", "fill_model": {"id": "G"}})))


def test_non_dict_fill_model_refuses_fill_model_invalid():
    _refused("fill_model_invalid", lambda: sel.fill_model(_spec({"id": "A1", "fill_model": "F"})))


def test_two_entries_carrying_fill_model_are_ambiguous_whatever_their_ids():
    _refused("fill_model_ambiguous", lambda: sel.fill_model(_spec({"fill_model": {"id": "F"}}, {"fill_model": {"id": "F"}})))
    _refused("fill_model_ambiguous", lambda: sel.fill_model(_spec({"fill_model": {"id": "G"}}, {"fill_model": None})))


# ---- 2. require_preregistration carries the selection into the receipt ------------------

def _prereg_case(tmp_path, amendments):
    spec = ptest._write_real_spec(tmp_path, amendments=amendments)
    digest = ptest._digest(spec)
    ledger = ptest._write_ledger(tmp_path / "LEDGER.csv", [ptest._planned_row(digest) for _ in range(len(amendments) + 1)])
    return spec, ledger


def _receipt(tmp_path, spec, ledger):
    return size_width_prereg.require_preregistration(
        spec_path=spec, ledger_path=ledger, run_log_path=ptest._run_log_path(tmp_path),
        experiments_dir=ptest._own_experiments(tmp_path))


def test_an_f_amendment_with_two_rows_puts_its_fill_model_in_the_receipt(tmp_path):
    fill = _fill()
    spec, ledger = _prereg_case(tmp_path, [{"id": "A1", "fill_model": fill}])
    receipt = _receipt(tmp_path, spec, ledger)
    assert receipt["fill_model"] == fill and receipt["spec_hash_checked"] is True


def test_no_fill_model_amendment_leaves_the_receipt_without_the_key(tmp_path):
    spec, ledger = _prereg_case(tmp_path, [])
    assert "fill_model" not in _receipt(tmp_path, spec, ledger)


def test_an_amendment_with_id_g_refuses_fill_model_id_through_the_guard(tmp_path):
    spec, ledger = _prereg_case(tmp_path, [{"id": "A1", "fill_model": {"id": "G"}}])
    assert ptest.refusal(lambda: _receipt(tmp_path, spec, ledger)) == "fill_model_id"


# ---- 3. open_run ----------------------------------------------------------------------

def test_no_selection_and_no_inputs_is_current_marks():
    assert sel.open_run(None, None, PLAN_FRAME, "snap-1") is None


def test_f_inputs_without_a_selection_refuse_f_inputs_unexpected():
    _refused("f_inputs_unexpected", lambda: sel.open_run(None, _inputs(), PLAN_FRAME, "snap-1"))


def test_a_selection_without_f_inputs_refuses_f_inputs_missing():
    _refused("f_inputs_missing", lambda: sel.open_run(_fill(), None, PLAN_FRAME, "snap-1"))


def test_another_carry_dividend_mode_refuses(tmp_path):
    _refused("carry_dividend_mode", lambda: sel.open_run(_fill(carry_dividend_mode="old"), _inputs(), PLAN_FRAME, "snap-1", root=tmp_path))


def test_a_different_fvf_id_refuses_fvf_id_mismatch(tmp_path):
    _refused("fvf_id_mismatch", lambda: sel.open_run(_fill(fvf_id="0" * 64), _inputs(), PLAN_FRAME, "snap-1", root=tmp_path))


def test_empty_pin_files_refuse_pins_invalid(tmp_path):
    _refused("pins_invalid", lambda: sel.open_run(_fill(), _inputs(_pins(files={})), PLAN_FRAME, "snap-1", root=tmp_path))


def test_a_none_quote_reader_refuses_f_quotes_invalid(tmp_path):
    _refused("f_quotes_invalid", lambda: sel.open_run(_fill(), sel.FInputs(_pins(), None, "snap-1"), PLAN_FRAME, "snap-1", root=tmp_path))


def test_a_pinned_manifest_sha_other_than_the_spec_refuses_manifest_binding(tmp_path):
    _write_manifest(tmp_path)  # the file matches the spec; only the pins differ
    _refused("manifest_binding", lambda: sel.open_run(_fill(), _inputs(_pins(manifest_sha256="OTHER")), PLAN_FRAME, "snap-1", root=tmp_path))


def test_an_absent_manifest_json_refuses_manifest_binding(tmp_path):
    _refused("manifest_binding", lambda: sel.open_run(_fill(), _inputs(), PLAN_FRAME, "snap-1", root=tmp_path))


def test_manifest_bytes_that_hash_differently_from_the_spec_refuse_manifest_binding(tmp_path):
    _write_manifest(tmp_path, blob=b'{"units": {}}')  # pins and spec agree on MANIFEST_SHA, the file does not
    _refused("manifest_binding", lambda: sel.open_run(_fill(), _inputs(), PLAN_FRAME, "snap-1", root=tmp_path))


def test_pins_matching_the_manifest_file_still_refuse_a_spec_hash_that_differs(tmp_path):
    sha = _write_manifest(tmp_path)  # the file and the pins agree on MANIFEST_SHA; the spec pins another hash
    fill = _fill(reingest={"path": "tables", "manifest_sha256": "0" * 64, "manifest_units": UNITS})
    _refused("manifest_binding", lambda: sel.open_run(fill, _inputs(_pins(manifest_sha256=sha)), PLAN_FRAME, "snap-1", root=tmp_path))


def test_manifest_units_differing_from_the_spec_refuse_manifest_binding(tmp_path):
    _write_manifest(tmp_path)  # the file's bytes hash to the spec's and the pins' sha; only the spec's units differ
    units = {**UNITS, "borrow30": "dollars_per_share"}
    fill = _fill(reingest={"path": "tables", "manifest_sha256": MANIFEST_SHA, "manifest_units": units})
    _refused("manifest_binding", lambda: sel.open_run(fill, _inputs(), PLAN_FRAME, "snap-1", root=tmp_path))


def test_an_extra_manifest_unit_key_the_spec_lacks_refuses_manifest_binding(tmp_path):
    sha = _write_manifest(tmp_path, blob=json.dumps({"units": {**UNITS, "extra": "percent"}}).encode("utf-8"))
    fill = _fill(reingest={"path": "tables", "manifest_sha256": sha, "manifest_units": UNITS})
    _refused("manifest_binding", lambda: sel.open_run(fill, _inputs(_pins(manifest_sha256=sha)), PLAN_FRAME, "snap-1", root=tmp_path))


def test_a_reader_snapshot_other_than_expected_refuses_snapshot_binding(tmp_path):
    _write_manifest(tmp_path)
    _refused("snapshot_binding", lambda: sel.open_run(_fill(), _inputs(), PLAN_FRAME, "snap-2", root=tmp_path))


def test_a_missing_reingest_refuses_fill_model_invalid(tmp_path):
    fill = _fill()
    del fill["reingest"]
    _refused("fill_model_invalid", lambda: sel.open_run(fill, _inputs(), PLAN_FRAME, "snap-1", root=tmp_path))


def test_an_extra_rate_rule_refuses_fill_model_invalid(tmp_path):
    fill = _fill(rate_rules={**RULES, "extra": "x"})
    _refused("fill_model_invalid", lambda: sel.open_run(fill, _inputs(), PLAN_FRAME, "snap-1", root=tmp_path))


def test_an_absolute_table_path_refuses_fill_model_invalid(tmp_path):
    fill = _fill(reingest={"path": "/abs", "manifest_sha256": MANIFEST_SHA, "manifest_units": UNITS})
    _refused("fill_model_invalid", lambda: sel.open_run(fill, _inputs(), PLAN_FRAME, "snap-1", root=tmp_path))


def test_a_valid_selection_binds_every_id_unit_and_pin_and_builds_the_f_seam(tmp_path):
    _write_manifest(tmp_path)
    fill = _fill(rate_rules={"rate_out_of_range": "row_unknown", "rate_empty_row": "earlier_row"})
    run = sel.open_run(fill, _inputs(), PLAN_FRAME, "snap-1", root=tmp_path)
    assert run.binding == {
        "fill_model_id": "F", "fvf_id": fill["fvf_id"], "reingest_manifest_sha256": MANIFEST_SHA,
        "every_table_file_sha256": {"quotes/year=2024/month=01.parquet": "Q1", "rates/year=2024/month=01.parquet": "R1"},
        "quote_snapshot_id": "snap-1",
        "units": {"vol_scale": 1.0, "rate_scale": 0.01, "manifest_units": UNITS},
        "rate_out_of_range": "row_unknown", "rate_empty_row": "earlier_row", "carry_dividend_mode": sel.MODE}
    assert isinstance(run.pricing.fills, size_width_f_fills.FFills)
    quotes = run.pricing.fills._quotes
    assert isinstance(quotes, size_width_f_quotes.FQuotes)
    assert quotes._scales == (1.0, 0.01)
    assert quotes._rules == {"rate_out_of_range": "row_unknown", "rate_empty_row": "earlier_row"}
    assert quotes._root == tmp_path / "tables"
    assert run.pricing is not seam.PRICING


# ---- 4. the driver: F binding, F pricing and replay -----------------------------------

def test_the_sweep_is_built_on_the_pricing_f_loaded(tmp_path, m0, pricing, f_pricing, monkeypatch):
    seen, real = [], deviation.DeviationSweep

    class Recorder(real):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("pricing"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(deviation, "DeviationSweep", Recorder)
    use_chunks(monkeypatch, [("2023Q3", chunk("E1")), ("2024Q3", chunk("E2"))])
    driver.run(sources(), "inner", tmp_path / "out", event_ids=["E1", "E2"], prereg=_f_receipt, f_inputs=_inputs())
    assert len(f_pricing) == 1 and isinstance(f_pricing[0][0], size_width_f_fills.FFills)
    loaded = f_pricing[0][1]
    assert loaded is not driver.PRICING and seen and all(p is loaded for p in seen)


def test_f_inner_receipts_bind_the_fill_and_replay_or_refuse_on_a_changed_table_hash(tmp_path, m0, pricing, f_pricing, monkeypatch):
    use_chunks(monkeypatch, [("2023Q3", chunk("E1")), ("2024Q3", chunk("E2"))])
    out = tmp_path / "out"
    binding = sel.open_run(_fill(), _inputs(), PLAN_FRAME, "snap-1").binding
    driver.run(sources(), "inner", out, event_ids=["E1", "E2"], prereg=_f_receipt, f_inputs=_inputs())
    assert receipts.load_receipt(out / "receipts" / "inputs_inner.json")["fill_model"] == binding
    driver.run(sources(), "inner", out, event_ids=["E1", "E2"], prereg=_f_receipt, f_inputs=_inputs())  # identical inputs replay
    changed = _pins(files={**PIN_FILES, "quotes/year=2024/month=01.parquet": "CHANGED"})
    with pytest.raises(Refusal) as caught:
        driver.run(sources(), "inner", out, event_ids=["E1", "E2"], prereg=_f_receipt, f_inputs=_inputs(changed))
    assert caught.value.details["reason"] == "inputs_receipt_changed"


def test_an_f_spec_without_f_inputs_refuses_before_any_receipt(tmp_path, m0, pricing, f_pricing, monkeypatch):
    use_chunks(monkeypatch, [("2023Q3", chunk("E1")), ("2024Q3", chunk("E2"))])
    out = tmp_path / "out"
    with pytest.raises(fvi.FairValueRefusal) as caught:
        driver.run(sources(), "inner", out, event_ids=["E1", "E2"], prereg=_f_receipt)
    assert caught.value.details["reason"] == "f_inputs_missing"
    assert not (out / "receipts").exists()


def test_a_none_quote_reader_in_the_driver_refuses_before_any_receipt(tmp_path, m0, pricing, f_pricing):
    out = tmp_path / "out"
    with pytest.raises(fvi.FairValueRefusal) as caught:
        driver.run(sources(), "inner", out, event_ids=["E1", "E2"], prereg=_f_receipt, f_inputs=sel.FInputs(_pins(), None, "snap-1"))
    assert caught.value.details["reason"] == "f_quotes_invalid" and caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not (out / "receipts").exists()


def test_a_current_marks_run_has_no_fill_model_in_its_inputs(tmp_path, m0, pricing, monkeypatch):
    use_chunks(monkeypatch, [("2023Q3", chunk("E1")), ("2024Q3", chunk("E2"))])
    out = tmp_path / "out"
    driver.run(sources(), "inner", out, event_ids=["E1", "E2"], prereg=OK)
    assert "fill_model" not in receipts.load_receipt(out / "receipts" / "inputs_inner.json")


# ---- 5. the real pricing path: F's fills reach the scored entry and exit ----------------

NAME = "CND-PS"
PUT_QUOTES = {94.0: (1.15, 1.25), 97.0: (0.95, 1.05), 103.0: (0.95, 1.05), 106.0: (1.15, 1.25)}
EVENT = pd.DataFrame({"event_id": ["E0"], "event_date": [pd.Timestamp(wq.ENTRY) + pd.Timedelta(days=1)],
                      "pred_move": [wq.PRED], "mcap_usd": [1e10]})
SIMS = {"E0": (np.arange(60.0, 141.0), 0.001 * (np.arange(60.0, 141.0) - 100.0) ** 2)}


def test_real_pricing_scores_the_f_debit_and_proceeds_from_the_tables(tmp_path):
    """Real tables, the real load. CND-PS on E0 sells the 97 and 103 puts (bid 0.95, ask 1.05) and buys the 94 and 106
    puts (bid 1.15, ask 1.25), the same quotes on both dates. Vol 2.5 (vol_scale 0.1 on the pinned smvVol 25) puts
    every model value in the tens, far above its ask, so V = ask on each leg.

    Entry: buys fill at 1.25; sells at 1.05 - c * 0.10 with c = 0.25 + 0.50 * (0.10 / 1.00 / 0.50) = 0.35 (mid 1.00).
    debit_F = 2 * 1.25 - 2 * 1.015 = 0.47; debit_mid = debit_fair = 0.40; 0.07 <= 0.25 * 0.40; spread_sum 0.40 <= 0.47.
    Exit: each opening long sells at 1.25 - c * 0.10 with c = 0.25 + 0.50 * (0.10 / 1.20 / 0.50); each short buys back
    at its ask 1.05: proceeds_F = 2 * (1.25 - c * 0.10) - 2 * 1.05 = 1/3.
    """
    quotes = {("P", k): q for k, q in PUT_QUOTES.items()}
    frames = wq._frames()
    frames[("AAA", wq._ts(wq.ENTRY))] = wq._listing("AAA", wq.ENTRY, quotes=quotes)
    frames[("AAA", wq._ts(wq.EXIT))] = wq._listing("AAA", wq.EXIT, quotes=quotes)
    base = wq._base(frames)
    plan = wq._plan(("E0", "AAA", wq.ENTRY, wq.EXIT))
    tables = wq._write(tmp_path, base).rename(tmp_path / "tables")
    pins = wq._pins(tables)
    manifest_blob = (tables / "manifest.json").read_bytes()
    assert json.loads(manifest_blob)["units"] == UNITS
    assert hashlib.sha256(manifest_blob).hexdigest() == pins["manifest_sha256"]
    fill = _fill(vol_scale=0.1, reingest={"path": "tables", "manifest_sha256": pins["manifest_sha256"],
                                          "manifest_units": UNITS})

    run = sel.open_run(fill, sel.FInputs(pins, wq._reader(base), "snap-1"), plan, "snap-1", root=tmp_path)
    assert run.binding["reingest_manifest_sha256"] == pins["manifest_sha256"]
    assert run.binding["units"]["manifest_units"] == UNITS

    chunk = wq._chunk(frames, plan)
    _, legs, reason = sv.resolve_priced(chunk, 0, NAME, wq.PRED, FillModel(0.5))
    assert reason is None
    traded = [leg for leg in legs if leg["qty"] > 0]
    assert sorted((leg["strike"], leg["side"], leg["qty"]) for leg in traded) == [
        (94.0, "buy", 1.0), (97.0, "sell", 1.0), (103.0, "sell", 1.0), (106.0, "buy", 1.0)]

    asks = {94.0: 1.25, 97.0: 1.05, 103.0: 1.05, 106.0: 1.25}
    table = run.pricing.fills._quotes.legs(chunk, 0, traded)
    assert list(table["V"]) == pytest.approx([asks[leg["strike"]] for leg in traded])

    scored = run.pricing.score_events(chunk, EVENT, SIMS, NAME, alpha=0.5)
    rows = run.pricing.score(chunk, EVENT, scored.assign(status="taken"), NAME, 0.5)
    c_entry = 0.25 + 0.50 * min((0.10 / 1.00) / 0.50, 1.0)
    c_exit = 0.25 + 0.50 * min((0.10 / 1.20) / 0.50, 1.0)
    entry_want = 2 * 1.25 - 2 * (1.05 - c_entry * (1.05 - 0.95))
    exit_want = 2 * (1.25 - c_exit * (1.25 - 1.15)) - 2 * 1.05
    assert entry_want == pytest.approx(0.47, abs=1e-12)
    assert exit_want == pytest.approx(1 / 3, abs=1e-12)
    assert float(scored.iloc[0]["entry_cost"]) == pytest.approx(entry_want, abs=1e-12)
    assert float(rows.iloc[0]["exit_value"]) == pytest.approx(exit_want, abs=1e-12)

    current = seam.load().score_events(chunk, EVENT, SIMS, NAME, alpha=0.5)
    current_rows = seam.load().score(chunk, EVENT, current.assign(status="taken"), NAME, 0.5)
    assert float(current.iloc[0]["entry_cost"]) != pytest.approx(entry_want, abs=1e-9)
    assert float(current_rows.iloc[0]["exit_value"]) != pytest.approx(exit_want, abs=1e-9)
