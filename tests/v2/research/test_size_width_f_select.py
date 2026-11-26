"""EXP-189 Amendment 1, slice W7d: F selection from the spec and the receipt binding.

Synthetic only: fixture spec dicts and tmp_path spec files, never the real spec.yaml selection, never data/."""
import hashlib
import json
import types
from pathlib import Path

import pytest

from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as fvr
from experiments import size_pnl_receipts as receipts
from experiments import size_width_deviation as deviation
from experiments import size_width_driver as driver
from experiments import size_width_f_fills, size_width_f_quotes
from experiments import size_width_f_select as sel
from experiments import size_width_pricing as seam
from experiments import size_width_prereg
from experiments.size_pnl_shortvol import Refusal
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
