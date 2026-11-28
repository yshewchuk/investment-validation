"""EXP-189 Amendment 2, slice S3: the report accepts a model A out-dir, and check_report binds model A (synthetic receipts only, no market data).

Expected strings are hand-written here; the oracle is the spec's text, not the code under test."""
import json

import pytest

from experiments import fair_value_inputs as fvi
from experiments import fair_value_run as fvr
from experiments import size_pnl_receipts as receipts
from experiments import size_width_f_select as sel
from experiments import size_width_report as report
from experiments.size_pnl_shortvol import NAMES
from experiments.size_width_f_fills import ENTRY_REASONS
from experiments.size_width_success import LADDER
from test_size_width_f_report import HALVES, _add_views, _f_run, _mutate, _refused, _retune, _text
from test_size_width_report import _write

A_REASON = "f_fill_over_mid"
INNER_SHA, OUTER_SHA = "1" * 64, "2" * 64
A_FILL = {"fill_model_id": "A", "fvf_id": "fake-fvf", "constants_id": "c" * 64, "variant": "primary",
          "model_a_source_sha256": {"size_width_a_fills.py": "3" * 64}, "carry_dividend_mode": "fake-mode",
          "every_table_file_sha256": {"quotes/year=2024/month=02.parquet": "4" * 64}, "quote_snapshot_id": "snap-1",
          "rate_empty_row": "fake-empty", "rate_out_of_range": "fake-range", "reingest_manifest_sha256": "5" * 64,
          "units": {"vol_scale": 0.01, "rate_scale": 0.01, "manifest_units": {"fake": "unit"}}}
F_FILL = {"fill_model_id": "F", "fvf_id": "fake-fvf"}
A_HEAD = ("POST-HOC (Amendment 2, model A fills): written after Amendment 1's results had been seen, and model A was fitted on the user's "
          "own fills; not a pre-registered test and nothing in it is blind. The original run stays EXP-189's only pre-registered result.")
NOTE_A_PRIMARY = ("Fill model A, variant primary. The verdict is read on this primary run, with spread050 shown beside it (its own run and "
                  "report; reported as not run if it was not required and not run).")
NOTE_A_SPREAD = "Fill model A, variant spread050. This is the sensitivity run; the verdict is read on the primary run."
NOTE_B = ("The fills used to fit model A all postdate every EXP-189 test period (2023H2, 2024, 2025): no overlap. Because those fills "
          "postdate every test period, the constants use information that was not available at any decision date in these runs.")
NOTE_C = ("Comparison with F: against Amendment 1's stored report, between whole runs (the taken sets can differ), not trade by trade; "
          "F is not recomputed here and the comparison is written by hand.")
NOTE_D = ("Exit concession: no model A statistic is defined; its inputs are stored raw under report_inputs/. The midfloor view equals the "
          "base under model A.")


def _add_a_counts(results, periods):
    """test_size_width_f_report's _add_counts for model A: the same counts, with f_fill_over_mid where f_fill_over_fair sat."""
    for label, by in results.items():
        for p in periods:
            res = by[p]
            res["account"]["nonpositive_debit"] = {f"{a:.2f}": 1 for a in LADDER}
            for n in NAMES:
                nt = {"candidates": 10, "f_skipped": 2, "reasons": {ENTRY_REASONS[0]: 1, A_REASON: 1, "below_cutoff": 4},
                      "only_arm_f_skipped": 1, "only_production_f_skipped": 3}
                if label == "E":
                    nt["any_k_f_skipped_share"] = 0.2
                res["structures"][n]["nontrades"] = nt


def _a_run(out, fill=A_FILL):
    """A synthetic model A run: _write's receipts retuned per period, the A binding in both inputs (a constants_file_sha256 that
    differs between them), beside-A views and model A's non-trade counts in both summaries."""
    _write(out, passing=("A",))
    _mutate(out, "inputs_inner.json", lambda b: b.update(fill_model=fill, constants_file_sha256=INNER_SHA))
    _mutate(out, "inputs_outer.json", lambda b: b.update(fill_model=fill, constants_file_sha256=OUTER_SHA))
    _mutate(out, "inner_summary.json",
            lambda b: (_retune(b["results"], HALVES), _add_views(b["results"], HALVES), _add_a_counts(b["results"], HALVES)))
    _mutate(out, "outer_summary.json",
            lambda b: (_retune(b["results"], ("2025",)), _add_views(b["results"], ("2025",)), _add_a_counts(b["results"], ("2025",))))


def _nontrades_2024(out, nt):
    """Replaces label A's first-structure non-trade counts in 2024 of the inner summary with ``nt``."""
    _mutate(out, "inner_summary.json", lambda b: b["results"]["A"]["2024"]["structures"][NAMES[0]].update(nontrades=nt))


def test_a_primary_run_is_labelled_renamed_and_carries_its_notes(tmp_path):
    _a_run(tmp_path)
    report.assemble(tmp_path)
    text = _text(tmp_path)
    label = "POST-HOC (Amendment 2, model A fills, primary)"
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings and all(line.endswith(f"[{label}]") for line in headings)
    assert json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))["label"] == label
    assert A_HEAD in text
    for note in (NOTE_A_PRIMARY, NOTE_B, NOTE_C, NOTE_D):
        assert note in text
    assert "### T5 production versus arms ($200,000 account at model A fills; 2025 NOT blind for widths)" in text
    assert "| arm | period | model A | mid | harsh | unknown0 | unknown1 | midfloor |" in text
    assert "### T2c F entry non-trades by reason, arm, year and structure" not in text
    t2c = "| arm | year | structure | candidates | f_unpriceable | f_rate_unknown | f_no_fair_value | f_not_a_debit | f_fill_over_mid | "
    assert t2c + "f_spread_over_debit | model A-skipped | only this arm model A-skipped | only production model A-skipped | " \
           "E: events with a k model A-skipped % |" in text
    assert "f_fill_over_fair" not in text
    assert "Amendment 1, F fills" not in text and "at marks_mid" not in text


@pytest.mark.parametrize("variant, note", [("primary", NOTE_A_PRIMARY), ("spread050", NOTE_A_SPREAD)])
def test_a_run_of_either_variant_passes_with_its_label_and_its_note(tmp_path, variant, note):
    _a_run(tmp_path, {**A_FILL, "variant": variant})
    report.assemble(tmp_path)
    text = _text(tmp_path)
    label = f"POST-HOC (Amendment 2, model A fills, {variant})"
    assert note in text
    assert json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))["label"] == label
    assert all(line.endswith(f"[{label}]") for line in text.splitlines() if line.startswith("#"))


def test_f_receipts_still_pass_with_the_amendment_1_label(tmp_path):
    _f_run(tmp_path)
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))["label"] == "POST-HOC (Amendment 1, F fills)"
    assert "| arm | period | F | mid | harsh | unknown0 | unknown1 | midfloor |" in text
    assert "POST-HOC (Amendment 2" not in text and "model A" not in text


@pytest.mark.parametrize("outer_fill", [
    F_FILL,
    {**A_FILL, "variant": "spread050"},
    {**A_FILL, "constants_id": "d" * 64},
    {**A_FILL, "quote_snapshot_id": "snap-2"},
], ids=["inner A outer F", "inner primary outer spread050", "constants_id differs", "quote_snapshot_id differs"])
def test_inner_and_outer_fill_models_that_differ_refuse_and_write_nothing(tmp_path, outer_fill):
    _a_run(tmp_path)
    _mutate(tmp_path, "inputs_outer.json", lambda b: b.update(fill_model=outer_fill))
    _refused(tmp_path, "tuning_receipt_changed")


@pytest.mark.parametrize("fill", [
    {**A_FILL, "fill_model_id": "B"},
    {**A_FILL, "fill_model_id": ["A"]},
    {**A_FILL, "variant": "other"},
    {**A_FILL, "variant": ["primary"]},
    {k: v for k, v in A_FILL.items() if k != "constants_id"},
], ids=["id B", "id a list", "variant other", "variant a list", "constants_id missing"])
def test_a_fill_model_that_does_not_bind_a_refuses_with_both_receipts_equal(tmp_path, fill):
    _a_run(tmp_path, fill)
    _refused(tmp_path, "tuning_receipt_changed")


@pytest.mark.parametrize("f_skipped", [1, 2])
def test_an_a_run_with_a_stored_f_fill_over_fair_count_refuses(tmp_path, f_skipped):
    _a_run(tmp_path)
    _nontrades_2024(tmp_path, {"candidates": 10, "f_skipped": f_skipped,
                               "reasons": {ENTRY_REASONS[0]: 1, "f_fill_over_fair": 1},
                               "only_arm_f_skipped": 1, "only_production_f_skipped": 3})
    _refused(tmp_path, "tuning_receipt_changed")


@pytest.mark.parametrize("f_skipped", [1, 2])
def test_an_f_run_with_a_stored_f_fill_over_mid_count_refuses(tmp_path, f_skipped):
    _f_run(tmp_path)
    _nontrades_2024(tmp_path, {"candidates": 10, "f_skipped": f_skipped,
                               "reasons": {ENTRY_REASONS[0]: 1, "f_fill_over_mid": 1},
                               "only_arm_f_skipped": 1, "only_production_f_skipped": 3})
    _refused(tmp_path, "tuning_receipt_changed")


SEL_A = {"id": "A", "constants_id": "c" * 64, "fvf_id": fvr.fvf_id()}
SEL_F = {"id": "F", "fvf_id": fvr.fvf_id()}


def _check_inputs(out, fill):
    """An inputs_inner receipt under ``out`` binding ``fill`` under the spec hash "SPEC"."""
    (out / "receipts").mkdir(parents=True, exist_ok=True)
    receipts.write_receipt(out / "receipts" / "inputs_inner.json", {"spec_hash": "SPEC", "fill_model": fill})


@pytest.mark.parametrize("variant", ["primary", "spread050"])
def test_check_report_passes_model_a_inputs_for_the_selected_constants_and_a_variant(tmp_path, variant):
    _check_inputs(tmp_path, {"fill_model_id": "A", "constants_id": "c" * 64, "variant": variant})
    assert sel.check_report({"spec_hash": "SPEC", "fill_model": SEL_A}, tmp_path) is None


@pytest.mark.parametrize("selected, inputs_fill", [
    (SEL_A, {"fill_model_id": "A", "constants_id": "d" * 64, "variant": "primary"}),
    (SEL_A, {"fill_model_id": "A", "constants_id": "c" * 64, "variant": "other"}),
    (SEL_A, {"fill_model_id": "F"}),
    (SEL_F, {"fill_model_id": "A", "constants_id": "c" * 64, "variant": "primary"}),
], ids=["wrong constants_id", "unknown variant", "F inputs under an A selection", "A inputs under an F selection"])
def test_check_report_refuses_inputs_that_do_not_bind_the_selected_model(tmp_path, selected, inputs_fill):
    _check_inputs(tmp_path, inputs_fill)
    with pytest.raises(fvi.FairValueRefusal) as caught:
        sel.check_report({"spec_hash": "SPEC", "fill_model": selected}, tmp_path)
    assert caught.value.details["reason"] == "f_report_unbound"
