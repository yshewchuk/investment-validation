"""EXP-189 Amendment 1, slice W7c2 part 1: F-run non-trade and non-positive-debit tables, their admission, and the report wiring.

Synthetic only: hand-written counts, a real F sweep on synthetic quotes (the fixtures of the sibling F tests), and synthetic receipts.
"""
import json

import pytest

from experiments import size_pnl_receipts as receipts
from experiments import size_pnl_shortvol as sv
from experiments import size_width_f_fills as ff
from experiments import size_width_f_report_counts as counts
from experiments import size_width_f_views as f_views
from experiments import size_width_pricing as seam
from experiments import size_width_report as report
from experiments import size_width_tuning as tuning
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_f_fills import ENTRY_REASONS
from experiments.size_width_sweep import E_ARM, Sweep
from test_size_width_f_report import _f_run
from test_size_width_report import _write
from tests.v2.research.test_size_width_f_fills import NAME, _tight_frames
from tests.v2.research.test_size_width_f_quotes import DEFAULT_PLAN, _frames, _setup
from tests.v2.research.test_size_width_f_rows import WINDOW, _all_taken, _month

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
LABELS, PERIODS = ("A", "E"), ("2024",)


def _nt(label):
    """A valid structure's counts for ``label``: 2 of 10 candidates F-skipped, both by F reasons; E carries a 0.2 share."""
    nt = {"candidates": 10, "f_skipped": 2, "reasons": {ENTRY_REASONS[0]: 1, ENTRY_REASONS[4]: 1, "below_cutoff": 4},
          "only_arm_f_skipped": 1, "only_production_f_skipped": 3}
    if label == "E":
        nt["any_k_f_skipped_share"] = 0.2
    return nt


def _body():
    """Labels A and E, one period, every structure of NAMES with valid counts and one non-positive debit per alpha on the account."""
    return {label: {"2024": {"account": {"trades": 400, "nonpositive_debit": {a: 1 for a in ("0.00", "0.25", "0.50", "0.75", "1.00")}},
                             "structures": {n: {"nontrades": _nt(label)} for n in NAMES}}} for label in LABELS}


def _at(body, label="A"):
    """The first structure's non-trade counts of ``label`` in the one period."""
    return body[label]["2024"]["structures"][NAMES[0]]["nontrades"]


def _debit(body, label="A"):
    return body[label]["2024"]["account"]["nonpositive_debit"]


def _empty_with_share(body):
    _at(body, "E").update(candidates=0, f_skipped=0, reasons={}, only_arm_f_skipped=0, only_production_f_skipped=0)


MUTATIONS = [
    ("missing nontrades", lambda b: b["A"]["2024"]["structures"][NAMES[0]].pop("nontrades")),
    ("missing one structure", lambda b: b["A"]["2024"]["structures"].pop(NAMES[-1])),
    ("extra key", lambda b: _at(b).update(extra=0)),
    ("bool count", lambda b: _at(b).update(only_arm_f_skipped=True)),
    ("negative count", lambda b: _at(b).update(only_arm_f_skipped=-1)),
    ("float count", lambda b: _at(b).update(f_skipped=2.0)),
    ("None only_arm_f_skipped", lambda b: _at(b).update(only_arm_f_skipped=None)),
    ("f_skipped above candidates", lambda b: _at(b).update(candidates=1)),
    ("only_arm above f_skipped", lambda b: _at(b).update(only_arm_f_skipped=3)),
    ("only_production above candidates minus f_skipped", lambda b: _at(b).update(only_production_f_skipped=9)),
    ("F reason sum differing from f_skipped", lambda b: _at(b)["reasons"].update({ENTRY_REASONS[0]: 2})),
    ("non-dict reasons", lambda b: _at(b).update(reasons=[1, 1])),
    ("E without share", lambda b: _at(b, "E").pop("any_k_f_skipped_share")),
    ("non-E with share", lambda b: _at(b).update(any_k_f_skipped_share=0.2)),
    ("share None with candidates above zero", lambda b: _at(b, "E").update(any_k_f_skipped_share=None)),
    ("share present with no candidates", lambda b: _empty_with_share(b)),
    ("share 1.5", lambda b: _at(b, "E").update(any_k_f_skipped_share=1.5)),
    ("share True", lambda b: _at(b, "E").update(any_k_f_skipped_share=True)),
    ("share NaN", lambda b: _at(b, "E").update(any_k_f_skipped_share=float("nan"))),
    ("nonpositive_debit missing", lambda b: b["A"]["2024"]["account"].pop("nonpositive_debit")),
    ("missing one alpha key", lambda b: _debit(b).pop("0.50")),
    ("extra alpha key", lambda b: _debit(b).update({"0.10": 1})),
    ("debit count above trades", lambda b: _debit(b).update({"1.00": 401})),
    ("non-int non-F reason", lambda b: _at(b)["reasons"].update(below_cutoff=4.0)),
    ("bool non-F reason", lambda b: _at(b)["reasons"].update(below_cutoff=True)),
    ("negative non-F reason", lambda b: _at(b)["reasons"].update(below_cutoff=-1)),
    ("bool nonpositive_debit", lambda b: _debit(b).update({"1.00": True})),
    ("negative debit", lambda b: _debit(b).update({"1.00": -1})),
    ("float debit", lambda b: _debit(b).update({"1.00": 1.0})),
]


def test_counts_markdown_is_the_literal_table(monkeypatch):
    monkeypatch.setattr(counts, "NAMES", ("BFLY-P", "CTR5"))
    results = {
        "A": {"2024": {"structures": {
            "BFLY-P": {"nontrades": {"candidates": 0, "f_skipped": 0, "reasons": {}, "only_arm_f_skipped": 0,
                                     "only_production_f_skipped": 0}},
            "CTR5": {"nontrades": {"candidates": 8, "f_skipped": 3, "reasons": {"f_unpriceable": 2, "f_fill_over_fair": 1, "below_cutoff": 5},
                                   "only_arm_f_skipped": 1, "only_production_f_skipped": 2}}}}},
        "E": {"2024": {"structures": {
            "BFLY-P": {"nontrades": {"candidates": 0, "f_skipped": 0, "reasons": {}, "only_arm_f_skipped": 0,
                                     "only_production_f_skipped": 0, "any_k_f_skipped_share": None}},
            "CTR5": {"nontrades": {"candidates": 8, "f_skipped": 3, "reasons": {"f_rate_unknown": 1, "f_spread_over_debit": 2, "below_cutoff": 4},
                                   "only_arm_f_skipped": 0, "only_production_f_skipped": 1, "any_k_f_skipped_share": 0.375}}}}},
    }
    expected = """### T2c test

| arm | year | structure | candidates | f_unpriceable | f_rate_unknown | f_no_fair_value | f_not_a_debit | f_fill_over_fair | f_spread_over_debit | F-skipped | only this arm F-skipped | only production F-skipped | E: events with a k F-skipped % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 2024 | BFLY-P | 0 | 0 (n/a) | 0 (n/a) | 0 (n/a) | 0 (n/a) | 0 (n/a) | 0 (n/a) | 0 (n/a) | 0 | 0 | n/a |
| A | 2024 | CTR5 | 8 | 2 (25.00) | 0 (0.00) | 0 (0.00) | 0 (0.00) | 1 (12.50) | 0 (0.00) | 3 (37.50) | 1 | 2 | n/a |
| E | 2024 | BFLY-P | 0 | 0 (n/a) | 0 (n/a) | 0 (n/a) | 0 (n/a) | 0 (n/a) | 0 (n/a) | 0 (n/a) | 0 | 0 | n/a |
| E | 2024 | CTR5 | 8 | 0 (0.00) | 1 (12.50) | 0 (0.00) | 0 (0.00) | 0 (0.00) | 2 (25.00) | 3 (37.50) | 0 | 1 | 37.50 |"""
    assert counts.counts_markdown("T2c test", ("A", "E"), ("2024",), results) == expected


def test_nonpositive_markdown_is_the_literal_table():
    results = {
        "A": {"2023H2": {"account": {"trades": 400, "nonpositive_debit": {"0.00": 0, "0.25": 1, "0.50": 2, "0.75": 3, "1.00": 4}}},
              "2024": {"account": {"trades": 410, "nonpositive_debit": {"0.00": 1, "0.25": 2, "0.50": 3, "0.75": 4, "1.00": 5}}}},
        "E": {"2023H2": {"account": {"trades": 380, "nonpositive_debit": {"0.00": 5, "0.25": 6, "0.50": 7, "0.75": 8, "1.00": 9}}},
              "2024": {"account": {"trades": 390, "nonpositive_debit": {"0.00": 0, "0.25": 0, "0.50": 1, "0.75": 1, "1.00": 2}}}},
    }
    expected = """### T5d F-selected taken trades with no ladder return (repriced debit not positive), seven structures, by arm and alpha (2025 NOT blind for widths)

| arm | year | trades | alpha 0.00 | alpha 0.25 | alpha 0.50 | alpha 0.75 | alpha 1.00 |
|---|---|---|---|---|---|---|---|
| A | 2023H2 | 400 | 0 | 1 | 2 | 3 | 4 |
| A | 2024 | 410 | 1 | 2 | 3 | 4 | 5 |
| E | 2023H2 | 380 | 5 | 6 | 7 | 8 | 9 |
| E | 2024 | 390 | 0 | 0 | 1 | 1 | 2 |"""
    assert counts.nonpositive_markdown(("A", "E"), ("2023H2", "2024"), results) == expected


def test_admit_accepts_a_valid_body():
    assert counts.admit(_body(), LABELS, PERIODS)


def test_admit_accepts_a_structure_with_no_candidates_and_no_share():
    body = _body()
    body["E"]["2024"]["structures"][NAMES[0]]["nontrades"] = {"candidates": 0, "f_skipped": 0, "reasons": {}, "only_arm_f_skipped": 0,
                                                             "only_production_f_skipped": 0, "any_k_f_skipped_share": None}
    assert counts.admit(body, LABELS, PERIODS)


def test_admit_accepts_boundary_equal_counts():
    # Case 1: f_skipped == candidates, only_arm == f_skipped, only_production == candidates - f_skipped == 0.
    # Case 2: only_arm < f_skipped, only_production == candidates - f_skipped > 0. Reasons sum to f_skipped in both.
    body = _body()
    body["A"]["2024"]["structures"][NAMES[0]]["nontrades"] = {"candidates": 2, "f_skipped": 2, "reasons": {ENTRY_REASONS[0]: 1, ENTRY_REASONS[4]: 1},
                                                             "only_arm_f_skipped": 2, "only_production_f_skipped": 0}
    body["A"]["2024"]["structures"][NAMES[1]]["nontrades"] = {"candidates": 10, "f_skipped": 2, "reasons": {ENTRY_REASONS[0]: 1, ENTRY_REASONS[4]: 1, "below_cutoff": 8},
                                                             "only_arm_f_skipped": 1, "only_production_f_skipped": 8}
    _debit(body)["1.00"] = 400
    assert counts.admit(body, LABELS, PERIODS)


def test_admit_refuses_none_and_malformed_input():
    assert not counts.admit(None, LABELS, PERIODS)
    assert not counts.admit({}, LABELS, PERIODS)


@pytest.mark.parametrize("mutate", [m[1] for m in MUTATIONS], ids=[m[0] for m in MUTATIONS])
def test_admit_refuses_a_single_malformed_value(mutate):
    body = _body()
    mutate(body)
    assert not counts.admit(body, LABELS, PERIODS)


def _f_sweep(quotes):
    """An F sweep over the one structure NAME with the ladder on and every entry taken (decide stands in)."""
    pricing = seam.load(fills=ff.FFills(quotes))
    pricing.decide = _all_taken
    return Sweep(ks=(1.0,), names=(NAME,), pricing=pricing, ladder=True)


def test_real_f_evaluate_output_passes_admit_and_is_json_safe(tmp_path, monkeypatch):
    # This sweep prices one structure; the other six come back from evaluate with no candidates and no production comparison
    # (their only-counts are None), so admit is checked for NAME alone.
    monkeypatch.setattr(counts, "NAMES", (NAME,))
    # AAA's entry quotes are tight, BBB keeps the default wide quotes: E1 is an F skip (fill over fair), E0 is taken.
    frames = {**_tight_frames(), **{key: frame for key, frame in _frames().items() if key[0] == "BBB"}}
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN)
    sweep = _f_sweep(fq)
    _month(sweep, chunk)
    got = {"A": {"2024": tuning.evaluate(sweep, {n: 1.0 for n in sv.NAMES}, WINDOW)},
           "E": {"2024": tuning.evaluate(sweep, {n: E_ARM for n in sv.NAMES}, WINDOW)}}
    got = json.loads(json.dumps(got, allow_nan=False))
    assert counts.admit(got, LABELS, PERIODS)
    nt = got["A"]["2024"]["structures"][NAME]["nontrades"]
    assert nt["candidates"] == 2 and nt["f_skipped"] == 1 and nt["reasons"] == {"f_fill_over_fair": 1}
    # any_k_f_skipped_share = |events F-skipped at some grid k, among E's candidates| / |E's candidates|. With ks=(1.0,) the only
    # F-skipped event is E1, and E's 2 candidates include it: 1 / 2.
    assert got["E"]["2024"]["structures"][NAME]["nontrades"]["any_k_f_skipped_share"] == 0.5


def _corrupt(out, name, fn):
    """Applies ``fn`` to receipt ``name``'s body and rewrites it as a fresh receipt (its own digest recomputed)."""
    path = out / "receipts" / name
    body = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k != "content_hash"}
    fn(body)
    path.unlink()
    receipts.write_receipt(path, body)


def _refuse(out):
    """assemble refuses as an identity conflict for tuning_receipt_changed and writes neither REPORT.md nor proposed_rule.json."""
    with pytest.raises(Refusal) as caught:
        report.assemble(out)
    assert caught.value.code == CONFLICT and caught.value.details["reason"] == "tuning_receipt_changed"
    assert "the F counts in results are missing, malformed or inconsistent" in str(caught.value)
    assert not (out / "REPORT.md").exists() and not (out / "proposed_rule.json").exists()


def test_f_report_renders_t2c_t5c_t5d_in_order_with_the_post_hoc_label(tmp_path):
    _f_run(tmp_path)
    report.assemble(tmp_path)
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    label = f_views.LABEL
    t2c = (f"### T2c F entry non-trades by reason, arm, year and structure (2023-24 tuning years; shares are % of candidates) "
           f"[{label}]")
    t5c = (f"### T5c F entry non-trades by reason, arm, year and structure (2025 NOT blind for widths; shares are % of candidates) "
           f"[{label}]")
    t5d = (f"### T5d F-selected taken trades with no ladder return (repriced debit not positive), seven structures, by arm and alpha "
           f"(2025 NOT blind for widths) [{label}]")
    assert t2c in text and t5c in text and t5d in text
    assert text.index("### T2b arm E") < text.index(t2c) < text.index("### T3 curves")
    assert text.index("### T5 beside F: return per trade") < text.index(t5c) < text.index(t5d) < text.index("### T6 ")
    assert "| E | 2024 | BFLY-P | 10 | 1 (10.00) | 0 (0.00) | 0 (0.00) | 0 (0.00) | 1 (10.00) | 0 (0.00) | 2 (20.00) | 1 | 3 | 20.00 |" in text
    assert "| A | 2024 | 400 | 1 | 1 | 1 | 1 | 1 |" in text


def test_corrupted_outer_count_refuses_before_anything_is_written(tmp_path):
    _f_run(tmp_path)
    _corrupt(tmp_path, "outer_summary.json",
             lambda b: b["results"]["A"]["2025"]["structures"][NAMES[0]]["nontrades"].update(only_production_f_skipped=9))
    _refuse(tmp_path)


def test_corrupted_inner_count_refuses_before_anything_is_written(tmp_path):
    _f_run(tmp_path)
    _corrupt(tmp_path, "inner_summary.json", lambda b: b["results"]["B"]["2024"]["account"]["nonpositive_debit"].update({"0.25": 401}))
    _refuse(tmp_path)


def test_non_f_report_has_no_counts_sections(tmp_path):
    _write(tmp_path, passing=("A",))
    report.assemble(tmp_path)
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "T2c" not in text and "T5c" not in text and "T5d" not in text
