"""EXP-189 Amendment 1, slice W7c1: F-run report labels, beside-F views and refusals (synthetic receipts only, no market data)."""
import copy
import json

import pytest

from engine.v2.foundation import content_hash
from experiments import size_pnl_receipts as receipts
from experiments import size_width_f_views as f_views
from experiments import size_width_report as report
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_f_fills import ENTRY_REASONS
from experiments.size_width_success import LADDER
from test_size_width_report import _write

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
FILL = {"fill_model_id": "F", "quotes": "synthetic"}
HALVES = ("2023H2", "2024")
ACCOUNT_VIEWS = ("F", "mid", "harsh")
ALL_VIEWS = (*ACCOUNT_VIEWS, "unknown0", "unknown1", "midfloor")
SHIFT = {"2023H2": 0.0, "2024": 0.001, "2025": 0.002}
STEP = {"F": 0.0, "mid": 0.001, "harsh": 0.002, "unknown0": 0.003, "unknown1": 0.004, "midfloor": 0.005}
BOUND = {"inputs_inner.json": ("inner_summary.json", "inputs_inner"), "inputs_outer.json": ("outer_summary.json", "inputs_outer")}

# Literal expectations, derived by hand from _write(passing=("A",)): production account 0.01 and breakeven 0.45;
# A 0.02 and 0.40; B, C, D, E 0.005 and 0.60; trades 400; structures at the account return; stats alpha0 = ret - 0.03.
EXPECTED_T5 = """### T5 production versus arms ($200,000 account at marks_mid; 2025 NOT blind for widths)

| arm | period | trades | account return % | return per trade % | breakeven alpha | harsh alpha-0 return per trade % |
|---|---|---|---|---|---|---|
| production | 2025 | 400 | 1.00 | 1.00 | 0.450 | -2.00 |
| production | 2023H2 | 400 | 1.00 | 1.00 | 0.450 | -2.00 |
| production | 2024 | 400 | 1.00 | 1.00 | 0.450 | -2.00 |
| A | 2025 | 400 | 2.00 | 2.00 | 0.400 | -1.00 |
| A | 2023H2 | 400 | 2.00 | 2.00 | 0.400 | -1.00 |
| A | 2024 | 400 | 2.00 | 2.00 | 0.400 | -1.00 |
| B | 2025 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| B | 2023H2 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| B | 2024 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| C | 2025 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| C | 2023H2 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| C | 2024 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| D | 2025 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| D | 2023H2 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| D | 2024 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| E | 2025 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| E | 2023H2 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |
| E | 2024 | 400 | 0.50 | 0.50 | 0.600 | -2.50 |"""

EXPECTED_T6 = """### T6 per-structure stand-alone accounts (2025, return % at marks_mid; 2025 NOT blind for widths)

| structure | production | A | B | C | D | E |
|---|---|---|---|---|---|---|
| BFLY-P | 1.00 | 2.00 | 0.50 | 0.50 | 0.50 | 0.50 |
| BFLY-P5 | 1.00 | 2.00 | 0.50 | 0.50 | 0.50 | 0.50 |
| CND-PS | 1.00 | 2.00 | 0.50 | 0.50 | 0.50 | 0.50 |
| CTR5 | 1.00 | 2.00 | 0.50 | 0.50 | 0.50 | 0.50 |
| RAMP7 | 1.00 | 2.00 | 0.50 | 0.50 | 0.50 | 0.50 |
| TWIN-P | 1.00 | 2.00 | 0.50 | 0.50 | 0.50 | 0.50 |
| TWIN-P5 | 1.00 | 2.00 | 0.50 | 0.50 | 0.50 | 0.50 |"""

EXPECTED_HEADINGS = [
    "# EXP-189 report: production short-vol width tuning (M0)",
    "### T1 error model (M0, 2023-24 inner)",
    "### T2 chosen factors (2023-24 tuning)",
    "### T2b arm E: share of candidate events at each k",
    "### T3 curves per structure (2023-24, account return % at marks_mid; 2025 NOT blind for widths)",
    "#### BFLY-P", "#### BFLY-P5", "#### CND-PS", "#### CTR5", "#### RAMP7", "#### TWIN-P", "#### TWIN-P5",
    "### T4 anchor (perfect forecast, 2023-24, descriptive only)",
    "### T5 production versus arms ($200,000 account at marks_mid; 2025 NOT blind for widths)",
    "### T6 per-structure stand-alone accounts (2025, return % at marks_mid; 2025 NOT blind for widths)",
    "### T7 success table (2025 NOT blind for widths)",
    "### Proposed board width rule (arm A)",
    "### T8-T9 tolerance bands and fallback",
]

EXPECTED_RULE = {"recommendation": "A", "rows": [
    {"structure": "BFLY-P", "k": 0.75, "divisor": "divisor / 0.75", "flatness_se": 0.0, "declined_share": 0.1},
    {"structure": "BFLY-P5", "k": 0.75, "divisor": "divisor / 0.75", "flatness_se": 0.0, "declined_share": 0.1},
    {"structure": "CND-PS", "k": 0.75, "divisor": "divisor / 0.75", "flatness_se": 0.0, "declined_share": 0.1},
    {"structure": "CTR5", "k": 0.75, "divisor": "divisor / 0.75", "flatness_se": 0.0, "declined_share": 0.1},
    {"structure": "RAMP7", "k": 0.75, "divisor": "divisor / 0.75", "flatness_se": 0.0, "declined_share": 0.1},
    {"structure": "TWIN-P", "k": 0.75, "divisor": "divisor / 0.75", "flatness_se": 0.0, "declined_share": 0.1},
    {"structure": "TWIN-P5", "k": 0.75, "divisor": "divisor / 0.75", "flatness_se": 0.0, "declined_share": 0.1},
]}

# F-run T6 at mid and harsh, 2025: production 0.012, A 0.022, B..E 0.007 at the account; structure i adds 0.0001 * i; then the view step.
T6_MID = [
    "| BFLY-P | 1.30 | 2.30 | 0.80 | 0.80 | 0.80 | 0.80 |",
    "| BFLY-P5 | 1.31 | 2.31 | 0.81 | 0.81 | 0.81 | 0.81 |",
    "| CND-PS | 1.32 | 2.32 | 0.82 | 0.82 | 0.82 | 0.82 |",
    "| CTR5 | 1.33 | 2.33 | 0.83 | 0.83 | 0.83 | 0.83 |",
    "| RAMP7 | 1.34 | 2.34 | 0.84 | 0.84 | 0.84 | 0.84 |",
    "| TWIN-P | 1.35 | 2.35 | 0.85 | 0.85 | 0.85 | 0.85 |",
    "| TWIN-P5 | 1.36 | 2.36 | 0.86 | 0.86 | 0.86 | 0.86 |",
]
T6_HARSH = [
    "| BFLY-P | 1.40 | 2.40 | 0.90 | 0.90 | 0.90 | 0.90 |",
    "| BFLY-P5 | 1.41 | 2.41 | 0.91 | 0.91 | 0.91 | 0.91 |",
    "| CND-PS | 1.42 | 2.42 | 0.92 | 0.92 | 0.92 | 0.92 |",
    "| CTR5 | 1.43 | 2.43 | 0.93 | 0.93 | 0.93 | 0.93 |",
    "| RAMP7 | 1.44 | 2.44 | 0.94 | 0.94 | 0.94 | 0.94 |",
    "| TWIN-P | 1.45 | 2.45 | 0.95 | 0.95 | 0.95 | 0.95 |",
    "| TWIN-P5 | 1.46 | 2.46 | 0.96 | 0.96 | 0.96 | 0.96 |",
]


def _body(path):
    """A receipt's body: its JSON without content_hash."""
    return {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k != "content_hash"}


def _rewrite(path, body):
    """Replaces the receipt at ``path`` with ``body``, written as a fresh receipt."""
    path.unlink()
    receipts.write_receipt(path, body)


def _mutate(out, name, fn):
    """Applies ``fn`` to receipt ``name``'s body and rewrites it; the summary that binds an inputs receipt gets the new digest."""
    rec = out / "receipts"
    body = _body(rec / name)
    fn(body)
    _rewrite(rec / name, body)
    if name in BOUND:
        summary, key = BOUND[name]
        bound = _body(rec / summary)
        bound[key] = content_hash(body)
        _rewrite(rec / summary, bound)


def _edit(name, fn):
    """A mutator for receipt ``name``."""
    return lambda out: _mutate(out, name, fn)


def _retune(results, periods):
    """Each label's primary return shifted by its period; each structure sits 0.0001 * index above it, so the rows are distinct."""
    for by in results.values():
        for p in periods:
            res = by[p]
            ret = res["account"]["total_return"] + SHIFT[p]
            res["account"]["total_return"] = ret
            for i, n in enumerate(NAMES):
                res["structures"][n]["total_return"] = ret + 0.0001 * i


def _add_views(results, periods):
    """Beside-F views in place: F equal to the deciding total_return, every other view F plus its STEP, mean_ret one 0.001 below."""
    def cell(entry, view):
        total = entry["total_return"] + STEP[view]
        return {"total_return": total, "mean_ret": total - 0.001, "wanted": entry["trades"]}
    for by in results.values():
        for p in periods:
            res = by[p]
            res["account"]["views"] = {v: cell(res["account"], v) for v in ALL_VIEWS}
            for n in NAMES:
                res["structures"][n]["views"] = {v: cell(res["structures"][n], v) for v in ACCOUNT_VIEWS}


def _add_counts(results, periods):
    """Non-trade and non-positive-debit counts in place: 2 of 10 candidates F-skipped (one each by two F reasons), E's any-k share 0.2,
    and one non-positive debit per alpha on every account."""
    for label, by in results.items():
        for p in periods:
            res = by[p]
            res["account"]["nonpositive_debit"] = {f"{a:.2f}": 1 for a in LADDER}
            for n in NAMES:
                nt = {"candidates": 10, "f_skipped": 2, "reasons": {ENTRY_REASONS[0]: 1, ENTRY_REASONS[4]: 1, "below_cutoff": 4},
                      "only_arm_f_skipped": 1, "only_production_f_skipped": 3}
                if label == "E":
                    nt["any_k_f_skipped_share"] = 0.2
                res["structures"][n]["nontrades"] = nt


def _f_run(out, fill=FILL):
    """A synthetic F run: _write's receipts retuned per period, the F binding in both inputs, beside-F views and non-trade counts in both summaries."""
    _write(out, passing=("A",))
    for name in ("inputs_inner.json", "inputs_outer.json"):
        _mutate(out, name, lambda b: b.update(fill_model=fill))
    _mutate(out, "inner_summary.json",
            lambda b: (_retune(b["results"], HALVES), _add_views(b["results"], HALVES), _add_counts(b["results"], HALVES)))
    _mutate(out, "outer_summary.json",
            lambda b: (_retune(b["results"], ("2025",)), _add_views(b["results"], ("2025",)), _add_counts(b["results"], ("2025",))))


def _text(out):
    return (out / "REPORT.md").read_text(encoding="utf-8")


def _section(text, start):
    """The section that starts with ``start``, up to the next heading."""
    return text.split(start, 1)[1].split("\n### ")[0]


def _block(text, heading):
    """``heading`` and its section up to the next heading, trailing whitespace removed."""
    return (heading + text.split(heading, 1)[1].split("\n### ")[0]).rstrip()


def _refused(out, reason):
    """assemble refuses as an identity conflict under ``reason`` and writes neither REPORT.md nor proposed_rule.json."""
    with pytest.raises(Refusal) as caught:
        report.assemble(out)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == reason
    assert not (out / "REPORT.md").exists() and not (out / "proposed_rule.json").exists()


def _acct(b, period="2024"):
    """Label A's account entry for ``period`` in a summary body."""
    return b["results"]["A"][period]["account"]


def _struct(b, period="2024"):
    """Label A's first structure entry for ``period`` in a summary body."""
    return b["results"]["A"][period]["structures"][NAMES[0]]


def test_f_run_labels_headings_and_tables(tmp_path):
    _f_run(tmp_path)
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert "POST-HOC (Amendment 1, F fills): chosen after the full run under mid fills had been seen" in text
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings and all(line.endswith("[POST-HOC (Amendment 1, F fills)]") for line in headings)
    assert "at marks_mid" not in text
    assert "### T3 curves per structure (2023-24, account return % at F fills; 2025 NOT blind for widths)" in text
    assert json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))["label"] == "POST-HOC (Amendment 1, F fills)"

    t5 = _section(text, "### T5 production versus arms")
    assert "### T5 production versus arms ($200,000 account at F fills; 2025 NOT blind for widths)" in text
    assert "| arm | period | trades | account return % | return per trade % | breakeven alpha |" in t5
    assert "| production | 2025 | 400 | 1.20 | 1.10 | 0.450 |" in t5
    assert "| A | 2024 | 400 | 2.10 | 2.00 | 0.400 |" in t5
    acct = _section(text, "### T5 beside F: seven-structure")
    assert "| arm | period | F | mid | harsh | unknown0 | unknown1 | midfloor |" in acct
    assert "| production | 2025 | 1.20 | 1.30 | 1.40 | 1.50 | 1.60 | 1.70 |" in acct
    assert "| A | 2024 | 2.10 | 2.20 | 2.30 | 2.40 | 2.50 | 2.60 |" in acct
    per = _section(text, "### T5 beside F: return per trade")
    assert "| A | 2024 | 2.00 | 2.10 | 2.20 | 2.30 | 2.40 | 2.50 |" in per
    assert "| production | 2025 | 1.10 | 1.20 | 1.30 | 1.40 | 1.50 | 1.60 |" in per

    for view, rows in (("mid", T6_MID), ("harsh", T6_HARSH)):
        section = _section(text, f"### T6 beside F: per-structure stand-alone accounts at {view} ")
        assert (f"### T6 beside F: per-structure stand-alone accounts at {view} (2025, return %; reported, never deciding; "
                f"2025 NOT blind for widths) [POST-HOC (Amendment 1, F fills)]") in text
        assert "| structure | production | A | B | C | D | E |" in section
        for row in rows:
            assert row in section


def test_non_f_run_is_unlabelled(tmp_path):
    _write(tmp_path, passing=("A",))
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert "POST-HOC" not in text
    assert "### T3 curves per structure (2023-24, account return % at marks_mid; 2025 NOT blind for widths)" in text
    assert _block(text, "### T5 production versus arms") == EXPECTED_T5
    assert _block(text, "### T6 per-structure") == EXPECTED_T6
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings == EXPECTED_HEADINGS
    rule = json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))
    assert rule == EXPECTED_RULE
    assert "label" not in rule and "POST-HOC" not in json.dumps(rule)


CASES = [
    ("account views missing", _edit("inner_summary.json", lambda b: _acct(b).pop("views")), "tuning_receipt_changed"),
    ("structure views missing", _edit("inner_summary.json", lambda b: _struct(b).pop("views")), "tuning_receipt_changed"),
    ("account a view missing", _edit("inner_summary.json", lambda b: _acct(b)["views"].pop("midfloor")), "tuning_receipt_changed"),
    ("structure a view missing", _edit("inner_summary.json", lambda b: _struct(b)["views"].pop("harsh")), "tuning_receipt_changed"),
    ("account an extra view", _edit("inner_summary.json",
                                    lambda b: _acct(b)["views"].update(extra={"total_return": 0.02, "mean_ret": 0.02, "wanted": 400})),
     "tuning_receipt_changed"),
    ("structure an extra view", _edit("inner_summary.json",
                                      lambda b: _struct(b)["views"].update(extra={"total_return": 0.02, "mean_ret": 0.02, "wanted": 50})),
     "tuning_receipt_changed"),
    ("account wanted differing from trades", _edit("inner_summary.json", lambda b: _acct(b)["views"]["mid"].update(wanted=399)),
     "tuning_receipt_changed"),
    ("structure wanted differing from trades", _edit("inner_summary.json", lambda b: _struct(b)["views"]["mid"].update(wanted=49)),
     "tuning_receipt_changed"),
    ("account trades a bool", _edit("inner_summary.json", lambda b: _acct(b).update(trades=True)), "tuning_receipt_changed"),
    ("structure trades a bool", _edit("inner_summary.json", lambda b: _struct(b).update(trades=True)), "tuning_receipt_changed"),
    ("account trades negative", _edit("inner_summary.json", lambda b: _acct(b).update(trades=-400)), "tuning_receipt_changed"),
    ("structure trades negative", _edit("inner_summary.json", lambda b: _struct(b).update(trades=-50)), "tuning_receipt_changed"),
    ("account total_return a string", _edit("inner_summary.json", lambda b: _acct(b).update(total_return="0.02")), "tuning_receipt_changed"),
    ("account view total_return a string", _edit("inner_summary.json", lambda b: _acct(b)["views"]["mid"].update(total_return="0.02")),
     "tuning_receipt_changed"),
    ("structure view total_return a string", _edit("inner_summary.json", lambda b: _struct(b)["views"]["mid"].update(total_return="0.02")),
     "tuning_receipt_changed"),
    ("account view total_return a bool", _edit("inner_summary.json", lambda b: _acct(b)["views"]["mid"].update(total_return=True)),
     "tuning_receipt_changed"),
    ("structure view total_return a bool", _edit("inner_summary.json", lambda b: _struct(b)["views"]["mid"].update(total_return=True)),
     "tuning_receipt_changed"),
    ("account mean_ret None with trades", _edit("inner_summary.json", lambda b: _acct(b)["views"]["F"].update(mean_ret=None)),
     "tuning_receipt_changed"),
    ("structure mean_ret None with trades", _edit("inner_summary.json", lambda b: _struct(b)["views"]["F"].update(mean_ret=None)),
     "tuning_receipt_changed"),
    ("account F total_return differing from the deciding value",
     _edit("inner_summary.json", lambda b: _acct(b)["views"]["F"].update(total_return=_acct(b)["views"]["F"]["total_return"] + 0.001)),
     "tuning_receipt_changed"),
    ("structure F total_return differing from the deciding value",
     _edit("inner_summary.json", lambda b: _struct(b)["views"]["F"].update(total_return=_struct(b)["views"]["F"]["total_return"] + 0.001)),
     "tuning_receipt_changed"),
    ("F views missing in the outer summary", _edit("outer_summary.json", lambda b: _acct(b, "2025").pop("views")), "tuning_receipt_changed"),
    ("fill_model only on inputs_inner", _edit("inputs_outer.json", lambda b: b.pop("fill_model")), "tuning_receipt_changed"),
    ("fill_model differing between inner and outer",
     _edit("inputs_outer.json", lambda b: b.update(fill_model={"fill_model_id": "F", "quotes": "other"})), "tuning_receipt_changed"),
]


@pytest.mark.parametrize("mutate, reason", [(c[1], c[2]) for c in CASES], ids=[c[0] for c in CASES])
def test_f_run_refusals(tmp_path, mutate, reason):
    _f_run(tmp_path)
    mutate(tmp_path)
    _refused(tmp_path, reason)


def test_fill_model_that_does_not_bind_f_refuses(tmp_path):
    _f_run(tmp_path, fill={"fill_model_id": "mid", "quotes": "synthetic"})
    _refused(tmp_path, "tuning_receipt_changed")


@pytest.mark.parametrize("where", ["account", "structure"])
def test_f_views_in_a_non_f_run_refuse(tmp_path, where):
    _write(tmp_path, passing=("A",))
    if where == "account":
        view = {"F": {"total_return": 0.02, "mean_ret": 0.02, "wanted": 400}}
        _mutate(tmp_path, "inner_summary.json", lambda b: _acct(b).update(views=view))
    else:
        view = {"F": {"total_return": 0.02, "mean_ret": 0.02, "wanted": 50}}
        _mutate(tmp_path, "inner_summary.json", lambda b: _struct(b).update(views=view))
    _refused(tmp_path, "tuning_receipt_changed")


@pytest.mark.parametrize("where", ["account", "structure"])
def test_f_views_in_a_non_f_outer_summary_refuse(tmp_path, where):
    _write(tmp_path, passing=("A",))
    if where == "account":
        view = {"F": {"total_return": 0.02, "mean_ret": 0.02, "wanted": 400}}
        _mutate(tmp_path, "outer_summary.json", lambda b: _acct(b, "2025").update(views=view))
    else:
        view = {"F": {"total_return": 0.02, "mean_ret": 0.02, "wanted": 50}}
        _mutate(tmp_path, "outer_summary.json", lambda b: _struct(b, "2025").update(views=view))
    _refused(tmp_path, "tuning_receipt_changed")


def test_fill_model_that_is_not_a_mapping_refuses(tmp_path):
    _f_run(tmp_path, fill="F")
    _refused(tmp_path, "tuning_receipt_changed")


def _extra_outer_2024(out):
    """Outer results A also holds a 2024 cell: a copy of the inner one with different numbers (F view kept equal to the account)."""
    cell = copy.deepcopy(_body(out / "receipts" / "inner_summary.json")["results"]["A"]["2024"])
    cell["account"]["total_return"] += 0.01
    cell["account"]["views"]["F"]["total_return"] += 0.01
    _mutate(out, "outer_summary.json", lambda b: b["results"]["A"].update({"2024": cell}))


def _extra_inner_label(out):
    _mutate(out, "inner_summary.json", lambda b: b["results"].update(Z=copy.deepcopy(b["results"]["A"])))


KEY_CASES = [
    ("outer extra 2024 cell", _extra_outer_2024),
    ("inner missing label A", lambda out: _mutate(out, "inner_summary.json", lambda b: b["results"].pop("A"))),
    ("inner label A a string", lambda out: _mutate(out, "inner_summary.json", lambda b: b["results"].update(A="x"))),
    ("inner extra label Z", _extra_inner_label),
]


@pytest.mark.parametrize("mutate", [c[1] for c in KEY_CASES], ids=[c[0] for c in KEY_CASES])
def test_f_run_refuses_a_label_or_period_key_mismatch(tmp_path, mutate):
    _f_run(tmp_path)
    mutate(tmp_path)
    _refused(tmp_path, "tuning_receipt_changed")


def _direct(views=True):
    """In-memory results for label A in 2024, the account and every structure; beside-F views added when ``views``."""
    results = {"A": {"2024": {"account": {"total_return": 0.021, "trades": 400},
                              "structures": {n: {"total_return": 0.021, "trades": 50} for n in NAMES}}}}
    if views:
        _add_views(results, ("2024",))
    return results


def _zero(views):
    """An entry that takes no rows: every view walks none, so wanted is 0 and mean_ret is None."""
    return {"total_return": 0.0, "trades": 0, "views": {v: {"total_return": 0.0, "mean_ret": None, "wanted": 0} for v in views}}


def _zero_results():
    return {"A": {"2024": {"account": _zero(ALL_VIEWS), "structures": {n: _zero(ACCOUNT_VIEWS) for n in NAMES}}}}


def _entry(results, where):
    """The account or the first structure of label A in 2024, from in-memory results."""
    res = results["A"]["2024"]
    return res["account"] if where == "account" else res["structures"][NAMES[0]]


def test_direct_admit_accepts_views_built_in_memory():
    assert f_views.admit(_direct(), ["A"], ["2024"])


def test_admit_accepts_a_zero_trades_entry():
    assert f_views.admit(_zero_results(), ["A"], ["2024"])


@pytest.mark.parametrize("where", ["account", "structure"])
def test_admit_refuses_a_huge_int_total_return(where):
    results = _direct()
    _entry(results, where)["views"]["mid"]["total_return"] = 10**400
    assert not f_views.admit(results, ["A"], ["2024"])


@pytest.mark.parametrize("where", ["account", "structure"])
def test_admit_refuses_a_nan_total_return(where):
    results = _direct()
    _entry(results, where)["views"]["mid"]["total_return"] = float("nan")
    assert not f_views.admit(results, ["A"], ["2024"])


@pytest.mark.parametrize("where", ["account", "structure"])
def test_admit_refuses_a_mean_ret_number_with_no_trades(where):
    results = _zero_results()
    _entry(results, where)["views"]["F"]["mean_ret"] = 0.001
    assert not f_views.admit(results, ["A"], ["2024"])


@pytest.mark.parametrize("wanted", [True, 1.0])
def test_admit_refuses_a_wanted_that_is_not_a_non_bool_int(wanted):
    results = _direct()
    _entry(results, "account")["trades"] = 1
    for v in f_views.VIEWS:
        _entry(results, "account")["views"][v]["wanted"] = 1
    _entry(results, "account")["views"]["mid"]["wanted"] = wanted
    assert not f_views.admit(results, ["A"], ["2024"])


def test_admit_accepts_a_wanted_int_equal_to_one_trade():
    results = _direct()
    _entry(results, "account")["trades"] = 1
    for v in f_views.VIEWS:
        _entry(results, "account")["views"][v]["wanted"] = 1
    assert f_views.admit(results, ["A"], ["2024"])


def test_admit_refuses_a_missing_label_or_malformed_input():
    assert not f_views.admit(_direct(), ["A", "B"], ["2024"])
    assert not f_views.admit("x", ["A"], ["2024"]) and not f_views.admit(None, ["A"], ["2024"])


def test_has_views_is_false_on_the_plain_shape_and_true_with_views():
    assert not f_views.has_views(_direct(views=False))
    assert f_views.has_views(_direct())


def test_direct_admit_refuses_an_extra_label_or_period():
    results = _direct()
    results["Z"] = copy.deepcopy(results["A"])
    assert not f_views.admit(results, ["A"], ["2024"])
    results = _direct()
    results["A"]["2025"] = copy.deepcopy(results["A"]["2024"])
    assert not f_views.admit(results, ["A"], ["2024"])


def test_direct_admit_refuses_a_missing_period():
    assert not f_views.admit(_direct(), ["A"], ["2024", "2025"])
