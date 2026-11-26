"""EXP-189 Amendment 1, slice W7c1: F-run report labels, beside-F views and refusals (synthetic receipts only, no market data)."""
import json

import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from experiments import size_pnl_receipts as receipts
from experiments import size_width_f_views as f_views
from experiments import size_width_report as report
from experiments import size_width_tuning as tuning
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_account import simulate_account
from test_size_width_report import _write

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
FILL = {"fill_model_id": "F", "quotes": "synthetic"}
LABELS = ("production", "A", "B", "C", "D", "E")
HALVES = ("2023H2", "2024")
SHIFT = {"2023H2": 0.0, "2024": 0.001, "2025": 0.002}
STEP = {"F": 0.0, "mid": 0.001, "harsh": 0.002, "unknown0": 0.003, "unknown1": 0.004, "midfloor": 0.005}
BASE25 = {"production": 0.012, "A": 0.022, "B": 0.007, "C": 0.007, "D": 0.007, "E": 0.007}
BOUND = {"inputs_inner.json": ("inner_summary.json", "inputs_inner"), "inputs_outer.json": ("outer_summary.json", "inputs_outer")}
WINDOW = (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"))
BY = {n: 1.0 for n in NAMES}


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
    for label in LABELS:
        for p in periods:
            res = results[label][p]
            ret = res["account"]["total_return"] + SHIFT[p]
            res["account"]["total_return"] = ret
            for i, n in enumerate(NAMES):
                res["structures"][n]["total_return"] = ret + 0.0001 * i


def _block(results, periods):
    """Beside-F views of every label: F equal to the primary results, every other view F plus its step."""
    out = {}
    for label in LABELS:
        out[label] = {}
        for p in periods:
            res = results[label][p]
            cells = {}
            for v in f_views.VIEWS:
                acc = res["account"]["total_return"] + STEP[v]
                cell = {"trades": res["account"]["trades"], "return_per_trade": acc - 0.001, "account_return": acc}
                if v in f_views.ACCOUNT_VIEWS:
                    cell["structures"] = {n: res["structures"][n]["total_return"] + STEP[v] for n in NAMES}
                cells[v] = cell
            out[label][p] = cells
    return out


def _f_run(out, fill=FILL):
    """A synthetic F run: _write's receipts retuned per period, the F binding in both inputs, beside-F views in both summaries."""
    _write(out, passing=("A",))
    for name in ("inputs_inner.json", "inputs_outer.json"):
        _mutate(out, name, lambda b: b.update(fill_model=fill))
    _mutate(out, "inner_summary.json", lambda b: (_retune(b["results"], HALVES), b.update(f_views=_block(b["results"], HALVES))))
    _mutate(out, "outer_summary.json", lambda b: (_retune(b["results"], ("2025",)), b.update(f_views=_block(b["results"], ("2025",)))))


def _text(out):
    return (out / "REPORT.md").read_text(encoding="utf-8")


def _section(text, start):
    """The section that starts with ``start``, up to the next heading."""
    return text.split(start, 1)[1].split("\n### ")[0]


def _refused(out, reason):
    """assemble refuses as an identity conflict under ``reason`` and writes neither REPORT.md nor proposed_rule.json."""
    with pytest.raises(Refusal) as caught:
        report.assemble(out)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == reason
    assert not (out / "REPORT.md").exists() and not (out / "proposed_rule.json").exists()


def _f_cell(b):
    return b["f_views"]["A"]["2024"]["F"]


def _nan_view(out):
    """An inner summary whose mid account return is NaN: receipts cannot carry NaN, so the file is unreadable."""
    path = out / "receipts" / "inner_summary.json"
    body = _body(path)
    body["f_views"]["A"]["2024"]["mid"]["account_return"] = float("nan")
    path.unlink()
    path.write_text(json.dumps({**body, "content_hash": "0" * 64}, allow_nan=True), encoding="utf-8")


def test_f_run_labels_headings_and_tables(tmp_path):
    _f_run(tmp_path)
    report.assemble(tmp_path)
    text = _text(tmp_path)
    assert "POST-HOC (Amendment 1, F fills): chosen after the full run under mid fills had been seen" in text
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings and all(line.endswith(f"[{f_views.LABEL}]") for line in headings)
    assert "at marks_mid" not in text
    assert "### T3 curves per structure (2023-24, account return % at F fills; 2025 NOT blind for widths)" in text
    assert json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))["label"] == f_views.LABEL

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

    for view, step in (("mid", STEP["mid"]), ("harsh", STEP["harsh"])):
        section = _section(text, f"### T6 beside F: per-structure stand-alone accounts at {view} ")
        assert (f"### T6 beside F: per-structure stand-alone accounts at {view} (2025, return %; reported, never deciding; "
                f"2025 NOT blind for widths) [{f_views.LABEL}]") in text
        assert "| structure | production | A | B | C | D | E |" in section
        for i, n in enumerate(NAMES):
            row = f"| {n} | " + " | ".join(f"{100 * (BASE25[label] + step + 0.0001 * i):.2f}" for label in LABELS) + " |"
            assert row in section


def test_non_f_run_is_unlabelled(tmp_path):
    _write(tmp_path, passing=("A",))
    report.assemble(tmp_path)
    assert "POST-HOC" not in _text(tmp_path)
    assert "at marks_mid" in _text(tmp_path)
    rule = json.loads((tmp_path / "proposed_rule.json").read_text(encoding="utf-8"))
    assert "label" not in rule and "POST-HOC" not in json.dumps(rule)


CASES = [
    ("f_views missing in inner", _edit("inner_summary.json", lambda b: b.pop("f_views")), "tuning_receipt_changed"),
    ("f_views missing in outer", _edit("outer_summary.json", lambda b: b.pop("f_views")), "tuning_receipt_changed"),
    ("a missing view", _edit("inner_summary.json", lambda b: b["f_views"]["A"]["2024"].pop("midfloor")), "tuning_receipt_changed"),
    ("non-finite account_return", _nan_view, "tuning_receipt_unreadable"),
    ("unequal trades across views", _edit("inner_summary.json", lambda b: b["f_views"]["A"]["2024"]["mid"].update(trades=399)),
     "tuning_receipt_changed"),
    ("F account_return differing from results",
     _edit("inner_summary.json", lambda b: _f_cell(b).update(account_return=_f_cell(b)["account_return"] + 0.001)), "tuning_receipt_changed"),
    ("F structure return differing from results",
     _edit("inner_summary.json", lambda b: _f_cell(b)["structures"].update({NAMES[0]: _f_cell(b)["structures"][NAMES[0]] + 0.001})),
     "tuning_receipt_changed"),
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


def test_f_views_in_a_non_f_run_refuse(tmp_path):
    _write(tmp_path, passing=("A",))
    _mutate(tmp_path, "inner_summary.json", lambda b: b.update(f_views={}))
    _refused(tmp_path, "tuning_receipt_changed")


class FakeSweep:
    """An F sweep stand-in: ``f`` and ``rows(pairs, window)`` over taken frames keyed by (structure, arm)."""

    def __init__(self, taken, f=True):
        self.taken, self.f = taken, f

    def rows(self, pairs, window):
        frames = [self.taken[(n, arm)] for n, arm in pairs if (n, arm) in self.taken]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)


# view -> (debits, exits) for two trades in 2024; every view's debit and exit differ from every other view's
PAIRS = {"F": ([1.00, 2.00], [1.20, 2.20]), "mid": ([1.10, 1.90], [1.25, 2.09]), "harsh": ([1.30, 2.20], [1.04, 2.42]),
         "unknown0": ([0.90, 1.00], [1.00, 0.80]), "unknown1": ([1.50, 2.50], [1.80, 2.70]), "midfloor": ([2.00, 1.20], [2.40, 1.14])}
ENTRY, EXIT = ["2024-03-01", "2024-04-01"], ["2024-03-10", "2024-04-15"]
EXPECTED_PER_TRADE = {"F": (0.20 + 0.10) / 2, "mid": (0.15 / 1.10 + 0.19 / 1.90) / 2, "harsh": (-0.26 / 1.30 + 0.22 / 2.20) / 2,
                      "unknown0": (0.10 / 0.90 - 0.20) / 2, "unknown1": (0.20 + 0.08) / 2, "midfloor": (0.20 - 0.05) / 2}


def _taken(mid_as_f=False):
    """The two taken F trades: every view's own debit and exit; ``mid_as_f`` sets the mid exit to the F exit (for evaluate's mid walk)."""
    frame = pd.DataFrame({"entry_date": pd.to_datetime(ENTRY), "exit_date": pd.to_datetime(EXIT), "secured_per_contract": [5000.0, 5000.0]})
    for v, (debit, exit_) in PAIRS.items():
        cost_col, exit_col = ("entry_cost", "exit_value") if v == "F" else (f"entry_cost_{v}", f"exit_value_{v}")
        frame[cost_col], frame[exit_col] = debit, exit_
    if mid_as_f:
        frame["exit_value_mid"] = frame["exit_value"]
    return frame


COLS = list(_taken().columns)


def _expected_frame(view):
    debit, exit_ = PAIRS[view]
    return pd.DataFrame({"entry_date": pd.to_datetime(ENTRY), "exit_date": pd.to_datetime(EXIT), "secured_per_contract": [5000.0, 5000.0],
                         "entry_cost": debit, "exit_value": exit_})


def _views(sweep):
    return f_views.view_tables(sweep, {"A": BY}, {"2024": WINDOW})


def _results():
    """The primary results a round trip compares with: evaluate on the sweep whose mid exit is the F exit."""
    return {"A": {"2024": tuning.evaluate(FakeSweep({(NAMES[0], 1.0): _taken(mid_as_f=True)}), BY, WINDOW)}}


def test_each_view_uses_its_own_debit_and_exit():
    got = _views(FakeSweep({(NAMES[0], 1.0): _taken()}))["A"]["2024"]
    for v in f_views.VIEWS:
        assert got[v]["trades"] == 2
        assert got[v]["return_per_trade"] == pytest.approx(EXPECTED_PER_TRADE[v])
        assert got[v]["account_return"] == pytest.approx(simulate_account(_expected_frame(v), "primary")["total_return"])
    assert set(got["F"]["structures"]) == set(NAMES)
    assert got["F"]["structures"][NAMES[0]] == pytest.approx(got["F"]["account_return"])
    assert all(got["F"]["structures"][n] == 0.0 for n in NAMES[1:])
    assert "structures" not in got["unknown0"]


def test_no_trades_gives_no_per_trade_and_a_zero_account():
    got = _views(FakeSweep({}))["A"]["2024"]
    for v in f_views.VIEWS:
        assert got[v]["trades"] == 0 and got[v]["return_per_trade"] is None and got[v]["account_return"] == 0.0


@pytest.mark.parametrize("column, value", [("entry_cost_harsh", 0.0), ("entry_cost_unknown1", float("nan")), ("entry_cost", -1.0),
                                           ("entry_cost_midfloor", float("inf"))])
def test_a_view_debit_that_is_not_finite_and_positive_refuses(column, value):
    taken = _taken()
    taken.loc[0, column] = value
    with pytest.raises(ValueError):
        _views(FakeSweep({(NAMES[0], 1.0): taken}))


def test_a_non_f_sweep_refuses():
    with pytest.raises(ValueError):
        _views(FakeSweep({(NAMES[0], 1.0): _taken()}, f=False))


def test_round_trip_admits_views_built_from_evaluate():
    block = _views(FakeSweep({(NAMES[0], 1.0): _taken()}))
    assert f_views.admit(block, _results(), ["A"], ["2024"])


def test_admit_refuses_disagreeing_malformed_and_non_finite_blocks():
    block = _views(FakeSweep({(NAMES[0], 1.0): _taken()}))
    res = _results()
    res["A"]["2024"]["account"]["total_return"] += 0.001
    assert not f_views.admit(block, res, ["A"], ["2024"])
    res = _results()
    res["A"]["2024"]["structures"][NAMES[0]]["total_return"] += 0.001
    assert not f_views.admit(block, res, ["A"], ["2024"])
    block["A"]["2024"]["mid"]["trades"] = 1
    assert not f_views.admit(block, _results(), ["A"], ["2024"])
    block = _views(FakeSweep({(NAMES[0], 1.0): _taken()}))
    block["A"]["2024"]["mid"]["account_return"] = float("nan")
    assert not f_views.admit(block, _results(), ["A"], ["2024"])
    assert not f_views.admit(_views(FakeSweep({(NAMES[0], 1.0): _taken()})), _results(), ["A", "B"], ["2024"])
    assert not f_views.admit("x", _results(), ["A"], ["2024"]) and not f_views.admit(None, {}, ["A"], ["2024"])
