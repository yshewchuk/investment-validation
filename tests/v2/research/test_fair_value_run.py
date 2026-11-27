"""Tests for the chunked pricing smoke (fill model F; design section 10 R2, R3, R5). Synthetic tables only."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from experiments import fair_value_fills as fvf
from experiments import fair_value_inputs as fvi
from experiments import fair_value_legs as fvl
from experiments import fair_value_run as run

STRIKES = (90.0, 100.0, 110.0)
EXPIRIES = ["2024-02-16", "2024-03-15", "2024-05-17", "2024-06-21"]
MONTHS = [(2024, m) for m in range(1, 7)]
OK_TABLE = {"files": 1, "rows": 1, "unreadable": 0, "unrecognized": 0}
SRC = Path(run.__file__).parent
ALLOWED = {"engine.fills", "experiments.fair_value_inputs", "experiments.fair_value_legs", "experiments.fair_value_fills"}
FORBIDDEN = ("debit", "proceeds", "return", "pnl", "status", "fill")
LEG_COLS = ["structure_id", "ticker", "entry_date", "exit_date", "expiry", "strike", "right", "side", "qty"]


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _chain(obs, expiries):
    return pd.DataFrame([{"ticker": "AAA", "obs_date": obs, "expiry": e, "strike": k, "right": r, "bid": 1.0,
                          "ask": 1.2, "iv": 0.3, "spot": 100.0, "quote_repaired": False, "open_interest": 50.0,
                          "bid_size": 5.0, "ask_size": 5.0, "chain_kind": "fetch"}
                         for e in expiries for k in STRIKES for r in ("C", "P")])


def _leg(sid, entry, leg_exit, expiry):
    return {"structure_id": sid, "ticker": "AAA", "entry_date": entry, "exit_date": leg_exit, "expiry": expiry,
            "strike": 100.0, "right": "C", "side": "buy", "qty": 1}


def _legs(*rows):
    return pd.DataFrame(list(rows), columns=LEG_COLS)


def _main_legs():
    # Q1 requests (01-02, 05-17) and (01-02, 02-16); Q2 requests (04-02, 05-17) (S1's exit).
    return _legs(_leg("S1", "2024-01-02", "2024-04-02", "2024-05-17"), _leg("S2", "2024-01-02", "2024-01-02", "2024-02-16"))


UNITS = {"borrow30": "percent", "borrow2yr": "percent", "divAmt": "dollars_per_share"}


def _carry(n, borrow30=10.0, borrow2yr=10.0, div_amt=0.0, div_date=None):
    """Carry columns for n raw rate-table rows: borrow in percent (the loader divides by 100), dividend in dollars."""
    return {"borrow30": [borrow30] * n, "borrow2yr": [borrow2yr] * n, "divAmt": [div_amt] * n,
            "divDate": pd.to_datetime([div_date] * n), "annIdiv": [0.0] * n, "annActDiv": [0.0] * n,
            "assetType": [1.0] * n}


@pytest.fixture
def world(tmp_path):
    root, qdir = tmp_path / "tables", tmp_path / "quotes"
    files = {}
    for y, m in MONTHS:
        files[f"quotes/year={y}/month={m:02d}.parquet"] = pd.DataFrame(
            [{"ticker": "AAA", "obs_date": pd.Timestamp(y, m, 2), "expiry": pd.Timestamp(e), "strike": k, "smvVol": 25.0}
             for e in EXPIRIES for k in STRIKES])
        files[f"rates/year={y}/month={m:02d}.parquet"] = pd.DataFrame(
            {"ticker": ["AAA"], "obs_date": [pd.Timestamp(y, m, 28)], "iRate5wk": [5.0], "iRateLt": [4.0], **_carry(1)})
    # The rate row for early-January queries sits in the previous month's file.
    files["rates/year=2023/month=12.parquet"] = pd.DataFrame(
        {"ticker": ["AAA"], "obs_date": [pd.Timestamp(2023, 12, 28)], "iRate5wk": [5.0], "iRateLt": [4.0], **_carry(1)})
    for rel, frame in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(root / rel, index=False)
    manifest = json.dumps({"version": "3", "tables": {"quotes": OK_TABLE, "rates": OK_TABLE}, "units": UNITS}).encode()
    (root / "manifest.json").write_bytes(manifest)
    qdir.mkdir()
    _chain("2024-01-02", ["2024-02-16", "2024-03-15", "2024-05-17"]).to_parquet(qdir / "2024Q1.parquet", index=False)
    _chain("2024-04-02", ["2024-05-17", "2024-06-21"]).to_parquet(qdir / "2024Q2.parquet", index=False)
    pins = {"manifest_sha256": _sha(manifest), "files": {rel: _sha((root / rel).read_bytes()) for rel in files}}
    return {"root": root, "quotes": qdir, "pins": pins}


@pytest.fixture
def spy(monkeypatch):
    calls = []
    real = fvf.leg_fills

    def wrapped(part, vols):
        out = real(part, vols)
        calls.append((part.copy(), out))
        return out

    monkeypatch.setattr(fvf, "leg_fills", wrapped)
    return calls


@pytest.fixture
def reads(monkeypatch):
    calls = []
    real = pd.read_parquet

    def wrapped(*args, **kwargs):
        calls.append(kwargs.get("columns") or [])
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", wrapped)
    return calls


def _run(w, legs=None, **over):
    kwargs = {"quotes_dir": w["quotes"], "tables_root": w["root"], "pins": w["pins"], "snapshot_id": "s",
              "expected_snapshot_id": "s", "vol_scale": 0.01, "rate_scale": 0.01, "max_rss_mb": 10_000.0, **over}
    return run.run_smoke(_main_legs() if legs is None else legs, **kwargs)


def _priced(calls):
    return pd.concat([out.set_index(fvl.QUOTE_KEY)[["V", "fill_buy", "fill_sell"]] for _, out in calls]).sort_index()


def _keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield str(key)
            yield from _keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _keys(value)


def test_fvf_id_is_stable_and_hex():
    assert run.fvf_id() == run.fvf_id()
    assert len(run.fvf_id()) == 64


@pytest.mark.parametrize("name", run.SOURCE_FILES)
def test_fvf_id_changes_when_one_source_file_changes_by_one_byte(tmp_path, name):
    for n in run.SOURCE_FILES:
        (tmp_path / n).write_bytes((SRC / n).read_bytes())
    before = run.fvf_id(root=tmp_path)
    assert before == run.fvf_id()
    data = (tmp_path / name).read_bytes()
    (tmp_path / name).write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    assert run.fvf_id(root=tmp_path) != before


def test_fvf_id_changes_with_min_meaningful_cost(monkeypatch):
    before = run.fvf_id()
    monkeypatch.setattr(run, "MIN_MEANINGFUL_COST", run.MIN_MEANINGFUL_COST + 1.0)
    assert run.fvf_id() != before


def test_fvf_id_refuses_source_file_over_read_cap(tmp_path):
    for n in run.SOURCE_FILES:
        (tmp_path / n).write_bytes((SRC / n).read_bytes())
    (tmp_path / run.SOURCE_FILES[0]).write_bytes(b"\0" * ((1 << 20) + 1))
    with pytest.raises(fvi.FairValueRefusal) as exc:
        run.fvf_id(root=tmp_path)
    assert exc.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert exc.value.details["reason"] == "input_too_large"


def test_source_set_matches_fair_value_modules():
    names = {p.name for p in SRC.glob("fair_value_*.py")}
    # The fit, constants and solve modules compute no current-F mark, so they are deliberately not in fvf_id; listed here only to match the module set.
    new = ("fair_value_fit", "fair_value_constants", "fair_value_solve")
    assert set(run.SOURCE_FILES) | {"fair_value_run.py", *(f"{m}.py" for m in new)} == names
    assert not {"fair_value_fit.py", "fair_value_constants.py", "fair_value_solve.py"} & set(run.SOURCE_FILES)
    for name in ["fair_value_run.py", *run.SOURCE_FILES]:
        text = (SRC / name).read_text(encoding="utf-8")
        assert not any(m in text for m in new), name


def test_require_fvf_id_accepts_pin_and_refuses_another():
    assert run.require_fvf_id(run.fvf_id()) == run.fvf_id()
    with pytest.raises(fvi.FairValueRefusal) as exc:
        run.require_fvf_id("0" * 64)
    assert exc.value.details["reason"] == "fvf_id_mismatch"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_two_quarter_report_counts_and_strict_json(world):
    report = _run(world)
    got = [(c["quarter"], c["rows"], c["put_rows"], c["expiries"], c["expiries_missing"], c["batches"])
           for c in report["chunks"]]
    assert got == [("2024Q1", 12, 6, 2, 0, 1), ("2024Q2", 6, 3, 1, 0, 1)]
    assert set(report["chunks"][0]) == {"quarter", "rows", "put_rows", "expiries", "expiries_missing", "batches",
                                        "load_s", "price_s", "total_s", "peak_rss_mb"}
    totals = report["totals"]
    assert len(report["chunks"]) == 2
    assert (totals["rows"], totals["put_rows"], totals["expiries"], totals["expiries_missing"]) == (18, 9, 3, 0)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert not [k for k in _keys(report) if any(word in k for word in FORBIDDEN)]


def test_early_quarter_rate_comes_from_previous_month(world, spy):
    _run(world)
    q1 = spy[0][0]
    assert q1["obs_date"].eq(pd.Timestamp("2024-01-02")).all()
    # Previous-month rate row: iRate5wk 5.0 -> 0.05, iRateLt 4.0 -> 0.04 at rate_scale 0.01; linear in days to expiry.
    days = (q1["expiry"] - pd.Timestamp("2024-01-02")).dt.days
    weight = ((days - 35) / (365 - 35)).clip(0, 1)
    assert q1["rate"].to_numpy() == pytest.approx((0.05 + weight * (0.04 - 0.05)).to_numpy())


def test_rate_table_borrow_and_dividend_reach_priced_rows(world, spy):
    # The previous-month rate row is the one Q1 queries read: borrow 3% (30d) and 4% (2y), a $0.50 dividend ex 2024-01-20.
    rel = "rates/year=2023/month=12.parquet"
    frame = pd.DataFrame({"ticker": ["AAA"], "obs_date": [pd.Timestamp(2023, 12, 28)], "iRate5wk": [5.0], "iRateLt": [4.0],
                          **_carry(1, borrow30=3.0, borrow2yr=4.0, div_amt=0.5, div_date="2024-01-20")})
    frame.to_parquet(world["root"] / rel, index=False)
    files = {**world["pins"]["files"], rel: _sha((world["root"] / rel).read_bytes())}
    _run(world, pins={**world["pins"], "files": files})
    q1 = spy[0][0]
    assert len(q1) == 12
    assert q1["borrow30"].eq(0.03).all() and q1["borrow2yr"].eq(0.04).all()
    assert q1["divAmt"].eq(0.5).all() and q1["divDate"].eq(pd.Timestamp("2024-01-20")).all()


def test_unrequested_expiries_are_not_priced(world, spy):
    _run(world)
    priced = {pd.Timestamp(e) for part, _ in spy for e in part["expiry"]}
    assert priced == {pd.Timestamp("2024-02-16"), pd.Timestamp("2024-05-17")}


def test_expiry_shared_by_entry_and_exit_is_priced_once(world, spy):
    s3 = _leg("S3", "2024-04-02", "2024-04-02", "2024-05-17")
    legs = _legs(_leg("S1", "2024-01-02", "2024-04-02", "2024-05-17"), s3, s3)
    report = _run(world, legs=legs)
    priced = pd.concat([part for part, _ in spy])
    assert not priced.duplicated(fvl.QUOTE_KEY).any()
    assert report["chunks"][1]["expiries"] == 1 and report["chunks"][1]["rows"] == 6


def test_batching_is_whole_expiry_and_independent_of_leg_order(world, spy, monkeypatch):
    with monkeypatch.context() as patched:
        patched.setattr(run, "BATCH_ROWS", 1)
        _run(world)
    assert len(spy) == 3 and [part["expiry"].nunique() for part, _ in spy] == [1, 1, 1]
    small = _priced(spy)
    spy.clear()
    report = _run(world)
    assert [c["batches"] for c in report["chunks"]] == [1, 1] and len(spy) == 2
    large = _priced(spy)
    spy.clear()
    _run(world, legs=_main_legs().iloc[::-1])
    shuffled = _priced(spy)
    pd.testing.assert_frame_equal(small, large)
    pd.testing.assert_frame_equal(large, shuffled)


def test_peak_rss_is_positive_per_chunk(world):
    report = _run(world)
    assert all(c["peak_rss_mb"] > 0 for c in report["chunks"])


def test_two_six_row_expiries_batch_by_threshold(world, spy, monkeypatch):
    monkeypatch.setattr(run, "BATCH_ROWS", 7)
    report = _run(world)
    assert report["chunks"][0]["batches"] == 1 and [len(part) for part, _ in spy] == [12, 6]
    spy.clear()
    monkeypatch.setattr(run, "BATCH_ROWS", 6)
    report = _run(world)
    assert report["chunks"][0]["batches"] == 2 and [len(part) for part, _ in spy] == [6, 6, 6]


def test_absent_expiries_are_counted_and_empty_quarters_have_no_batches(world):
    legs = _legs(_leg("S1", "2024-01-02", "2024-04-02", "2024-05-17"), _leg("S2", "2024-01-02", "2024-01-02", "2024-07-19"),
                 _leg("S3", "2024-04-02", "2024-04-02", "2024-07-19"))
    report = _run(world, legs=legs)
    got = [(c["quarter"], c["rows"], c["batches"], c["expiries"], c["expiries_missing"]) for c in report["chunks"]]
    assert got == [("2024Q1", 6, 1, 2, 1), ("2024Q2", 6, 1, 2, 1)]
    empty = _legs(_leg("S1", "2024-01-02", "2024-01-02", "2024-07-19"), _leg("S2", "2024-04-02", "2024-04-02", "2024-07-19"))
    report = _run(world, legs=empty)
    got = [(c["quarter"], c["rows"], c["batches"], c["expiries_missing"]) for c in report["chunks"]]
    assert got == [("2024Q1", 0, 0, 1), ("2024Q2", 0, 0, 1)]


@pytest.mark.parametrize("make", [
    lambda legs: legs.drop(columns=["exit_date"]),
    lambda legs: legs.iloc[:0],
    lambda legs: legs.assign(expiry=legs["expiry"].mask(legs.index == 0)),
    lambda legs: _legs(_leg("S1", "2024-04-02", "2024-01-02", "2024-05-17")),
], ids=["missing-column", "empty", "null-expiry", "exit-before-entry"])
def test_legs_invalid_refused(make):
    with pytest.raises(fvi.FairValueRefusal) as exc:
        run.requested_expiries(make(_main_legs()))
    assert exc.value.details["reason"] == "legs_invalid"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_missing_second_quarter_file_refused_before_any_pricing(world, spy):
    (world["quotes"] / "2024Q2.parquet").unlink()
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world)
    assert exc.value.details["reason"] == "quotes_file_absent"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert spy == []


def test_snapshot_mismatch_refused(world):
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world, expected_snapshot_id="other")
    assert exc.value.details["reason"] == "snapshot_binding"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_fvf_id_mismatch_refused_before_any_file_read(tmp_path):
    absent = tmp_path / "absent"
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run({"quotes": absent, "root": absent, "pins": {}}, expected_fvf_id="0" * 64)
    assert exc.value.details["reason"] == "fvf_id_mismatch"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_pins_without_manifest_refused_before_any_file_read(tmp_path):
    absent = tmp_path / "absent"
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run({"quotes": absent, "root": absent, "pins": {}})
    assert exc.value.details["reason"] == "pins_invalid"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("over,reason", [({"max_rss_mb": 0.0}, "rss_limit"),
                                         ({"max_rss_mb": float("nan")}, "rss_limit")])
def test_bad_limits_refused_before_any_file_read(tmp_path, over, reason):
    absent = tmp_path / "absent"
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run({"quotes": absent, "root": absent, "pins": {}}, **over)
    assert exc.value.details["reason"] == reason
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_limit_bounds_the_tables_with_resource_code(world, spy):
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world, max_rss_mb=0.001)
    assert exc.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert spy == []


def test_rss_over_limit_after_load_refused_before_pricing(world, spy, monkeypatch):
    monkeypatch.setattr(run, "_peak_mb", lambda: 1e9)
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world)
    assert exc.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert exc.value.details["reason"] == "rss_over_limit"
    assert spy == []


def test_rss_over_limit_after_first_batch_stops_pricing(world, spy, monkeypatch):
    readings = iter([1.0, 1e9])
    monkeypatch.setattr(run, "_peak_mb", lambda: next(readings))
    monkeypatch.setattr(run, "BATCH_ROWS", 1)
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world)
    assert exc.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert exc.value.details["reason"] == "rss_over_limit"
    assert len(spy) == 1


def test_quote_file_over_limit_refused_before_any_decode(world, spy, reads):
    (world["quotes"] / "2024Q1.parquet").write_bytes(b"\0" * (2**20 + 1))
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world, max_rss_mb=1.0)
    assert exc.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert exc.value.details["reason"] == "input_too_large"
    assert not any("bid" in cols for cols in reads)
    assert spy == []


def test_quote_footer_estimate_over_limit_refused_before_decode(world, spy, reads, monkeypatch):
    fake = SimpleNamespace(num_rows=10**12, num_row_groups=0, schema=SimpleNamespace(names=[]))
    monkeypatch.setattr(run.pyarrow.parquet, "read_metadata", lambda _source: fake)
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world)
    assert exc.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert exc.value.details["reason"] == "input_too_large"
    assert not any("bid" in cols for cols in reads)
    assert spy == []


def test_quote_read_takes_only_contract_columns(world, spy, monkeypatch):
    quote = world["quotes"] / "2024Q1.parquet"
    _chain("2024-01-02", ["2024-02-16", "2024-03-15", "2024-05-17"]).assign(junk=1.0).to_parquet(quote, index=False)
    blob, seen, real = quote.read_bytes(), [], pd.read_parquet

    def wrapped(source, *args, **kwargs):
        if isinstance(source, io.BytesIO) and source.getvalue() == blob:  # only quote-file reads, not table reads
            seen.append(kwargs.get("columns"))
        return real(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", wrapped)
    _run(world)
    assert seen == [list(fvi.QUOTE_COLUMNS)]
    assert "junk" not in spy[0][0].columns


def test_quote_file_without_iv_refused_required_column(world, spy):
    _chain("2024-01-02", ["2024-02-16", "2024-03-15", "2024-05-17"]).drop(columns=["iv"]).to_parquet(
        world["quotes"] / "2024Q1.parquet", index=False)
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.details["reason"] == "required_column"
    assert spy == []


def test_wrong_table_pin_refused(world):
    files = {**world["pins"]["files"], "quotes/year=2024/month=01.parquet": "0" * 64}
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world, pins={**world["pins"], "files": files})
    assert exc.value.details["reason"] == "table_hash"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def _pins_without(w, *gone):
    return {**w["pins"], "files": {k: v for k, v in w["pins"]["files"].items() if k not in gone}}


def test_missing_quotes_month_refused_with_path(world):
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world, pins=_pins_without(world, "quotes/year=2024/month=01.parquet"))
    assert exc.value.details["reason"] == "table_absent"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.details["paths"] == ["quotes/year=2024/month=01.parquet"]


def test_missing_previous_month_rate_refused_with_path(world):
    # obs 2024-01-02 minus the 7-day rate lookback reaches December 2023.
    with pytest.raises(fvi.FairValueRefusal) as exc:
        _run(world, pins=_pins_without(world, "rates/year=2023/month=12.parquet"))
    assert exc.value.details["reason"] == "table_absent"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.details["paths"] == ["rates/year=2023/month=12.parquet"]


def test_run_succeeds_with_only_the_months_requested_dates_need(world):
    needed = ("quotes/year=2024/month=01.parquet", "quotes/year=2024/month=04.parquet",
              "rates/year=2023/month=12.parquet", "rates/year=2024/month=01.parquet",
              "rates/year=2024/month=03.parquet", "rates/year=2024/month=04.parquet")
    pins = _pins_without(world, *[k for k in world["pins"]["files"] if k not in needed])
    report = _run(world, pins=pins)
    assert [(c["quarter"], c["rows"]) for c in report["chunks"]] == [("2024Q1", 12), ("2024Q2", 6)]


def _cli(w, tmp_path, *extra):
    legs, pins = tmp_path / "legs.csv", tmp_path / "pins.json"
    _main_legs().to_csv(legs, index=False)
    pins.write_text(json.dumps(w["pins"]), encoding="utf-8")
    return ["--legs", str(legs), "--quotes-dir", str(w["quotes"]), "--tables-root", str(w["root"]),
            "--pins", str(pins), "--snapshot-id", "s", "--expected-snapshot-id", "s", "--vol-scale", "0.01",
            "--rate-scale", "0.01", "--max-rss-mb", "10000", *extra]


def test_main_success_prints_exactly_one_json_line(world, tmp_path, capsys):
    assert run.main(_cli(world, tmp_path)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and len(json.loads(lines[0])["chunks"]) == 2


def test_main_refusal_prints_one_refusal_line(world, tmp_path, capsys):
    assert run.main(_cli(world, tmp_path, "--expected-fvf-id", "0" * 64)) == 2
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    out = json.loads(lines[0])
    assert out["refusal"] == "EXPERIMENT_IDENTITY_CONFLICT" and out["details"]["reason"] == "fvf_id_mismatch"


def test_main_pins_over_cap_refused_with_one_line(world, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(fvi, "MAX_MANIFEST_BYTES", 1)
    assert run.main(_cli(world, tmp_path)) == 2
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    out = json.loads(lines[0])
    assert out["refusal"] == "RESOURCE_LIMIT_EXCEEDED" and out["details"]["reason"] == "input_too_large"


def test_main_legs_over_cap_refused_with_one_line(world, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(fvi, "MAX_TABLE_BYTES", 1)
    assert run.main(_cli(world, tmp_path)) == 2
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    out = json.loads(lines[0])
    assert out["refusal"] == "RESOURCE_LIMIT_EXCEEDED" and out["details"]["reason"] == "input_too_large"


def _mutate(where, data):
    """Overwrite the named input with `data`; `data` None removes it."""
    def mutate(w, tmp):
        path = {"pins": tmp / "pins.json", "legs": tmp / "legs.csv", "quote": w["quotes"] / "2024Q1.parquet"}[where]
        if data is None:
            path.unlink()
        else:
            path.write_bytes(data)
    return mutate


MALFORMED = [
    pytest.param(_mutate("pins", b"{not json"), "input_unreadable", id="pins-not-json"),
    pytest.param(_mutate("pins", None), "input_unreadable", id="pins-missing"),
    pytest.param(_mutate("pins", b"[]"), "pins_invalid", id="pins-list"),
    pytest.param(_mutate("pins", b"{}"), "pins_invalid", id="pins-empty-object"),
    pytest.param(_mutate("pins", b'{"manifest_sha256": 1, "files": {}}'), "pins_invalid", id="pins-sha-not-str"),
    pytest.param(_mutate("pins", b'{"manifest_sha256": "a", "files": []}'), "pins_invalid", id="pins-files-list"),
    pytest.param(_mutate("legs", b""), "input_unreadable", id="legs-empty"),
    pytest.param(_mutate("legs", None), "input_unreadable", id="legs-missing"),
    pytest.param(_mutate("quote", b"not parquet at all"), "input_unreadable", id="quote-garbage"),
]


@pytest.mark.parametrize("mutate,reason", MALFORMED)
def test_main_malformed_input_refused_with_one_line(world, tmp_path, capsys, mutate, reason):
    argv = _cli(world, tmp_path)
    mutate(world, tmp_path)
    assert run.main(argv) == 2
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    out = json.loads(lines[0])
    assert out["refusal"] == "EXPERIMENT_IDENTITY_CONFLICT" and out["details"]["reason"] == reason


@pytest.mark.parametrize("error", [pa.ArrowTypeError, pa.ArrowNotImplementedError])
def test_main_arrow_read_error_refused_with_one_line(world, tmp_path, capsys, monkeypatch, error):
    def raise_arrow(*_args, **_kwargs):
        raise error("unreadable quote footer")
    monkeypatch.setattr(run.pyarrow.parquet, "read_metadata", raise_arrow)
    assert run.main(_cli(world, tmp_path)) == 2
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    out = json.loads(lines[0])
    assert out["refusal"] == "EXPERIMENT_IDENTITY_CONFLICT" and out["details"]["reason"] == "input_unreadable"


def test_module_imports_only_allowed_modules():
    tree = ast.parse((SRC / "fair_value_run.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module == "experiments":
            names |= {f"experiments.{alias.name}" for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module)
    for name in names:
        top = name.split(".")[0]
        assert name in ALLOWED or top in sys.stdlib_module_names or top in {"pandas", "numpy", "pyarrow"}, name


def test_run_creates_no_file_under_tmp_path(world, tmp_path):
    before = sorted(tmp_path.rglob("*"))
    _run(world)
    assert sorted(tmp_path.rglob("*")) == before
