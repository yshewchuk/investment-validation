"""Tests for the fair-value input layer (fill model F1; design sections 3, 10, 13)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments import fair_value_inputs as fvi
from experiments.fair_value_inputs import FairValueRefusal

QUOTE_REL = "quotes/year=2024/month=01.parquet"
RATE_REL = "rates/year=2024/month=01.parquet"
NAN = float("nan")
ABSENT = object()
OK_Q = {"files": 1, "rows": 4, "unreadable": 0, "unrecognized": 0}
OK_R = {"files": 1, "rows": 2, "unreadable": 0, "unrecognized": 0, "extra_key": 7}

QUOTES = pd.DataFrame({
    "ticker": ["AAA"] * 4,
    "obs_date": pd.to_datetime(["2024-01-02"] * 4),
    "expiry": pd.to_datetime(["2024-02-16"] * 4),
    "strike": [100.0, 110.0, 90.0, 95.0],
    "smvVol": [25.0, 0.0, -1.0, NAN],
    "callMidIv": [0.3] * 4,
    "putMidIv": [0.3] * 4,
    "stockPrice": [100.0] * 4,
})
RATES = pd.DataFrame({
    "ticker": ["AAA", "BBB"],
    "obs_date": pd.to_datetime(["2024-01-02"] * 2),
    "iRate5wk": [5.3, 30.0],
    "iRateLt": [4.1, 4.1],
    "borrow30": [0.2, 0.2],
    "annIdiv": [0.0, 0.0],
    "divDate": ["", ""],
    "assetType": ["stock", "stock"],
})
# Lookup fixture: queries at obs 2024-01-10 (see LOOKUP_CASES).
LOOKUP_RATES = pd.DataFrame({
    "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "GGG", "HHH"],
    "obs_date": pd.to_datetime(["2024-01-10", "2024-01-10", "2024-01-10", "2024-01-03", "2024-01-02",
                                "2024-01-10", "2024-01-03", "2024-01-10", "2024-01-12"]),
    "iRate5wk": [5.0, 30.0, 30.0, 5.0, 5.0, 5.0, 5.0, NAN, 5.0],
    "iRateLt": [4.0, 4.0, -9.0, 4.0, 4.0, NAN, 4.0, NAN, 4.0],
    "borrow30": 0.1, "annIdiv": 0.0, "divDate": "x", "assetType": "stock",
})


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _mani(q=OK_Q, r=OK_R):
    tables = {}
    if q is not None:
        tables["quotes"] = q
    if r is not None:
        tables["rates"] = r
    return {"version": "3", "tables": tables, "dividend_fields": {}, "extra_top": 1}


def _setup(tmp_path, files=None, manifest=None):
    """Write the tool layout; return (root, file pins, manifest pin). manifest: None=default, ABSENT=not written."""
    written = {QUOTE_REL: QUOTES, RATE_REL: RATES, **(files or {})}
    root = tmp_path / "root"
    for rel, obj in written.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, bytes):
            path.write_bytes(obj)
        else:
            obj.to_parquet(path, index=False)
    manifest_sha = "0" * 64
    if manifest is not ABSENT:
        blob = manifest if isinstance(manifest, bytes) else json.dumps(_mani() if manifest is None else manifest).encode()
        (root / "manifest.json").write_bytes(blob)
        manifest_sha = _sha(blob)
    return root, {rel: _sha((root / rel).read_bytes()) for rel in written}, manifest_sha


def _load(root, file_sha, manifest_sha, **scales):
    args = {"vol_scale": 0.01, "rate_scale": 0.01, **scales}
    return fvi.load_tables(root, manifest_sha256=manifest_sha, file_sha256=file_sha, **args)


def _refused(exc, reason, code="EXPERIMENT_IDENTITY_CONFLICT"):
    assert exc.value.code == code
    assert exc.value.details["reason"] == reason


def _plan(root, rels):
    """(rel, running total before, footer estimate, measured decoded size) per table, as load_tables admits them."""
    rows, running = [], 0
    for rel in rels:
        blob, kind = (root / rel).read_bytes(), rel.split("/")[0]
        estimate = fvi._estimate(blob, kind, rel)
        measured = int(fvi._parse(blob, kind, rel).memory_usage(index=True, deep=True).sum())
        rows.append((rel, running, estimate, measured))
        running += measured
    return rows


def _parquet_bytes(tmp_path, frame, **kwargs):
    path = tmp_path / "scratch.parquet"
    frame.to_parquet(path, index=False, **kwargs)
    return path.read_bytes()


def _no_decode(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("pd.read_parquet must not run for a refused table")
    monkeypatch.setattr(pd, "read_parquet", boom)


def test_happy_load_selects_columns_scales_and_hashes(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path)
    tables = _load(root, file_sha, manifest_sha)
    assert list(tables.vols.columns) == fvi.VOL_KEYS + ["smvVol"]
    assert list(tables.rates.columns) == fvi.RATE_KEYS + fvi.RATE_COLUMNS
    vols = tables.vols.set_index("strike")["smvVol"]
    assert vols[100.0] == pytest.approx(0.25)
    assert vols[[110.0, 90.0, 95.0]].isna().all()
    rates = tables.rates.set_index("ticker")
    assert rates.loc["AAA", "iRate5wk"] == pytest.approx(0.053)
    assert rates.loc["AAA", "iRateLt"] == pytest.approx(0.041)
    assert np.isnan(rates.loc["BBB", "iRate5wk"])
    assert set(tables.hashes) == set(file_sha) | {"manifest.json"}
    assert tables.hashes[QUOTE_REL] == file_sha[QUOTE_REL]


def test_happy_load_detects_swapped_scales(tmp_path):
    # vols are percent (vol_scale 0.01); rates are basis points (rate_scale 0.0001). Swapping the scales breaks the vol median.
    rates_bps = RATES.assign(iRate5wk=[530.0, 3000.0], iRateLt=[410.0, 410.0])
    root, file_sha, manifest_sha = _setup(tmp_path, files={RATE_REL: rates_bps})
    tables = _load(root, file_sha, manifest_sha, vol_scale=0.01, rate_scale=0.0001)
    assert tables.vols.set_index("strike")["smvVol"][100.0] == pytest.approx(0.25)
    rates = tables.rates.set_index("ticker")
    assert rates.loc["AAA", "iRate5wk"] == pytest.approx(0.053)
    assert rates.loc["AAA", "iRateLt"] == pytest.approx(0.041)
    assert np.isnan(rates.loc["BBB", "iRate5wk"])


def test_missing_scale_argument_is_a_type_error(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path)
    with pytest.raises(TypeError):
        fvi.load_tables(root, manifest_sha256=manifest_sha, file_sha256=file_sha, rate_scale=0.01)


@pytest.mark.parametrize("name", ["vol_scale", "rate_scale"])
@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_bad_unit_scale_refused(tmp_path, name, bad):
    root, file_sha, manifest_sha = _setup(tmp_path)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha, **{name: bad})
    _refused(exc, "unit_scale")


MANIFEST_CASES = [
    (ABSENT, "manifest_absent"),
    (b"{not json", "manifest_invalid"),
    (b"[]", "manifest_invalid"),
    (_mani(q={**OK_Q, "unreadable": 1}), "manifest_unread_payloads"),
    (_mani(r={**OK_R, "unrecognized": 2}), "manifest_unread_payloads"),
    (_mani(q={"rows": 4, "unrecognized": 0}), "manifest_unread_payloads"),
    (_mani(r=None), "manifest_unread_payloads"),
    (_mani(q={**OK_Q, "unreadable": False}), "manifest_unread_payloads"),
    (_mani(q={**OK_Q, "unrecognized": 0.0}), "manifest_unread_payloads"),
]
MANIFEST_IDS = ["absent", "not-json", "not-object", "quotes-unreadable", "rates-unrecognized",
                "quotes-no-key", "rates-missing", "bool-count", "float-count"]


@pytest.mark.parametrize("manifest,reason", MANIFEST_CASES, ids=MANIFEST_IDS)
def test_manifest_refusals(tmp_path, manifest, reason):
    root, file_sha, manifest_sha = _setup(tmp_path, manifest=manifest)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, reason)


def test_manifest_hash_mismatch_refused(tmp_path):
    root, file_sha, _ = _setup(tmp_path)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, "0" * 64)
    _refused(exc, "manifest_hash")


def test_manifest_read_cap_refused(tmp_path, monkeypatch):
    root, file_sha, manifest_sha = _setup(tmp_path)
    monkeypatch.setattr(fvi, "MAX_MANIFEST_BYTES", 10)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "input_too_large", code="RESOURCE_LIMIT_EXCEEDED")
    assert exc.value.details["path"] == "manifest.json"


@pytest.mark.parametrize("bad", ["../x", "quotes/year=2024/month=1.parquet", "other/year=2024/month=01.parquet"])
def test_bad_table_path_refused(tmp_path, bad):
    root, file_sha, manifest_sha = _setup(tmp_path)
    file_sha[bad] = "0" * 64
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "table_path")


def test_only_quotes_bound_is_table_absent(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, {QUOTE_REL: file_sha[QUOTE_REL]}, manifest_sha)
    _refused(exc, "table_absent")


def test_listed_file_absent_is_table_absent(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path)
    file_sha["quotes/year=2024/month=02.parquet"] = "0" * 64
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "table_absent")


def test_table_hash_mismatch_refused(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path)
    file_sha[QUOTE_REL] = "0" * 64
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "table_hash")


def test_missing_required_column_refused(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path, files={QUOTE_REL: QUOTES.drop(columns=["smvVol"])})
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "required_column")
    assert "smvVol" in exc.value.details["columns"]


def test_non_parquet_bytes_with_matching_hash_unreadable(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path, files={QUOTE_REL: b"not parquet at all"})
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "table_unreadable")


def test_duplicate_vol_key_across_months_refused(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path, files={"quotes/year=2024/month=02.parquet": QUOTES})
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "duplicate_key")
    assert exc.value.details["table"] == "quotes"


def test_duplicate_rate_key_across_months_refused(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path, files={"rates/year=2024/month=02.parquet": RATES})
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "duplicate_key")
    assert exc.value.details["table"] == "rates"


@pytest.mark.parametrize("vols,scale", [([25.0] * 4, 1.0), ([5.0] * 4, 1.0), ([0.05] * 4, 1.0),
                                        ([0.0, -1.0, NAN, 0.0], 0.01)])
def test_vol_unit_refused(tmp_path, vols, scale):
    root, file_sha, manifest_sha = _setup(tmp_path, files={QUOTE_REL: QUOTES.assign(smvVol=vols)})
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha, vol_scale=scale)
    _refused(exc, "vol_unit")


def test_oversize_table_refused(tmp_path, monkeypatch):
    root, file_sha, manifest_sha = _setup(tmp_path)
    monkeypatch.setattr(fvi, "MAX_TABLE_BYTES", 64)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "input_too_large", code="RESOURCE_LIMIT_EXCEEDED")


Q2_REL = "quotes/year=2024/month=02.parquet"


def test_aggregate_budget_refuses_when_quote_files_sum_over(tmp_path):
    # Each quote file is admitted on its own; the running total crosses at the second file.
    q2 = QUOTES.assign(obs_date=pd.to_datetime(["2024-01-03"] * 4))
    root, file_sha, manifest_sha = _setup(tmp_path, files={Q2_REL: q2})
    (_, _, e1, m1), (_, before2, e2, m2), _ = _plan(root, [QUOTE_REL, Q2_REL, RATE_REL])
    budget = before2 + max(e2, m2) - 1  # just below the second file's admitted size
    assert max(e1, m1) <= budget
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha, max_total_bytes=budget)
    _refused(exc, "input_too_large", code="RESOURCE_LIMIT_EXCEEDED")
    assert exc.value.details["path"] == Q2_REL
    assert exc.value.details["total_bytes"] == before2 + max(e2, m2)
    assert exc.value.details["max_total_bytes"] == budget


def test_aggregate_budget_exact_total_accepted_and_rates_count(tmp_path):
    q2 = QUOTES.assign(obs_date=pd.to_datetime(["2024-01-03"] * 4))
    root, file_sha, manifest_sha = _setup(tmp_path, files={Q2_REL: q2})
    plan = _plan(root, [QUOTE_REL, Q2_REL, RATE_REL])
    # Smallest budget that admits every table: each table needs running total + max(estimate, measured).
    needed = max(before + max(est, meas) for _, before, est, meas in plan)
    tables = _load(root, file_sha, manifest_sha, max_total_bytes=needed)
    assert len(tables.vols) == 8
    # The rates file counts: the budget exceeds what the two quote files alone add up to.
    assert needed > sum(meas for _, _, _, meas in plan[:2])
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha, max_total_bytes=needed - 1)
    _refused(exc, "input_too_large", code="RESOURCE_LIMIT_EXCEEDED")
    crossing = next(rel for rel, before, est, meas in plan if before + max(est, meas) == needed)
    assert exc.value.details["path"] == crossing
    assert exc.value.details["total_bytes"] == needed


def test_over_budget_by_estimate_refused_before_decode(tmp_path, monkeypatch):
    root, file_sha, manifest_sha = _setup(tmp_path)
    _no_decode(monkeypatch)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha, max_total_bytes=1)
    _refused(exc, "input_too_large", code="RESOURCE_LIMIT_EXCEEDED")
    assert exc.value.details["path"] == QUOTE_REL


def test_missing_required_column_refused_before_decode(tmp_path, monkeypatch):
    root, file_sha, manifest_sha = _setup(tmp_path, files={QUOTE_REL: QUOTES.drop(columns=["smvVol"])})
    _no_decode(monkeypatch)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "required_column")
    assert exc.value.details["columns"] == ["smvVol"]


def test_compressible_table_refused_by_row_estimate_before_decode(tmp_path, monkeypatch):
    big = pd.concat([QUOTES] * 50_000, ignore_index=True)  # 200_000 rows of identical values
    budget = 10_000_000
    root, file_sha, manifest_sha = _setup(tmp_path, files={QUOTE_REL: big})
    assert (root / QUOTE_REL).stat().st_size * 10 < budget  # far below the budget on disk
    assert len(big) * fvi.DECODED_ROW_BYTES > budget  # but rows * DECODED_ROW_BYTES is not
    _no_decode(monkeypatch)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha, max_total_bytes=budget)
    _refused(exc, "input_too_large", code="RESOURCE_LIMIT_EXCEEDED")
    assert exc.value.details["path"] == QUOTE_REL


def test_uncompressed_column_size_refused_before_decode(tmp_path, monkeypatch):
    # Few rows, long distinct tickers, no dictionary: the uncompressed chunk size is what exceeds the budget.
    wide = QUOTES.assign(ticker=[f"{i}" + "A" * 10_000 for i in range(4)])
    blob = _parquet_bytes(tmp_path, wide, use_dictionary=False)
    root, file_sha, manifest_sha = _setup(tmp_path, files={QUOTE_REL: blob})
    budget = 20_000
    assert 4 * fvi.DECODED_ROW_BYTES < budget  # rows * DECODED_ROW_BYTES alone would admit it
    _no_decode(monkeypatch)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha, max_total_bytes=budget)
    _refused(exc, "input_too_large", code="RESOURCE_LIMIT_EXCEEDED")
    assert exc.value.details["path"] == QUOTE_REL
    assert exc.value.details["total_bytes"] >= 40_000  # the estimate is the uncompressed size


def test_post_decode_check_still_fires(tmp_path, monkeypatch):
    big = pd.concat([QUOTES] * 12_500, ignore_index=True)  # 50_000 rows of identical values
    root, file_sha, manifest_sha = _setup(tmp_path, files={QUOTE_REL: big})
    monkeypatch.setattr(fvi, "DECODED_ROW_BYTES", 0)  # estimate falls back to the uncompressed size
    ((_, _, estimate, measured),) = _plan(root, [QUOTE_REL])
    assert estimate < measured
    budget = (estimate + measured) // 2  # admitted by the estimate, refused by the measured size
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha, max_total_bytes=budget)
    _refused(exc, "input_too_large", code="RESOURCE_LIMIT_EXCEEDED")
    assert exc.value.details["path"] == QUOTE_REL
    assert exc.value.details["total_bytes"] == measured


def test_aggregate_budget_constant_read_at_call_time(tmp_path, monkeypatch):
    root, file_sha, manifest_sha = _setup(tmp_path)
    monkeypatch.setattr(fvi, "MAX_TOTAL_TABLE_BYTES", 64)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "input_too_large", code="RESOURCE_LIMIT_EXCEEDED")
    assert exc.value.details["max_total_bytes"] == 64


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_bad_load_budget_refused_before_reading(tmp_path, bad):
    _, file_sha, manifest_sha = _setup(tmp_path)
    with pytest.raises(FairValueRefusal) as exc:
        _load(tmp_path / "absent", file_sha, manifest_sha, max_total_bytes=bad)  # root does not exist: nothing is read
    _refused(exc, "load_budget")


def _chain(*overrides):
    base = {"ticker": "AAA", "obs_date": "2024-01-02", "expiry": "2024-02-16", "strike": 100.0, "right": "C",
            "bid": 1.0, "ask": 1.2, "iv": 0.3, "spot": 100.0, "quote_repaired": False,
            "open_interest": 50.0, "bid_size": 5.0, "ask_size": 5.0, "chain_kind": "fetch"}
    return pd.DataFrame([{**base, **o} for o in (overrides or ({},))])


@pytest.mark.parametrize("snap,expected", [("a", "b"), ("", ""), (None, None)])
def test_bind_snapshot_refusals(snap, expected):
    with pytest.raises(FairValueRefusal) as exc:
        fvi.bind_quotes(_chain(), snapshot_id=snap, expected_snapshot_id=expected)
    _refused(exc, "snapshot_binding")


def test_bind_required_column_refused():
    with pytest.raises(FairValueRefusal) as exc:
        fvi.bind_quotes(_chain().drop(columns=["iv"]), snapshot_id="s", expected_snapshot_id="s")
    _refused(exc, "required_column")
    assert exc.value.details["columns"] == ["iv"]


def test_bind_duplicate_key_refused():
    with pytest.raises(FairValueRefusal) as exc:
        fvi.bind_quotes(_chain({}, {}), snapshot_id="s", expected_snapshot_id="s")
    _refused(exc, "duplicate_key")
    assert exc.value.details["table"] == "option_chains"


def test_bind_drops_extra_columns_keeps_index_and_allows_empty():
    frame = _chain({}, {"strike": 105.0}).assign(extra=1, junk="x")
    frame.index = [7, 3]
    out = fvi.bind_quotes(frame, snapshot_id="s", expected_snapshot_id="s")
    assert list(out.columns) == fvi.QUOTE_COLUMNS
    assert list(out.index) == [7, 3]
    assert fvi.bind_quotes(frame.iloc[:0], snapshot_id="s", expected_snapshot_id="s").empty


@pytest.mark.parametrize("override,column", [
    ({"ticker": None}, "ticker"),
    ({"strike": None}, "strike"),
    ({"right": None}, "right"),
    ({"obs_date": None}, "obs_date"),
    ({"expiry": None}, "expiry"),
    ({"strike": "abc"}, "strike"),
    ({"obs_date": "garbage"}, "obs_date"),
    ({"expiry": "2024-13-45"}, "expiry"),
])
def test_bind_bad_key_refused(override, column):
    with pytest.raises(FairValueRefusal) as exc:
        fvi.bind_quotes(_chain(override), snapshot_id="s", expected_snapshot_id="s")
    _refused(exc, "key_value")
    assert exc.value.details["columns"] == [column]


@pytest.fixture
def lookup_tables(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path, files={RATE_REL: LOOKUP_RATES})
    return _load(root, file_sha, manifest_sha).rates


def _query(ticker, obs, days):
    start = pd.Timestamp(obs)
    return pd.DataFrame({"ticker": [ticker], "obs_date": [start], "expiry": [start + pd.Timedelta(days=days)]})


LOOKUP_CASES = [
    ("AAA", 0, 0.05), ("AAA", 35, 0.05), ("AAA", 200, 0.045), ("AAA", 365, 0.04), ("AAA", 700, 0.04),
    ("BBB", 0, 0.04),    # r5 out of range (missing after load): r1 used
    ("BBB", 700, 0.04),
    ("FFF", 700, 0.05),  # r1 missing: r5 used
    ("CCC", 0, NAN),     # both out of range
    ("GGG", 0, NAN),     # same-day row with both missing: no fallback to 2024-01-03
    ("DDD", 0, 0.05),    # row exactly 7 days back
    ("EEE", 0, NAN),     # 8 days back
    ("HHH", 0, NAN),     # later-dated row only
    ("III", 0, NAN),     # other ticker never used
]


@pytest.mark.parametrize("ticker,days,expected", LOOKUP_CASES)
def test_lookup_rate_cases(lookup_tables, ticker, days, expected):
    got = fvi.lookup_rate(_query(ticker, "2024-01-10", days), lookup_tables)
    np.testing.assert_allclose(got.to_numpy(), [expected], rtol=1e-9)


def test_lookup_rate_is_independent_of_row_order(lookup_tables):
    rows = pd.concat([_query(t, "2024-01-10", d) for t, d, _ in LOOKUP_CASES], ignore_index=True)
    base = fvi.lookup_rate(rows, lookup_tables)
    order = np.random.default_rng(0).permutation(len(rows))
    shuffled = fvi.lookup_rate(rows.iloc[order], lookup_tables)
    pd.testing.assert_series_equal(shuffled.sort_index(), base.sort_index())


def test_attach_vol_shares_strike_across_rights_and_keeps_order():
    vols = pd.DataFrame({"ticker": ["AAA"], "obs_date": pd.to_datetime(["2024-01-02"]),
                         "expiry": pd.to_datetime(["2024-02-16"]), "strike": [100.0], "smvVol": [0.25]})
    quotes = pd.DataFrame({"ticker": ["AAA"] * 3, "obs_date": ["2024-01-02"] * 3,
                           "expiry": ["2024-02-16"] * 3, "strike": [100.0, 105.0, 100.0],
                           "right": ["P", "C", "C"]}, index=[7, 3, 5])
    before = quotes.copy()
    got = fvi.attach_vol(quotes, vols)
    assert got.name == "smvVol" and got.index.equals(quotes.index)
    np.testing.assert_allclose(got.to_numpy(), [0.25, NAN, 0.25])
    pd.testing.assert_frame_equal(quotes, before)


@pytest.mark.parametrize("column", fvi.VOL_KEYS)
def test_attach_vol_missing_key_column_refused(column):
    quotes = pd.DataFrame({"ticker": ["AAA"], "obs_date": ["2024-01-02"], "expiry": ["2024-02-16"],
                           "strike": [100.0]}).drop(columns=[column])
    with pytest.raises(FairValueRefusal) as exc:
        fvi.attach_vol(quotes, _coverage_tables().vols)
    _refused(exc, "required_column")
    assert exc.value.details["columns"] == [column]


@pytest.mark.parametrize("column", ["obs_date", "expiry"])
def test_attach_vol_unparseable_date_refused(column):
    quotes = pd.DataFrame({"ticker": ["AAA"], "obs_date": ["2024-01-02"], "expiry": ["2024-02-16"],
                           "strike": [100.0]})
    quotes[column] = "not a date"
    with pytest.raises(FairValueRefusal) as exc:
        fvi.attach_vol(quotes, _coverage_tables().vols)
    _refused(exc, "key_value")
    assert exc.value.details["column"] == column


@pytest.mark.parametrize("column", ["ticker", "obs_date", "expiry"])
def test_lookup_rate_missing_key_column_refused(lookup_tables, column):
    rows = _query("AAA", "2024-01-10", 0).drop(columns=[column])
    with pytest.raises(FairValueRefusal) as exc:
        fvi.lookup_rate(rows, lookup_tables)
    _refused(exc, "required_column")
    assert exc.value.details["columns"] == [column]


@pytest.mark.parametrize("column", ["obs_date", "expiry"])
def test_lookup_rate_unparseable_date_refused(lookup_tables, column):
    rows = _query("AAA", "2024-01-10", 0)
    rows[column] = "not a date"
    with pytest.raises(FairValueRefusal) as exc:
        fvi.lookup_rate(rows, lookup_tables)
    _refused(exc, "key_value")
    assert exc.value.details["column"] == column


def test_usable_quote_rejects_zero_bid_and_ask():
    assert fvi.usable_quote([0.0], [0.0]).tolist() == [False]


def _entry_rows():
    return _chain(
        {},                                                                                  # usable, vol, rate
        {"right": "P", "bid": 0.0, "ask": 0.5, "open_interest": 0.0, "bid_size": None},      # bid 0 usable
        {"strike": 110.0, "bid": 2.0, "ask": 1.0, "open_interest": None, "chain_kind": "eod"},  # crossed
        {"expiry": "2024-03-15", "bid": -0.1, "ask": 0.5, "open_interest": None, "chain_kind": None},  # negative bid, no vol
        {"obs_date": "2025-01-02", "bid": 0.5, "ask": 0.0, "open_interest": 10.0},           # ask <= 0
        {"obs_date": "2025-01-02", "right": "P", "bid": None, "ask": 1.0, "open_interest": None},  # NaN bid
    )


def _coverage_tables():
    vols = pd.DataFrame({"ticker": ["AAA"], "obs_date": pd.to_datetime(["2024-01-02"]),
                         "expiry": pd.to_datetime(["2024-02-16"]), "strike": [100.0], "smvVol": [0.25]})
    rates = pd.DataFrame({"ticker": ["AAA"], "obs_date": pd.to_datetime(["2024-01-02"]),
                          "iRate5wk": [0.05], "iRateLt": [0.04]})
    return fvi.InputTables(vols=vols, rates=rates, hashes={}, rate_rules={})


def test_entry_coverage_counts_by_year_and_chain_kind():
    emitted = []
    out = fvi.entry_coverage(_entry_rows(), _coverage_tables(), snapshot_id="s", expected_snapshot_id="s",
                             emit=emitted.append)
    assert len(emitted) == 1 and emitted[0].startswith("[fvf] entry input coverage\n")
    cols = ["year", "chain_kind", "rows", "usable_quote", "smvvol_positive", "rate_found",
            "open_interest_known", "bid_size_known", "ask_size_known"]
    assert out[cols].values.tolist() == [
        [2024, "eod", 1, 0, 0, 1, 0, 1, 1],
        [2024, "fetch", 2, 2, 2, 2, 2, 1, 2],
        [2024, "null", 1, 0, 0, 1, 0, 1, 1],
        [2025, "fetch", 2, 0, 0, 0, 1, 2, 2],
    ]
    for name in cols[3:]:
        np.testing.assert_allclose(out[name + "_share"], out[name] / out["rows"])


def test_entry_coverage_empty_input_still_emits():
    emitted = []
    out = fvi.entry_coverage(_chain().iloc[:0], _coverage_tables(), snapshot_id="s", expected_snapshot_id="s",
                             emit=emitted.append)
    assert out.empty and len(emitted) == 1
    assert list(out.columns)[:3] == ["year", "chain_kind", "rows"] and "rate_found_share" in out.columns


@pytest.mark.parametrize("frame,snap,expected,reason", [
    (_chain(), "a", "b", "snapshot_binding"),
    (_chain().drop(columns=["bid"]), "s", "s", "required_column"),
    (_chain().drop(columns=["expiry"]), "s", "s", "required_column"),
    (_chain({}, {}), "s", "s", "duplicate_key"),
    (_chain({"obs_date": None}), "s", "s", "key_value"),
    (_chain({"expiry": "not a date"}), "s", "s", "key_value"),
], ids=["wrong-id", "missing-bid", "missing-expiry", "duplicate-key", "null-obs-date", "unparseable-expiry"])
def test_entry_coverage_refuses_before_emit(frame, snap, expected, reason):
    emitted = []
    with pytest.raises(FairValueRefusal) as exc:
        fvi.entry_coverage(frame, _coverage_tables(), snapshot_id=snap, expected_snapshot_id=expected,
                           emit=emitted.append)
    _refused(exc, reason)
    assert emitted == []


def test_rate_rules_recorded(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path)
    assert _load(root, file_sha, manifest_sha).rate_rules == {"rate_out_of_range": "missing", "rate_empty_row": "rate_unknown"}
    tables = _load(root, file_sha, manifest_sha, rate_out_of_range="row_unknown", rate_empty_row="earlier_row")
    assert tables.rate_rules == {"rate_out_of_range": "row_unknown", "rate_empty_row": "earlier_row"}


@pytest.mark.parametrize("name,bad", [("rate_out_of_range", None), ("rate_out_of_range", "Missing"),
                                      ("rate_out_of_range", "earlier_row"), ("rate_empty_row", None),
                                      ("rate_empty_row", "missing"), ("rate_empty_row", True)])
def test_bad_rate_rule_refused_before_reading(tmp_path, name, bad):
    _, file_sha, manifest_sha = _setup(tmp_path)
    with pytest.raises(FairValueRefusal) as exc:
        _load(tmp_path / "absent", file_sha, manifest_sha, **{name: bad})  # root does not exist: nothing is read
    _refused(exc, "rate_rule")
    assert exc.value.details["name"] == name


# Rate edge rows, queried at 2024-01-10 (expiry +10 days, so a row with both values uses iRate5wk).
# A: same-day row has iRate5wk out of range; B: same-day row both NaN; C: good row only 8 days back;
# D: same-day iRate5wk NaN; E: same-day iRate5wk +inf. Earlier good rows at 4% sit 5 days back (A, B, E).
EDGE_RATES = pd.DataFrame({
    "ticker": ["A", "A", "B", "B", "C", "C", "D", "E", "E"],
    "obs_date": pd.to_datetime(["2024-01-05", "2024-01-10", "2024-01-05", "2024-01-10", "2024-01-02",
                                "2024-01-10", "2024-01-10", "2024-01-05", "2024-01-10"]),
    "iRate5wk": [4.0, 30.0, 4.0, NAN, 4.0, NAN, NAN, 4.0, float("inf")],
    "iRateLt": [4.0, 3.0, 4.0, NAN, 4.0, NAN, 3.0, 4.0, 3.0],
    "borrow30": 0.1, "annIdiv": 0.0, "divDate": "x", "assetType": "stock",
})
EDGE_EXPECT = {
    ("missing", "rate_unknown"): {"A": 0.03, "B": NAN, "C": NAN, "D": 0.03, "E": 0.03},
    ("missing", "earlier_row"): {"A": 0.03, "B": 0.04, "C": NAN, "D": 0.03, "E": 0.03},
    ("row_unknown", "rate_unknown"): {"A": NAN, "B": NAN, "C": NAN, "D": 0.03, "E": NAN},
    ("row_unknown", "earlier_row"): {"A": 0.04, "B": 0.04, "C": NAN, "D": 0.03, "E": 0.04},
}


@pytest.mark.parametrize("combo", list(EDGE_EXPECT))
def test_rate_rule_combinations_on_edge_rows(tmp_path, combo):
    root, file_sha, manifest_sha = _setup(tmp_path, files={RATE_REL: EDGE_RATES})
    tables = _load(root, file_sha, manifest_sha, rate_out_of_range=combo[0], rate_empty_row=combo[1])
    rows = pd.concat([_query(t, "2024-01-10", 10) for t in "ABCDE"], ignore_index=True)
    got = fvi.lookup_rate(rows, tables.rates).to_numpy()
    np.testing.assert_allclose(got, [EDGE_EXPECT[combo][t] for t in "ABCDE"], rtol=1e-9)


def test_duplicate_rate_key_with_both_rows_empty_refused_under_earlier_row(tmp_path):
    dup = pd.DataFrame({"ticker": ["A", "A"], "obs_date": pd.to_datetime(["2024-01-10"] * 2),
                        "iRate5wk": [NAN, NAN], "iRateLt": [NAN, NAN]})
    root, file_sha, manifest_sha = _setup(tmp_path, files={RATE_REL: dup})
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha, rate_empty_row="earlier_row")
    _refused(exc, "duplicate_key")
    assert exc.value.details["table"] == "rates"
