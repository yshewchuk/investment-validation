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


def _entry_rows():
    return fvi.bind_quotes(_chain(
        {},                                                                                  # usable, vol, rate
        {"right": "P", "bid": 0.0, "ask": 0.5, "open_interest": 0.0, "bid_size": None},      # bid 0 usable
        {"strike": 110.0, "bid": 2.0, "ask": 1.0, "open_interest": None, "chain_kind": "eod"},  # crossed
        {"expiry": "2024-03-15", "bid": -0.1, "ask": 0.5, "open_interest": None, "chain_kind": None},  # negative bid, no vol
        {"obs_date": "2025-01-02", "bid": 0.5, "ask": 0.0, "open_interest": 10.0},           # ask <= 0
        {"obs_date": "2025-01-02", "right": "P", "bid": None, "ask": 1.0, "open_interest": None},  # NaN bid
    ), snapshot_id="s", expected_snapshot_id="s")


def _coverage_tables():
    vols = pd.DataFrame({"ticker": ["AAA"], "obs_date": pd.to_datetime(["2024-01-02"]),
                         "expiry": pd.to_datetime(["2024-02-16"]), "strike": [100.0], "smvVol": [0.25]})
    rates = pd.DataFrame({"ticker": ["AAA"], "obs_date": pd.to_datetime(["2024-01-02"]),
                          "iRate5wk": [0.05], "iRateLt": [0.04]})
    return fvi.InputTables(vols=vols, rates=rates, hashes={})


def test_entry_coverage_counts_by_year_and_chain_kind():
    emitted = []
    out = fvi.entry_coverage(_entry_rows(), _coverage_tables(), emit=emitted.append)
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
    out = fvi.entry_coverage(fvi.bind_quotes(_chain().iloc[:0], snapshot_id="s", expected_snapshot_id="s"),
                             _coverage_tables(), emit=emitted.append)
    assert out.empty and len(emitted) == 1
    assert list(out.columns)[:3] == ["year", "chain_kind", "rows"] and "rate_found_share" in out.columns
