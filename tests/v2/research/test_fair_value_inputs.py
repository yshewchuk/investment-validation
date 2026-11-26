"""Tests for the fair-value input layer (fill model F1; design sections 3, 10, 13)."""

from __future__ import annotations

import hashlib
import json
import logging

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
UNITS = {"borrow30": "percent", "borrow2yr": "percent", "divAmt": "dollars_per_share"}


def _carry(n):
    """Carry columns for n loaded rate rows: borrow in percent (divided by 100 on load), no dividend."""
    return {"borrow30": [10.0] * n, "borrow2yr": [10.0] * n, "divAmt": [0.0] * n, "divDate": pd.to_datetime([None] * n),
            "annIdiv": [0.0] * n, "annActDiv": [0.0] * n, "assetType": [1.0] * n}


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
    **_carry(2),
})
# Lookup fixture: queries at obs 2024-01-10 (see LOOKUP_CASES).
LOOKUP_RATES = pd.DataFrame({
    "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "GGG", "HHH"],
    "obs_date": pd.to_datetime(["2024-01-10", "2024-01-10", "2024-01-10", "2024-01-03", "2024-01-02",
                                "2024-01-10", "2024-01-03", "2024-01-10", "2024-01-12"]),
    "iRate5wk": [5.0, 30.0, 30.0, 5.0, 5.0, 5.0, 5.0, NAN, 5.0],
    "iRateLt": [4.0, 4.0, -9.0, 4.0, 4.0, NAN, 4.0, NAN, 4.0],
    **_carry(9),
})


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _mani(q=OK_Q, r=OK_R):
    tables = {}
    if q is not None:
        tables["quotes"] = q
    if r is not None:
        tables["rates"] = r
    return {"version": "3", "tables": tables, "units": UNITS, "dividend_fields": {}, "extra_top": 1}


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
    assert list(tables.rates.columns) == fvi.RATE_KEYS + fvi.RATE_COLUMNS + fvi.CARRY_COLUMNS
    vols = tables.vols.set_index("strike")["smvVol"]
    assert vols[100.0] == pytest.approx(0.25)
    assert vols[[110.0, 90.0, 95.0]].isna().all()
    rates = tables.rates.set_index("ticker")
    assert rates.loc["AAA", "iRate5wk"] == pytest.approx(0.053)
    assert rates.loc["AAA", "iRateLt"] == pytest.approx(0.041)
    assert np.isnan(rates.loc["BBB", "iRate5wk"])
    assert set(tables.hashes) == set(file_sha) | {"manifest.json"}
    for rel, digest in tables.hashes.items():  # manifest and every table: sha256 of the bytes on disk
        assert digest == _sha((root / rel).read_bytes())


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


def test_load_nan_smvvol_becomes_missing(tmp_path):
    root, file_sha, manifest_sha = _setup(tmp_path, files={QUOTE_REL: QUOTES.assign(smvVol=[25.0, NAN, 30.0, NAN])})
    vols = _load(root, file_sha, manifest_sha).vols.set_index("strike")["smvVol"]
    assert vols[100.0] == pytest.approx(0.25) and vols[90.0] == pytest.approx(0.30)
    assert np.isnan(vols[110.0]) and np.isnan(vols[95.0])


def test_load_rate_bounds_inclusive_after_conversion(tmp_path):
    # rate_scale 0.01: -5.0 and 20.0 convert to exactly -0.05 and 0.20 (kept); -5.01 and 20.01 convert to -0.0501 and 0.2001 (nulled).
    rates = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC", "DDD"], "obs_date": pd.to_datetime(["2024-01-02"] * 4),
        "iRate5wk": [-5.0, 20.0, -5.01, 4.0], "iRateLt": [-5.0, 20.0, 4.0, 20.01],
        **_carry(4),
    })
    root, file_sha, manifest_sha = _setup(tmp_path, files={RATE_REL: rates})
    got = _load(root, file_sha, manifest_sha).rates.set_index("ticker")
    assert got.loc["AAA", "iRate5wk"] == pytest.approx(-0.05) and got.loc["AAA", "iRateLt"] == pytest.approx(-0.05)
    assert got.loc["BBB", "iRate5wk"] == pytest.approx(0.20) and got.loc["BBB", "iRateLt"] == pytest.approx(0.20)
    assert np.isnan(got.loc["CCC", "iRate5wk"]) and got.loc["CCC", "iRateLt"] == pytest.approx(0.04)
    assert got.loc["DDD", "iRate5wk"] == pytest.approx(0.04) and np.isnan(got.loc["DDD", "iRateLt"])


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


KEY_CASES = [
    (QUOTE_REL, {"ticker": [None, "AAA", "AAA", "AAA"]}, "ticker"),
    (RATE_REL, {"ticker": [None, "BBB"]}, "ticker"),
    (RATE_REL, {"obs_date": pd.to_datetime([None, "2024-01-02"])}, "obs_date"),
    (QUOTE_REL, {"strike": [None, 110.0, 90.0, 95.0]}, "strike"),
    (QUOTE_REL, {"expiry": ["not a date"] + ["2024-02-16"] * 3}, "expiry"),
]


@pytest.mark.parametrize("rel,override,column", KEY_CASES,
                         ids=["quote-null-ticker", "rate-null-ticker", "rate-null-obs-date",
                              "quote-null-strike", "quote-unparseable-expiry"])
def test_table_null_or_unconvertible_key_refused(tmp_path, rel, override, column):
    base = QUOTES if rel == QUOTE_REL else RATES
    root, file_sha, manifest_sha = _setup(tmp_path, files={rel: base.assign(**override)})
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "key_value")
    assert exc.value.details["columns"] == [column]


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


@pytest.mark.parametrize("override", [{"obs_date": pd.Timestamp("2024-01-02")}, {"strike": "100"}])
def test_bind_normalised_duplicate_key_refused(override):
    with pytest.raises(FairValueRefusal) as exc:
        fvi.bind_quotes(_chain({}, override), snapshot_id="s", expected_snapshot_id="s")
    _refused(exc, "duplicate_key")
    assert exc.value.details["table"] == "option_chains"


def test_bind_drops_extra_columns_keeps_index_and_allows_empty():
    frame = _chain({}, {"strike": 105.0}).assign(extra=1, junk="x")
    frame.index = [7, 3]
    out = fvi.bind_quotes(frame, snapshot_id="s", expected_snapshot_id="s")
    assert list(out.columns) == fvi.QUOTE_COLUMNS
    pd.testing.assert_frame_equal(out, frame[fvi.QUOTE_COLUMNS])  # values, dtypes and index preserved
    empty = fvi.bind_quotes(frame.iloc[:0], snapshot_id="s", expected_snapshot_id="s")
    pd.testing.assert_frame_equal(empty, frame.iloc[:0][fvi.QUOTE_COLUMNS])


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


def test_lookup_rate_is_independent_of_row_order():
    # Rate table and query rows are both unsorted, with two obs_dates per ticker: a missing sort changes or breaks the result.
    rates = pd.DataFrame({
        "ticker": ["BBB", "AAA", "BBB", "AAA"],
        "obs_date": pd.to_datetime(["2024-01-02", "2024-01-08", "2024-01-08", "2024-01-02"]),
        "iRate5wk": [0.02, 0.06, 0.07, 0.04],
        "iRateLt": [0.02, 0.05, 0.06, 0.03],
        **_carry(4),
    })
    rows = pd.concat([_query("BBB", "2024-01-10", 200), _query("AAA", "2024-01-10", 0), _query("AAA", "2024-01-03", 0),
                      _query("BBB", "2024-01-10", 0), _query("AAA", "2024-01-09", 200), _query("AAA", "2024-01-10", 365)],
                     ignore_index=True)
    # Hand-computed: BBB 01-08 row at 200 days = 0.07 + 0.5 * (0.06 - 0.07); AAA 01-08 row at 0 days = 0.06;
    # AAA query on 01-03 uses the 01-02 row = 0.04; BBB at 0 days = 0.07; AAA 01-08 row at 200 days = 0.06 + 0.5 * (0.05 - 0.06);
    # AAA at 365 days = rL 0.05.
    expected = [0.065, 0.06, 0.04, 0.07, 0.055, 0.05]
    np.testing.assert_allclose(fvi.lookup_rate(rows, rates).to_numpy(), expected, rtol=1e-9)


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
    assert exc.value.details["columns"] == [column]


@pytest.mark.parametrize("column", fvi.VOL_KEYS)
def test_attach_vol_null_key_refused(column):
    quotes = pd.DataFrame({"ticker": ["AAA"], "obs_date": ["2024-01-02"], "expiry": ["2024-02-16"],
                           "strike": [100.0]})
    quotes[column] = [None]
    with pytest.raises(FairValueRefusal) as exc:
        fvi.attach_vol(quotes, _coverage_tables().vols)
    _refused(exc, "key_value")
    assert exc.value.details["columns"] == [column]


def _vol_table(obs):
    n = len(obs)
    return pd.DataFrame({"ticker": ["AAA"] * n, "obs_date": obs, "expiry": pd.to_datetime(["2024-02-16"] * n),
                         "strike": [100.0] * n, "smvVol": [0.25] * n})


@pytest.mark.parametrize("obs", [pd.to_datetime(["2024-01-02"] * 2), ["2024-01-02", pd.Timestamp("2024-01-02")]],
                         ids=["exact-duplicate", "string-vs-timestamp"])
def test_attach_vol_duplicate_vol_key_refused(obs):
    with pytest.raises(FairValueRefusal) as exc:
        fvi.attach_vol(_chain(), _vol_table(obs))
    _refused(exc, "duplicate_key")
    assert exc.value.details["table"] == "quotes"


@pytest.mark.parametrize("column", fvi.VOL_KEYS)
def test_attach_vol_null_vol_key_refused(column):
    vols = _coverage_tables().vols
    vols[column] = [None]
    with pytest.raises(FairValueRefusal) as exc:
        fvi.attach_vol(_chain(), vols)
    _refused(exc, "key_value")
    assert exc.value.details["columns"] == [column]


def test_attach_vol_missing_smvvol_refused():
    vols = _coverage_tables().vols.drop(columns=["smvVol"])
    with pytest.raises(FairValueRefusal) as exc:
        fvi.attach_vol(_chain(), vols)
    _refused(exc, "required_column")
    assert exc.value.details["columns"] == ["smvVol"]


def test_attach_vol_empty_frame_still_works():
    got = fvi.attach_vol(_chain().iloc[:0], _coverage_tables().vols)
    assert got.empty and got.name == "smvVol"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("-inf")], ids=["zero", "negative", "inf", "-inf"])
def test_attach_vol_invalid_vol_value_refused(bad):
    vols = _coverage_tables().vols.assign(smvVol=[bad])
    with pytest.raises(FairValueRefusal) as exc:
        fvi.attach_vol(_chain(), vols)
    _refused(exc, "table_value")
    assert exc.value.details["table"] == "quotes"
    assert exc.value.details["columns"] == ["smvVol"]


@pytest.mark.parametrize("column", ["ticker", "obs_date", "expiry"])
def test_lookup_rate_null_key_refused(lookup_tables, column):
    rows = _query("AAA", "2024-01-10", 0)
    rows[column] = None
    with pytest.raises(FairValueRefusal) as exc:
        fvi.lookup_rate(rows, lookup_tables)
    _refused(exc, "key_value")
    assert exc.value.details["columns"] == [column]


def test_lookup_rate_null_ticker_in_rates_refused():
    rates = _coverage_tables().rates.assign(ticker=[None])
    with pytest.raises(FairValueRefusal) as exc:
        fvi.lookup_rate(_query("AAA", "2024-01-10", 0), rates)
    _refused(exc, "key_value")
    assert exc.value.details["columns"] == ["ticker"]


@pytest.mark.parametrize("column", fvi.RATE_COLUMNS)
def test_lookup_rate_missing_rate_column_refused(column):
    rates = _coverage_tables().rates.drop(columns=[column])
    with pytest.raises(FairValueRefusal) as exc:
        fvi.lookup_rate(_query("AAA", "2024-01-10", 0), rates)
    _refused(exc, "required_column")
    assert exc.value.details["columns"] == [column]


def _rate_table(obs, r5=0.05, rl=0.04):
    n = len(obs)
    return pd.DataFrame({"ticker": ["AAA"] * n, "obs_date": obs, "iRate5wk": [r5] * n, "iRateLt": [rl] * n, **_carry(n)})


@pytest.mark.parametrize("rows", ["one-row", "empty-rows"])
@pytest.mark.parametrize("obs", [pd.to_datetime(["2024-01-02"] * 2), ["2024-01-02", pd.Timestamp("2024-01-02")]],
                         ids=["exact-duplicate", "string-vs-timestamp"])
def test_lookup_rate_duplicate_rate_key_refused(rows, obs):
    query = _query("AAA", "2024-01-10", 0)
    if rows == "empty-rows":
        query = query.iloc[:0]
    with pytest.raises(FairValueRefusal) as exc:
        fvi.lookup_rate(query, _rate_table(obs))
    _refused(exc, "duplicate_key")
    assert exc.value.details["table"] == "rates"


def test_lookup_rate_empty_frame_still_works(lookup_tables):
    got = fvi.lookup_rate(_query("AAA", "2024-01-10", 0).iloc[:0], lookup_tables)
    assert got.empty and got.name == "rate"


@pytest.mark.parametrize("empty", [False, True], ids=["one-row", "empty-rows"])
@pytest.mark.parametrize("column", fvi.RATE_COLUMNS)
@pytest.mark.parametrize("bad", [0.21, -0.06, float("inf"), float("-inf")], ids=["above", "below", "inf", "-inf"])
def test_lookup_rate_out_of_bounds_rate_refused(column, bad, empty):
    rates = _coverage_tables().rates.assign(**{column: [bad]})
    query = _query("AAA", "2024-01-10", 0)
    if empty:
        query = query.iloc[:0]
    with pytest.raises(FairValueRefusal) as exc:
        fvi.lookup_rate(query, rates)
    _refused(exc, "table_value")
    assert exc.value.details["table"] == "rates"
    assert exc.value.details["columns"] == [column]


@pytest.mark.parametrize("value", [-0.05, 0.20, NAN])
def test_lookup_rate_boundary_and_nan_rates_accepted(value):
    rates = _coverage_tables().rates.assign(iRate5wk=[value], iRateLt=[value])
    got = fvi.lookup_rate(_query("AAA", "2024-01-02", 0), rates)
    np.testing.assert_allclose(got.to_numpy(), [value], rtol=1e-9)


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
    assert exc.value.details["columns"] == [column]


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
                          "iRate5wk": [0.05], "iRateLt": [0.04], **_carry(1)})
    return fvi.InputTables(vols=vols, rates=rates, hashes={}, rate_rules={})


COVERAGE_COLUMNS = ["year", "chain_kind", "rows", "usable_quote", "usable_quote_share", "smvvol_positive",
                    "smvvol_positive_share", "rate_found", "rate_found_share", "open_interest_known",
                    "open_interest_known_share", "bid_size_known", "bid_size_known_share", "ask_size_known",
                    "ask_size_known_share"]


def test_entry_coverage_counts_by_year_and_chain_kind():
    emitted = []
    out = fvi.entry_coverage(_entry_rows(), _coverage_tables(), snapshot_id="s", expected_snapshot_id="s",
                             emit=emitted.append)
    assert emitted == ["[fvf] entry input coverage\n" + out.to_string(index=False)]
    assert list(out.columns) == COVERAGE_COLUMNS
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
    assert list(out.columns) == COVERAGE_COLUMNS
    assert emitted == ["[fvf] entry input coverage\n" + out.to_string(index=False)]


def test_entry_coverage_invalid_table_value_refused_before_emit():
    base = _coverage_tables()
    tables = fvi.InputTables(vols=base.vols.assign(smvVol=[-1.0]), rates=base.rates, hashes={}, rate_rules={})
    emitted = []
    with pytest.raises(FairValueRefusal) as exc:
        fvi.entry_coverage(_entry_rows(), tables, snapshot_id="s", expected_snapshot_id="s", emit=emitted.append)
    _refused(exc, "table_value")
    assert exc.value.details["table"] == "quotes"
    assert exc.value.details["columns"] == ["smvVol"]
    assert emitted == []


def _bad_coverage_tables(case):
    base = _coverage_tables()
    vols, rates = base.vols, base.rates
    if case == "vol-zero":
        vols = vols.assign(smvVol=[0.0])
    elif case == "rate-out-of-range":
        rates = rates.assign(iRate5wk=[0.21])
    elif case == "missing-smvvol":
        vols = vols.drop(columns=["smvVol"])
    elif case == "missing-rate-column":
        rates = rates.drop(columns=["iRate5wk"])
    elif case == "null-vol-key":
        vols = vols.assign(strike=[None])
    elif case == "duplicate-rate-key":
        rates = pd.concat([rates, rates], ignore_index=True)
    return fvi.InputTables(vols=vols, rates=rates, hashes={}, rate_rules={})


@pytest.mark.parametrize("case,reason", [
    ("vol-zero", "table_value"),
    ("rate-out-of-range", "table_value"),
    ("missing-smvvol", "required_column"),
    ("missing-rate-column", "required_column"),
    ("null-vol-key", "key_value"),
    ("duplicate-rate-key", "duplicate_key"),
])
def test_entry_coverage_bad_table_refused_on_empty_entry_rows(case, reason):
    emitted = []
    with pytest.raises(FairValueRefusal) as exc:
        fvi.entry_coverage(_chain().iloc[:0], _bad_coverage_tables(case), snapshot_id="s", expected_snapshot_id="s",
                           emit=emitted.append)
    _refused(exc, reason)
    assert emitted == []


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
    **_carry(9),
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
                        "iRate5wk": [NAN, NAN], "iRateLt": [NAN, NAN], **_carry(2)})
    root, file_sha, manifest_sha = _setup(tmp_path, files={RATE_REL: dup})
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha, rate_empty_row="earlier_row")
    _refused(exc, "duplicate_key")
    assert exc.value.details["table"] == "rates"


@pytest.mark.parametrize("units", [
    None,
    "percent",
    {"borrow30": "decimal", "borrow2yr": "percent", "divAmt": "dollars_per_share"},
    {"borrow30": "percent", "borrow2yr": "percent", "divAmt": "cents"},
    {"borrow30": "percent", "divAmt": "dollars_per_share"},
], ids=["missing", "not-object", "borrow-decimal", "divamt-cents", "borrow2yr-absent"])
def test_input_unit_refused(tmp_path, units):
    manifest = _mani()
    if units is None:
        del manifest["units"]
    else:
        manifest["units"] = units
    root, file_sha, manifest_sha = _setup(tmp_path, manifest=manifest)
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "input_unit")


@pytest.mark.parametrize("column", fvi.CARRY_COLUMNS)
def test_missing_carry_column_refused(tmp_path, column):
    root, file_sha, manifest_sha = _setup(tmp_path, files={RATE_REL: RATES.drop(columns=[column])})
    with pytest.raises(FairValueRefusal) as exc:
        _load(root, file_sha, manifest_sha)
    _refused(exc, "required_column")
    assert exc.value.details["columns"] == [column]


def test_borrow_percent_divided_explicitly_bounds_inclusive_and_counted(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="fair_value_inputs")
    rates = RATES.assign(borrow30=[-100.0, 20.01], borrow2yr=[20.0, np.inf])
    root, file_sha, manifest_sha = _setup(tmp_path, files={RATE_REL: rates})
    got = _load(root, file_sha, manifest_sha, rate_scale=0.0001).rates.set_index("ticker")  # rate_scale does not touch borrow
    assert got.loc["AAA", "borrow30"] == pytest.approx(-1.0) and got.loc["AAA", "borrow2yr"] == pytest.approx(0.2)
    assert np.isnan(got.loc["BBB", "borrow30"]) and np.isnan(got.loc["BBB", "borrow2yr"])  # 0.2001 and inf
    assert "borrow_out_of_range=2" in caplog.text


def test_negative_dividend_kept_and_missing_div_date_is_nat(tmp_path):
    rates = RATES.assign(divAmt=[-0.5, 1.25], divDate=pd.to_datetime(["2024-03-01", None]))
    root, file_sha, manifest_sha = _setup(tmp_path, files={RATE_REL: rates})
    got = _load(root, file_sha, manifest_sha).rates.set_index("ticker")
    assert got.loc["AAA", "divAmt"] == -0.5 and got.loc["BBB", "divAmt"] == 1.25  # the pricer rules decide, not the loader
    assert got.loc["AAA", "divDate"] == pd.Timestamp("2024-03-01") and pd.isna(got.loc["BBB", "divDate"])


def _one_rate(**over):
    """One AAA rate row on 2024-01-02 with decimal carry, for direct lookup_inputs calls (no load)."""
    row = {"ticker": ["AAA"], "obs_date": pd.to_datetime(["2024-01-02"]), "iRate5wk": [0.05], "iRateLt": [0.04],
           "borrow30": [0.02], "borrow2yr": [0.03], "divAmt": [0.0], "divDate": pd.to_datetime([None]),
           "annIdiv": [0.0], "annActDiv": [0.0], "assetType": [1.0]}
    return pd.DataFrame({**row, **over})


def test_lookup_inputs_takes_every_carry_field_from_the_selected_row():
    rates = pd.DataFrame({
        "ticker": ["AAA", "AAA"], "obs_date": pd.to_datetime(["2023-12-31", "2024-01-02"]),
        "iRate5wk": [0.05, 0.06], "iRateLt": [0.04, 0.05], "borrow30": [0.02, np.nan], "borrow2yr": [0.03, 0.03],
        "divAmt": [0.0, 0.5], "divDate": pd.to_datetime(["2024-01-20", "2024-02-01"]),
        "annIdiv": [0.0, 0.0], "annActDiv": [0.0, 0.0], "assetType": [1.0, 1.0],
    })
    got = fvi.lookup_inputs(_query("AAA", "2024-01-03", 30), rates).iloc[0]
    # The 2024-01-02 row is selected; its missing borrow30 is not filled from the 2023-12-31 row.
    assert np.isnan(got["borrow30"]) and got["borrow2yr"] == pytest.approx(0.03)
    assert got["divAmt"] == 0.5 and got["divDate"] == pd.Timestamp("2024-02-01") and got["rate"] == pytest.approx(0.06)


def test_lookup_inputs_no_row_gives_nan_and_nat():
    got = fvi.lookup_inputs(_query("ZZZ", "2024-01-03", 30), _one_rate())
    assert got["divDate"].isna().all() and got.drop(columns="divDate").isna().all().all()


def test_lookup_inputs_empty_frames_keep_columns_and_datetime_dates():
    empty = fvi.lookup_inputs(_query("AAA", "2024-01-03", 30).iloc[:0], _one_rate())
    assert empty.empty and list(empty.columns) == ["rate"] + fvi.CARRY_COLUMNS
    assert empty["divDate"].dtype == "datetime64[ns]"
    none = fvi.lookup_inputs(_query("AAA", "2024-01-03", 30), _one_rate().iloc[:0])
    assert len(none) == 1 and none["divDate"].dtype == "datetime64[ns]" and pd.isna(none["divDate"].iloc[0])


@pytest.mark.parametrize("bad", [0.21, -1.01, np.inf, -np.inf])
def test_lookup_inputs_out_of_range_borrow_refused(bad):
    with pytest.raises(FairValueRefusal) as exc:
        fvi.lookup_inputs(_query("AAA", "2024-01-03", 30), _one_rate(borrow2yr=[bad]))
    _refused(exc, "table_value")
    assert exc.value.details["columns"] == ["borrow2yr"]


@pytest.mark.parametrize("column", fvi.CARRY_COLUMNS)
def test_lookup_inputs_missing_carry_column_refused(column):
    with pytest.raises(FairValueRefusal) as exc:
        fvi.lookup_inputs(_query("AAA", "2024-01-03", 30), _one_rate().drop(columns=[column]))
    _refused(exc, "required_column")
    assert exc.value.details["columns"] == [column]
