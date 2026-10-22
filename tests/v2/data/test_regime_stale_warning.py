"""#1103 — the regime block must say so when the ^GSPC series trails the frame.

The legacy ``gspc_daily.csv`` was fetched once and nothing refreshed it, so a
frame whose decision dates run past the series end was silently served stale
market state. The fix is a warning only: one ``UserWarning`` and one stderr
line, once per ``(path, mtime)``, with every computed value unchanged. These
tests pin the warning's shape, the once-per-file-version suppression, the
strictly-seven-day threshold, and value identity across the call that warned
and the one that stayed quiet.

Synthetic data only: no network, nothing under the real data root.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from engine.data.features import panel
from engine.data.features.panel import add_regime_features

SERIES_END = pd.Timestamp("2026-08-27")
STALE_DECISION = pd.Timestamp("2026-10-29")  # 63 days past the series end
WITHIN_WINDOW = pd.Timestamp("2026-09-03")  # exactly 7 days past: must not warn

STALE_MATCH = "regime features are stale"


@pytest.fixture(autouse=True)
def _fresh_dedup_state(monkeypatch):
    """Once-per-(path, mtime) is module state; give each test an empty set."""
    monkeypatch.setattr(panel, "_STALE_REGIME_WARNED", set())


def _write_gspc(path, periods: int = 400) -> None:
    """The legacy multi-header ^GSPC layout with strictly-rising closes."""
    dates = pd.date_range(end=SERIES_END, periods=periods, freq="D")
    closes = 1000.0 + np.arange(len(dates), dtype=float)
    lines = [
        "Price,Adj Close,Close,High,Low,Open,Volume",
        "Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC",
        "Date,,,,,,",
    ]
    for day, close in zip(dates, closes):
        lines.append(f"{day.date()},{close},{close},{close},{close},{close},1000")
    path.write_text("\n".join(lines) + "\n")


def _frame(decision: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime([decision])})


def test_stale_frame_warns_once_and_a_second_call_stays_quiet(tmp_path):
    path = tmp_path / "gspc_daily.csv"
    _write_gspc(path)
    frame = _frame(STALE_DECISION)

    with pytest.warns(UserWarning, match=STALE_MATCH):
        first = add_regime_features(frame, gspc_path=path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        second = add_regime_features(frame, gspc_path=path)
    assert [w for w in caught if "stale" in str(w.message)] == []

    # Values are identical with and without the warning, and they are the real
    # anchor: the last close strictly before the decision date is the series
    # end (row 399). The staleness path is diagnostic only.
    pd.testing.assert_frame_equal(first, second)
    raw = pd.read_csv(path, skiprows=3, header=None)
    closes = raw[2].to_numpy(dtype=float)
    expected = (closes[399] / closes[399 - 21] - 1.0) * 100
    assert first["spy_ret21"].iloc[0] == pytest.approx(expected)
    assert first["regime_asof"].iloc[0] == np.datetime64(str(SERIES_END.date()))


def test_stale_frame_emits_exactly_one_stderr_line(tmp_path, capsys):
    path = tmp_path / "gspc_daily.csv"
    _write_gspc(path)
    expected_line = (
        "[regime] WARNING gspc series ends 2026-08-27, decision date "
        "2026-10-29: regime features are stale"
    )

    with pytest.warns(UserWarning, match=STALE_MATCH):
        add_regime_features(_frame(STALE_DECISION), gspc_path=path)
    err = capsys.readouterr().err
    assert expected_line in err
    assert err.count("[regime] WARNING") == 1

    # The same file version never repeats it, on stderr either.
    add_regime_features(_frame(STALE_DECISION), gspc_path=path)
    assert capsys.readouterr().err == ""


def test_decision_within_seven_days_does_not_warn(tmp_path, capsys):
    path = tmp_path / "gspc_daily.csv"
    _write_gspc(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = add_regime_features(_frame(WITHIN_WINDOW), gspc_path=path)

    assert [w for w in caught if "stale" in str(w.message)] == []
    assert "[regime] WARNING" not in capsys.readouterr().err
    # It still computed the block normally: the series end is the anchor.
    assert out["regime_asof"].iloc[0] == np.datetime64(str(SERIES_END.date()))
    assert np.isfinite(out["spy_ret21"].iloc[0])


def test_refreshed_file_can_warn_again(tmp_path):
    """The suppression key is (path, mtime), not path: a refreshed file that is
    still stale behind a new decision date must be able to warn again."""
    path = tmp_path / "gspc_daily.csv"
    _write_gspc(path)
    frame = _frame(STALE_DECISION)

    with pytest.warns(UserWarning, match=STALE_MATCH):
        add_regime_features(frame, gspc_path=path)

    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    with pytest.warns(UserWarning, match=STALE_MATCH):
        add_regime_features(frame, gspc_path=path)
