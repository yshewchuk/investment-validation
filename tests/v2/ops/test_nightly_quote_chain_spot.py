"""The option-chain ``spot`` staged onto quote rows as ``chain_spot``.

``scan_quote_rows`` carries the chain's own spot through for the later
``entry_cost_pct`` consumer: every row surviving the expiry filter gets a
``chain_spot`` float, an unusable or disagreeing spot on a SURVIVING row is
``CONTRACT_MISMATCH`` (the same class as an unusable bid/ask) and no partial
tuple is returned, and an empty slice stays an empty ``QuoteRowInputs``. Rows
for another expiry are never judged on it.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from engine.v2.data.errors import DataError
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_quote_rows import QuoteRowInputs, scan_quote_rows
from tests.v2.ops.chain_support import (
    SESSION as _SESSION,
)
from tests.v2.ops.chain_support import (
    STRADDLE_KEY as _KEY,
)
from tests.v2.ops.chain_support import (
    chain_row as _physical_chain_row,
)
from tests.v2.ops.chain_support import (
    price_row as _price_row,
)
from tests.v2.ops.chain_support import (
    snapshot as _snapshot,
)

_EXPIRY = "2024-02-16"
_TARGET = datetime(2024, 2, 16)
_OTHER = datetime(2024, 3, 15)
_OBS = datetime(2024, 1, 5)


def _row(*, right="C", strike=100.0, bid=1.0, ask=1.2, spot=100.0, expiry=_TARGET,
         obs_date=_OBS, ticker="AAA") -> dict:
    return {"ticker": ticker, "obs_date": obs_date, "expiry": expiry, "strike": strike,
            "right": right, "bid": bid, "ask": ask, "spot": spot}


def _stage(*rows) -> QuoteRowInputs:
    return scan_quote_rows(None, None, _KEY, expiry=_EXPIRY, decision_session=_SESSION,
                           chain_rows=list(rows))


def test_every_surviving_row_carries_the_chain_spot():
    result = _stage(_row(spot=87.5), _row(right="P", strike=105.0, spot=87.5))

    assert result.quote_status == "recorded"
    assert len(result.quote_rows) == 2
    assert [row["chain_spot"] for row in result.quote_rows] == [87.5, 87.5]
    assert all(isinstance(row["chain_spot"], float) for row in result.quote_rows)
    assert result.quote_rows[0]["bid"] == 1.0
    assert result.quote_rows[0]["observed_at"] == _SESSION


def test_the_chain_spot_is_staged_alongside_null_quotes():
    result = _stage(_row(bid=None, ask=None, spot=42.0))

    assert result.quote_rows[0]["bid"] is None
    assert result.quote_rows[0]["ask"] is None
    assert result.quote_rows[0]["chain_spot"] == 42.0


def test_an_integer_or_numeric_string_spot_stages_as_a_float():
    result = _stage(_row(spot=100), _row(right="P", spot="100.0"))

    assert [row["chain_spot"] for row in result.quote_rows] == [100.0, 100.0]


@pytest.mark.parametrize("spot", [
    None, float("nan"), float("inf"), -float("inf"), 0.0, -12.5, True, False,
    "not-a-spot", [],
])
def test_an_unusable_spot_on_a_surviving_row_is_contract_mismatch(spot):
    with pytest.raises(DataError) as exc:
        _stage(_row(spot=spot))
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_surviving_rows_that_disagree_on_the_spot_are_contract_mismatch():
    with pytest.raises(DataError) as exc:
        _stage(_row(spot=100.0), _row(right="P", spot=100.01))
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_the_spot_refusal_never_echoes_a_staged_value():
    with pytest.raises(DataError) as exc:
        _stage(_row(spot=1234.5678), _row(right="P", spot="not-a-spot"))

    assert exc.value.code == "CONTRACT_MISMATCH"
    assert "1234.5678" not in str(exc.value)
    assert "not-a-spot" not in str(exc.value)


def test_a_row_for_another_expiry_is_never_judged_on_its_spot():
    result = _stage(_row(spot=100.0), _row(right="P", spot=100.0),
                    _row(expiry=_OTHER, spot=None), _row(expiry=_OTHER, right="P", spot="junk"))

    assert result.quote_status == "recorded"
    assert [row["expiry"] for row in result.quote_rows] == [_EXPIRY, _EXPIRY]
    assert [row["chain_spot"] for row in result.quote_rows] == [100.0, 100.0]


def test_a_slice_with_no_surviving_row_stays_empty():
    result = _stage(_row(expiry=_OTHER, spot=None), _row(right="P", expiry=_OTHER, spot=0.0))

    assert result == QuoteRowInputs(quote_rows=(), quote_status="empty")


def test_a_staged_row_without_the_spot_column_is_invalid_request():
    row = _row()
    row.pop("spot")
    with pytest.raises(OpsError) as exc:
        _stage(row)
    assert exc.value.code == "INVALID_REQUEST"


def test_the_per_key_scan_stages_the_spot_it_reads(tmp_path):
    chains = [dict(_physical_chain_row("AAA", _OBS, _TARGET, 100.0, right="C"), spot=64.25),
              dict(_physical_chain_row("AAA", _OBS, _TARGET, 105.0, right="P"), spot=64.25)]
    repository, snapshot = _snapshot(
        tmp_path, price={"AAA": [_price_row("AAA", _SESSION)]}, chains=chains)

    result = scan_quote_rows(repository, snapshot, _KEY, expiry="2024-02-16",
                             decision_session=_SESSION)

    assert [row["chain_spot"] for row in result.quote_rows] == [64.25, 64.25]


def test_a_stored_null_spot_fails_the_per_key_scan(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path, price={"AAA": [_price_row("AAA", _SESSION)]},
        chains=[dict(_physical_chain_row("AAA", _OBS, _TARGET, 100.0, right="C"), spot=None)])

    with pytest.raises(DataError) as exc:
        scan_quote_rows(repository, snapshot, _KEY, expiry="2024-02-16",
                        decision_session=_SESSION)
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_a_stored_null_spot_on_another_expiry_does_not_fail_the_scan(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path, price={"AAA": [_price_row("AAA", _SESSION)]},
        chains=[dict(_physical_chain_row("AAA", _OBS, _OTHER, 100.0, right="C"), spot=None),
                _physical_chain_row("AAA", _OBS, _TARGET, 100.0, right="C")])

    result = scan_quote_rows(repository, snapshot, _KEY, expiry="2024-02-16",
                             decision_session=_SESSION)

    assert [row["chain_spot"] for row in result.quote_rows] == [100.0]
