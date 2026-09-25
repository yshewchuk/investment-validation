"""Native ORATS market-wide fetcher for the ``daily_market`` refresh slice.

One ``incremental_refresh`` fetch unit denominates one ORATS ``tradeDate``, and
one unit costs TWO provider calls -- ``hist/summaries`` and ``hist/cores`` --
because the daily_market row the merge consumes needs fields from both. This
module is the only v2 network edge for daily_market acquisition: the closure it
returns is the ``fetcher(unit)`` seam ``run_daily_market_refresh`` documents,
and ``http_get`` is its sole test seam.

The row builder ports the per-row field mapping of
``engine/data/normalize/n_daily.py::normalize_ticker`` (ORATS decimal IVs to
vol points, the three-era ``mktCap`` conversion, the ``src_*`` provenance
columns) verbatim rather than importing that legacy module. It does not walk
back over recent sessions: a market-wide date that is not published yet (404 on
both endpoints, or a 2xx with no rows) is a ``SOURCE_NOT_FINAL`` refusal so the
job's own retry policy owns that decision, and the unused ``lookback_days``
parameter is kept for a future caller that plans one unit per candidate date.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Mapping, Sequence

from engine.v2.foundation import canonical_json
from engine.v2.ops.errors import fail
from engine.v2.ops.incremental_data import classify_response

__all__ = ["SUMMARY_FIELDS", "orats_daily_market_fetcher"]

TABLE_NAME = "daily_market"
ORATS_BASE_URL = "https://api.orats.io/datav2"
SUMMARIES_ENDPOINT = "hist/summaries"
CORES_ENDPOINT = "hist/cores"
REQUEST_TIMEOUT_SECONDS = 30.0
SENTINEL_THRESHOLD = 1e30

#: raw ORATS key -> (daily_market column, multiplier), ported from
#: ``engine/data/normalize/n_daily.py``'s SUMMARY_FIELDS. The multiplier turns
#: decimals into vol points / percent, the panel's unit.
SUMMARY_FIELDS: dict[str, tuple[str, float]] = {
    "stockPrice": ("spot", 1.0),
    "iv10d": ("iv10", 100.0),
    "iv30d": ("iv30", 100.0),
    "exErnIv10d": ("exern_iv10", 100.0),
    "exErnIv30d": ("exern_iv30", 100.0),
    "impliedMove": ("implied_move", 100.0),
    "rVol30": ("rvol30", 100.0),
    "skewing": ("skew", 1.0),
    "contango": ("contango", 1.0),
    "fwd90_30": ("fwd90_30", 100.0),
    "fexErn90_30": ("fexern90_30", 100.0),
    "ieeEarnEffect": ("iee", 1.0),
}

#: Outcome kind -> the registered failure code whose ops status the refusal
#: becomes (``_failure_for_refresh_status`` is the reverse table).
_FAILURE_CODE_BY_KIND = {
    "partial": "TRANSIENT_SOURCE",
    "unsupported": "TRANSIENT_SOURCE",
    "not_final": "SOURCE_NOT_FINAL",
    "credential_invalid": "CREDENTIAL_INVALID",
    "rate_limited": "RATE_LIMITED",
    "transient": "TRANSIENT_SOURCE",
}
#: Higher is worse, so two endpoint outcomes collapse to the worse code.
_FAILURE_SEVERITY = {
    "TRANSIENT_SOURCE": 0,
    "SOURCE_NOT_FINAL": 1,
    "RATE_LIMITED": 2,
    "CREDENTIAL_INVALID": 3,
}


def orats_daily_market_fetcher(*, http_get: Callable[..., tuple] | None = None,
                               api_key: str | None = None,
                               lookback_days: int = 6) -> Callable[[dict], tuple]:
    """Return a ``fetcher(unit)`` callable for ``run_daily_market_refresh``.

    ``api_key`` is resolved lazily inside the closure: an explicit argument
    wins, otherwise ``ORATS_API_KEY`` is read from the environment at call time
    (matching ``OratsAdapter.__init__``). ``lookback_days`` is accepted and
    recorded on the closure but unused here: it exists so a future caller can
    plan one ``RefreshUnit`` per candidate session date and try the newest
    first. This fetcher never walks back inside one unit.
    """

    def fetcher(unit):
        if str(unit.get("table_name", "")) != TABLE_NAME:
            raise fail("INVALID_REQUEST", "orats fetcher only serves daily_market")
        token = api_key if api_key is not None else os.environ.get("ORATS_API_KEY")
        if not token:
            raise fail("CREDENTIAL_INVALID", "ORATS_API_KEY is unset")
        request = http_get or _requests_get
        session_date = str(unit["partition_key"])
        summaries_status, _, summaries_body = _fetch_endpoint(
            request, SUMMARIES_ENDPOINT, session_date, token)
        cores_status, _, cores_body = _fetch_endpoint(
            request, CORES_ENDPOINT, session_date, token)
        summaries_kind = _classify(unit, summaries_status, summaries_body).kind
        cores_kind = _classify(unit, cores_status, cores_body).kind
        response_kind = _overall_kind(summaries_kind, cores_kind, session_date)
        raw_bytes = canonical_json({
            "summaries": _json_document(summaries_body),
            "cores": _json_document(cores_body),
        }).encode()
        response_meta = {"summaries_status": int(summaries_status),
                         "cores_status": int(cores_status), "trade_date": session_date}
        ticker_rows = _merge_ticker_rows(
            _data_rows(summaries_body), _data_rows(cores_body),
            expected_keys=tuple(str(key) for key in unit.get("expected_keys", ())))
        return raw_bytes, response_kind, response_meta, ticker_rows

    fetcher.lookback_days = lookback_days
    # S4B2: the cache-only completion path in the data layer rebuilds
    # ticker_rows from an already-published raw payload and must merge
    # summaries/cores exactly as a live fetch does. It cannot import this ops
    # module (data never imports ops), so the merge callable rides the
    # fetcher closure the ops layer already injects through
    # ``incremental_data._load_data_refresh_callback``.
    fetcher.merge_ticker_rows = _merge_ticker_rows
    return fetcher


def _requests_get(url: str, *, timeout: float) -> tuple[int, dict, bytes]:
    """The default network client: one plain GET; ORATS auth is the query token."""
    import requests

    response = requests.get(url, timeout=timeout)
    return response.status_code, dict(response.headers), response.content


def _fetch_endpoint(http_get: Callable[..., tuple], endpoint: str,
                    session_date: str, api_key: str) -> tuple[int, dict, bytes]:
    url = _orats_url(endpoint, {"tradeDate": session_date}, api_key)
    return http_get(url, timeout=REQUEST_TIMEOUT_SECONDS)


def _orats_url(endpoint: str, params: Mapping[str, Any], api_key: str) -> str:
    query = "&".join(f"{key}={value}" for key, value in params.items())
    return f"{ORATS_BASE_URL}/{endpoint}?{query}&token={api_key}"


def _json_document(body: Any) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _data_rows(body: Any) -> list[dict]:
    document = _json_document(body)
    data = document.get("data") if isinstance(document, dict) else None
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _provider_tickers(body: Any) -> tuple[str, ...]:
    return tuple(str(row["ticker"]) for row in _data_rows(body) if row.get("ticker"))


def _classify(unit: dict, status: int, body: Any):
    expected = tuple(str(key) for key in unit.get("expected_keys", ()))
    present = set(_provider_tickers(body))
    request_id = str(unit.get("request_id", ""))
    if 200 <= status < 300 and not present:
        # A published date always carries market rows; an empty 2xx is not final.
        return classify_response(status, expected, final=False, request_id=request_id)
    if status == 404:
        # _response_kind checks 404 before final, so pass 200 to force not_final.
        return classify_response(200, expected, final=False, request_id=request_id)
    returned = tuple(key for key in expected if key in present)
    empty = tuple(key for key in expected if key not in present)
    return classify_response(status, expected, returned_keys=returned,
                             empty_keys=empty, final=True, request_id=request_id)


def _overall_kind(summaries_kind: str, cores_kind: str, trade_date: str) -> str:
    if summaries_kind == cores_kind == "complete":
        return "complete"
    if summaries_kind == cores_kind == "empty":
        return "legitimate_empty"
    if summaries_kind == cores_kind == "not_final":
        raise fail("SOURCE_NOT_FINAL",
                   f"ORATS has not published tradeDate={trade_date} yet")
    raise fail(_failure_code(summaries_kind, cores_kind),
               f"orats daily_market response was not complete "
               f"(summaries={summaries_kind}, cores={cores_kind})")


def _failure_code(*kinds: str) -> str:
    worst = "TRANSIENT_SOURCE"
    for kind in kinds:
        code = _FAILURE_CODE_BY_KIND.get(kind, "TRANSIENT_SOURCE")
        if _FAILURE_SEVERITY[code] > _FAILURE_SEVERITY[worst]:
            worst = code
    return worst


def _merge_ticker_rows(summaries_rows: Sequence[Mapping[str, Any]],
                       cores_rows: Sequence[Mapping[str, Any]],
                       expected_keys: Sequence[str] | None = None) -> list[dict]:
    summaries = _rows_by_ticker(summaries_rows)
    cores = _rows_by_ticker(cores_rows)
    allowed = None if expected_keys is None else {str(key) for key in expected_keys}
    rows = []
    for ticker in sorted(set(summaries) | set(cores)):
        if allowed is not None and ticker not in allowed:
            continue
        summary, core = summaries.get(ticker), cores.get(ticker)
        session_date = _trade_date(summary) or _trade_date(core)
        if session_date:
            rows.append(_daily_market_row(ticker, session_date, summary, core))
    return rows


def _rows_by_ticker(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("ticker"):
            indexed[str(row["ticker"])] = row
    return indexed


def _trade_date(row: Mapping[str, Any] | None) -> str | None:
    value = row.get("tradeDate") if row else None
    return str(value)[:10] if value else None


def _daily_market_row(ticker: str, session_date: str,
                      summary: Mapping[str, Any] | None,
                      core: Mapping[str, Any] | None) -> dict:
    row = {"ticker": ticker, "date": session_date, "year": int(str(session_date)[:4])}
    for raw_key, (column, multiplier) in SUMMARY_FIELDS.items():
        row[column] = _scaled_number(summary, raw_key, multiplier)
    row["src_iv"] = "orats.summaries"
    market_cap = _market_cap(core, session_date)
    row["mcap_usd"] = market_cap
    row["mcap_log"] = math.log(market_cap) if market_cap is not None else None
    row["mcap_asof"] = session_date if market_cap is not None else None
    row["mcap_age_days"] = 0.0 if market_cap is not None else None
    row["implied_reconstructed"] = False
    row["src_spot"] = row["src_iv"]
    row["src_mcap"] = "orats.cores" if market_cap is not None else None
    return row


def _scaled_number(source: Mapping[str, Any] | None, key: str,
                   multiplier: float) -> float | None:
    if not source:
        return None
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or abs(number) >= SENTINEL_THRESHOLD:
        return None
    return number * multiplier


def _market_cap(core: Mapping[str, Any] | None, session_date: str) -> float | None:
    if not core:
        return None
    value = core.get("mktCap")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number * _mcap_multiplier(session_date)


def _mcap_multiplier(session_date: str) -> float:
    day = str(session_date)[:10]
    if day >= "2026-03-11":
        return 1e3
    if day >= "2017-06-28":
        return 1e6
    return 1e9