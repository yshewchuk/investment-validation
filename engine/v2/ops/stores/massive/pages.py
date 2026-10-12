"""Bounded page scope and continuation checks before raw Massive persistence."""
import json
import re
from datetime import date, datetime, time, timedelta, timezone
from urllib.parse import parse_qsl, urlsplit
from zoneinfo import ZoneInfo

from engine.v2.ops import massive_plan
from engine.v2.ops.errors import fail
from engine.v2.ops.providers.massive_policy import MAX_RAW_BYTES

_EASTERN = ZoneInfo("America/New_York")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_CURSOR = re.compile(r"[A-Za-z0-9_-]{1,2048}={0,2}")


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError
    return result


def _day(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError
    return parsed


def _instant(value):
    if type(value) is not int:
        raise ValueError
    return _EPOCH + timedelta(milliseconds=value)


def _lower_bound(request, endpoint):
    prefix = f"/v2/aggs/ticker/{request['ticker']}/range/1/day/"
    match = re.fullmatch(re.escape(prefix) + r"([^/]+)/" + re.escape(request["end"]), endpoint)
    if match is None:
        raise ValueError
    value = match[1]
    if re.fullmatch(r"(?:0|[1-9][0-9]{0,15})", value):
        lower = _instant(int(value))
    else:
        lower = datetime.combine(_day(value), time(), _EASTERN)
    if not request["start"] <= lower.astimezone(_EASTERN).date().isoformat() <= request["end"]:
        raise ValueError
    return lower


def _parse_next_url(url, original):
    if (not isinstance(url, str) or len(url) > 4096
            or re.fullmatch(r"[!-~]+", url) is None or "#" in url):
        raise ValueError
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "api.massive.com":
        raise ValueError
    pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=10)
    query = dict(pairs)
    if (len(query) != len(pairs)
            or _CURSOR.fullmatch(query.get("cursor", "")) is None):
        raise ValueError
    expected = {k: str(v).lower() if type(v) is bool else str(v)
                for k, v in original["params"].items()}
    if any(k != "cursor" and (k not in expected or v != expected[k]) for k, v in pairs):
        raise ValueError
    return parsed.path, query["cursor"]


def _next_request(url, original, current):
    if url is None:
        return None
    endpoint, cursor = _parse_next_url(url, original)
    if original["dataset"] == "daily":
        if _lower_bound(original, endpoint) < _lower_bound(original, current["endpoint"]):
            raise ValueError
    elif endpoint != original["endpoint"]:
        raise ValueError
    result = original | {"endpoint": endpoint, "params": original["params"] | {"cursor": cursor}}
    if result == current:
        raise ValueError
    return result


def _page_scope(original_request, page_request):
    params = page_request["params"]
    cursor = params.get("cursor")
    expected = original_request | {"endpoint": page_request["endpoint"],
                                   "params": original_request["params"] | {"cursor": cursor}}
    if page_request != original_request and (type(cursor) is not str
            or _CURSOR.fullmatch(cursor) is None or page_request != expected):
        raise ValueError
    if any(type(params[k]) is not type(v) for k, v in original_request["params"].items()):
        raise ValueError
    if original_request["dataset"] != "daily" and page_request["endpoint"] != original_request["endpoint"]:
        raise ValueError


def validate_rows(body: bytes, original_request: dict, page_request: dict) -> dict:
    """Validate source envelope and row scope without normalizing price values."""
    if type(body) is not bytes:
        raise fail("SOURCE_INVALID", "Massive response body must be bytes")
    if len(body) > MAX_RAW_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive response exceeds byte policy")
    massive_plan._complete_body(body, original_request)
    try:
        _page_scope(original_request, page_request)
        document = json.loads(body, object_pairs_hook=_unique_object)
        rows = document.get("results", [])
        daily = original_request["dataset"] == "daily"
        if daily:
            lower = _lower_bound(original_request, page_request["endpoint"])
        for row in rows:
            if daily:
                instant = _instant(row["t"])
                day = instant.astimezone(_EASTERN).date().isoformat()
                valid = lower <= instant and day <= original_request["end"]
            else:
                field = "execution_date" if original_request["dataset"] == "splits" else "ex_dividend_date"
                day = _day(row[field]).isoformat()
                valid = (massive_plan._valid_ticker(row.get("ticker"))
                         and row["ticker"].upper() == original_request["ticker"]
                         and original_request["start"] <= day <= original_request["end"])
            if not valid:
                raise ValueError
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        raise fail("SOURCE_INVALID", "Massive page row scope is invalid") from None
    if not rows:
        raise fail("SOURCE_EMPTY", "Massive page has no source rows")
    return document


def validate_page(body: bytes, original_request: dict, page_request: dict) -> dict | None:
    """Return a checked credential-free continuation before any raw persistence."""
    document = validate_rows(body, original_request, page_request)
    try:
        return _next_request(document.get("next_url"), original_request, page_request)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        raise fail("SOURCE_INVALID", "Massive page continuation is invalid") from None


def closed_day(request: dict, observed_at: datetime) -> None:
    """Versioned local daily eligibility only; this is not source finality evidence."""
    if request["dataset"] == "daily":
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise fail("SOURCE_NOT_FINAL", "Massive observation requires an explicit timezone")
        if request["end"] >= observed_at.astimezone(_EASTERN).date().isoformat():
            raise fail("SOURCE_NOT_FINAL", "Massive daily range includes an unclosed Eastern date")
