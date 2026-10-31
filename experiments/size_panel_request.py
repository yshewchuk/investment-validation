"""Pure closed panel wire validation shared by SIZE binding and IV previews."""
import sys
from datetime import date

from engine.v2.ops.errors import fail
from engine.v2.research.eligibility_policy import eligibility_policy


def validate_panel_request(value):
    """Closed local wire envelope only; installed readers retain source admission."""
    try:
        strings = {"feature_snapshot_id", "history_snapshot", "price_history_snapshot",
                   "as_of_month", "history_start", "source_start"}
        limits = {"limits": {"max_source_rows", "max_fragments", "max_object_bytes", "max_total_bytes", "max_output_rows"},
            "metadata_limits": {"max_catalog_tables", "max_catalog_fragments", "max_catalog_bytes",
                "max_fragments", "max_partitions", "max_tickers", "max_scan_rows", "max_object_bytes",
                "max_total_object_bytes", "max_batch_rows", "max_retained_rows",
                "max_retained_metadata_bytes", "max_proof_bytes"}}
        if (type(value) is not dict or set(value) != strings | set(limits) | {"policy", "event_ids", "horizons"}
                or any(type(value[k]) is not str or not value[k] or value[k] != value[k].strip() for k in strings)):
            raise ValueError
        for day in (value["history_start"], value["source_start"], value["as_of_month"] + "-01"):
            if date.fromisoformat(day).isoformat() != day:
                raise ValueError
        events, horizons = value["event_ids"], value["horizons"]
        if (type(events) is not list or not events or any(type(e) is not str or not e or e != e.strip() for e in events)
                or len(set(events)) != len(events) or type(horizons) is not list or horizons != [1, 3, 5]
                or any(type(h) is not int for h in horizons)
                or type(value["policy"]) is not dict or type(value["policy"].get("embargo_days")) is not int
                or value["policy"] != eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)):
            raise ValueError
        for key, fields in limits.items():
            if (type(value[key]) is not dict or set(value[key]) != fields
                    or any(type(n) is not int or not 0 < n < sys.maxsize for n in value[key].values())):
                raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid closed SIZE panel request") from None
    return value
