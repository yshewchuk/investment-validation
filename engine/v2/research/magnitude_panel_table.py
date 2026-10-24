"""Closed stage-2 research panel schema; no feature production or source reads."""
import json
import math
import re
from dataclasses import replace

from engine.v2.contracts.data import ColumnContract, TableContract
from engine.v2.data import errors
from engine.v2.data.manifests import table_contract_hash
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.foundation.experiment_holdouts import calendar_date
from engine.v2.research.magnitude_history_store import validate_features
from engine.v2.research.magnitude_history_table import CONTRACT as HISTORY_CONTRACT
from engine.v2.research.magnitude_history_table import FEATURES as HISTORY
from engine.v2.research.progress import Progress, stage

TABLE = "research_magnitude_panel"
BLOCKS = {
    "history": HISTORY,
    "prior_events": ("signed_streak", "ema12r_abs"),
    "runup": ("dist_high", "dist_ema", "ret5", "ret10", "ret20"),
    "spy": tuple("spy_" + name for name in (
        "ret21", "ret63", "ret252", "dd252", "vol5", "vol20", "vol60", "vol252", "vol20_rel252")),
    "daily_market": ("implied_move", "iv10", "iv30", "exern_iv10", "exern_iv30", "iee",
        "skew", "contango", "fwd90_30", "fexern90_30", "rvol30", "spot", "mcap_log"),
}
FEATURES = tuple(name for names in BLOCKS.values() for name in names)
SETS = {"H": HISTORY, "HM": FEATURES[:-13], "HMO": FEATURES}
METADATA = ("schema_version", "event_id", "context_id", "history_context_id", "horizon_sessions",
            "event_date", "cutoff_id", "source_session_ceiling", "lineage", "row_id")
_UNITS = {"signed_streak": "count", "spy_vol20_rel252": "ratio", "spot": "usd",
          "mcap_log": "log_usd", "iee": "unknown", "skew": "unknown", "contango": "unknown"}


def feature_dictionary(subset="HMO"):
    """Fixed ordered aliases: native history units, declared overrides, otherwise percent."""
    if not isinstance(subset, str) or subset not in SETS:
        raise errors.fail("CONTRACT_MISMATCH", "unknown panel feature subset")
    history = {column.name: column for column in HISTORY_CONTRACT.columns}
    return [{"name": name, "block": block, "source_alias": name,
             "native_output_alias": "im" if name == "implied_move" else name,
             "physical_type": history[name].physical_type if block == "history" else "float64",
             "unit": history[name].unit if block == "history" else _UNITS.get(name, "percent"),
             "nullable": name != "n_prior", "on_missing": "zero_count" if name == "n_prior" else "null"}
            for block, names in BLOCKS.items() for name in names if name in SETS[subset]]


def recipe(subset="HMO"):
    """Indicators are selected-column-only; quote status is not an H/HM predictor."""
    dictionary = feature_dictionary(subset)
    return {"schema_version": TABLE + ".v1", "subset": subset, "dictionary": dictionary,
            "dictionary_hash": content_hash(dictionary), "imputation": "none",
            "missing_indicators": [f["name"] for f in dictionary if f["nullable"]],
            "quote_indicator": subset == "HMO"}


def _contract():
    columns = tuple(ColumnContract(name=name,
        physical_type="int64" if name == "horizon_sessions" else "string", nullable=False)
        for name in METADATA)
    columns += tuple(ColumnContract(name=f["name"], physical_type=f["physical_type"],
        nullable=f["nullable"], unit=f["unit"], null_policy=f["on_missing"]) for f in feature_dictionary())
    contract = replace(HISTORY_CONTRACT, contract_id=TABLE + ".v1", table_name=TABLE,
        columns=columns, definition_hash="sha256:" + "0" * 64,
        provenance_semantics="explicit history binding; per-block source and availability lineage",
        finality_semantics="immutable research reconstruction; session dates are not publication proof")
    return replace(contract, definition_hash=table_contract_hash(contract))


CONTRACT: TableContract = _contract()


def canonical_day(value):
    """Require a canonical date string without coercion or timezone interpretation."""
    try:
        if not isinstance(value, str) or calendar_date(value).isoformat() != value:
            raise ValueError
    except (ValueError, TypeError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "panel dates must be canonical date strings") from None
    return value


def _source(source, ceiling):
    """One explicit contributing observation; source date is not publication evidence."""
    if (type(source) is not dict or set(source) != {"source_date", "available_as_of",
            "knowledge_mode", "row_id", "fragment_id", "content_hash"}
            or any(not isinstance(source[k], str) or not source[k].strip()
                   for k in ("row_id", "fragment_id", "content_hash"))
            or re.fullmatch(r"sha256:[0-9a-f]{64}", source["content_hash"]) is None):
        raise ValueError
    day, available = canonical_day(source["source_date"]), source["available_as_of"]
    if (day > ceiling or (available is not None and not day <= canonical_day(available) <= ceiling)
            or source["knowledge_mode"] not in ("observed", "reconstructed")
            or (source["knowledge_mode"] == "observed" and available is None)):
        raise ValueError
    return source["row_id"], source["fragment_id"]


def _block(item, block, row):
    """Associate each anchor with its row/fragment; disallow ambiguous duplicate lineage."""
    if set(item) != {"sources", "missing", "quote_status"} or type(item["sources"]) is not list:
        raise ValueError
    identities = [_source(source, row["source_session_ceiling"]) for source in item["sources"]]
    if len(set(identities)) != len(identities):
        raise ValueError
    names = BLOCKS[block]
    if any(row[name] is not None for name in names) and not identities:
        raise ValueError
    if (type(item["missing"]) is not dict
            or set(item["missing"]) != {name for name in names if row[name] is None}
            or any(reason not in ("unavailable", "insufficient_history", "late_capture", "no_quote")
                   for reason in item["missing"].values())):
        raise ValueError
    _quote(item["quote_status"], block, row["implied_move"])


def _quote(quote, block, implied):
    """Keep absent/zero quote provenance separate from an observed positive quote."""
    if block != "daily_market":
        if quote != "not_applicable":
            raise ValueError
    elif (quote not in ("present", "zero", "absent", "unavailable")
            or (quote == "present") != (implied is not None)):
        raise ValueError


_MAX_LINEAGE_BYTES = 1024 * 1024


def _lineage(row):
    """Validate every block, with explicit unavailable values and quote status."""
    lineage = row["lineage"]
    if (type(lineage) is not str or len(lineage) > _MAX_LINEAGE_BYTES
            or len(lineage.encode("utf-8")) > _MAX_LINEAGE_BYTES):
        raise ValueError
    blocks = json.loads(lineage)
    if set(blocks) != set(BLOCKS) or canonical_json(blocks) != row["lineage"]:
        raise ValueError
    for block in BLOCKS:
        _block(blocks[block], block, row)
        if block in ("history", "prior_events") and any(
                source["knowledge_mode"] != "reconstructed" for source in blocks[block]["sources"]):
            raise ValueError


def _values(row, dictionary):
    """Closed finite numeric types; never convert a missing quote to numeric zero."""
    validate_features(row, row["n_prior"])
    for field in dictionary:
        value = row[field["name"]]
        if value is None:
            if not field["nullable"]:
                raise ValueError
        elif (type(value) not in (int, float) or not math.isfinite(value)
              or (field["physical_type"] == "int64" and (type(value) is not int or value < 0))):
            raise ValueError
    if row["implied_move"] is not None and row["implied_move"] <= 0:
        raise ValueError


def validate_rows(rows, context):
    """Pure complete-cohort contract check, not authority to read or publish rows."""
    from engine.v2.research.magnitude_panel_plan import validate_context

    validate_context(context)
    expected = {(event["event_id"], cutoff["horizon_sessions"]): (event, cutoff)
        for event in context["history_context"]["temporal_plan"]["events"] for cutoff in event["forecasts"]}
    seen = set()
    with stage("panel rows", report_refusals=True):
        try:
            progress = Progress("panel rows", len(expected))
            dictionary = feature_dictionary()
            for row in rows:
                key = row["event_id"], row["horizon_sessions"]
                if (set(row) != set(METADATA) | set(FEATURES) or key in seen or key not in expected
                        or type(row["horizon_sessions"]) is not int):
                    raise ValueError
                event, cutoff = expected[key]
                pins = {"schema_version": CONTRACT.contract_id, "context_id": context["context_id"],
                        "history_context_id": context["history_context"]["context_id"],
                        "event_date": event["event_date"], "cutoff_id": cutoff["cutoff_id"],
                        "source_session_ceiling": cutoff["source_session_ceiling"]}
                if any(row[name] != value for name, value in pins.items()):
                    raise ValueError
                _values(row, dictionary)
                _lineage(row)
                if row["row_id"] != content_hash({k: v for k, v in row.items() if k != "row_id"}):
                    raise ValueError
                seen.add(key)
                progress.tick()
            if seen != set(expected):
                raise ValueError
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
            raise errors.fail("CONTRACT_MISMATCH", "research panel rows differ from their closed contract") from None
