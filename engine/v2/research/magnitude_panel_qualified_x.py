"""HMOSX block descriptors; declared only, appended last, nothing computed."""

TAIL = ("tail_max_abs", "tail_std_abs", "tail_p75_abs", "tail_p90_abs",
        "tail_share_gt10", "tail_max4_abs", "tail_std_over_mean", "tail_max_over_mean")
IVDYN = ("iv10_30", "iv_rv", "ex_ratio", "imp_chg5", "imp_chg10", "iv30_chg5",
         "iv30_chg20", "iv10_chg5")
OWNVOL = ("own_rvol5", "own_rvol10", "own_rvol60", "own_absret1", "own_rvol5_60")
CALENDAR = ("cal_amc", "cal_weekday", "cal_month", "cal_days_since_prior")
X_COLUMNS = (*TAIL, *IVDYN, *OWNVOL, *CALENDAR)

_SOURCE_DEPS = {
    "tail": ("stored_history", {"schema_version": "tail_history_source.v1",
        "reader": "history_json_contributors",
        "causal": "available_as_of_date <= decision_date", "own_ticker": "only"}),
    "ivdyn": ("daily_market", {"schema_version": "iv_dynamics_source.v1",
        "reader": "daily_market_surface_rows", "causal": "date <= source_session_ceiling",
        "stale_reference": "missing_if_reference_older_than_2k_plus_4_calendar_days"}),
    "ownvol": ("price_history", {"schema_version": "own_vol_source.v1",
        "reader": "price_history_close_adj", "causal": "date <= source_session_ceiling",
        "stale_last_close": "missing_if_older_than_5_calendar_days"}),
    "calendar": ("event_calendar", {"schema_version": "calendar_features_source.v1",
        "reader": "stored_event_session_and_date",
        "causal": "event_date_and_session_define_forecast_clock"}),
}
_DERIVATIONS = {
    "tail": ("tail_history.v1", "compute_tail_history"),
    "ivdyn": ("iv_dynamics.v1", "compute_iv_dynamics"),
    "ownvol": ("own_vol.v1", "compute_own_vol"),
    "calendar": ("calendar_features.v1", "compute_calendar_features"),
}
_FIELDS = (
    ("tail_max_abs", "tail", "float64", "percent", True, "null"),
    ("tail_std_abs", "tail", "float64", "percent", True, "null"),
    ("tail_p75_abs", "tail", "float64", "percent", True, "null"),
    ("tail_p90_abs", "tail", "float64", "percent", True, "null"),
    ("tail_share_gt10", "tail", "float64", "fraction", True, "null"),
    ("tail_max4_abs", "tail", "float64", "percent", True, "null"),
    ("tail_std_over_mean", "tail", "float64", "ratio", True, "null"),
    ("tail_max_over_mean", "tail", "float64", "ratio", True, "null"),
    ("iv10_30", "ivdyn", "float64", "ratio", True, "null"),
    ("iv_rv", "ivdyn", "float64", "ratio", True, "null"),
    ("ex_ratio", "ivdyn", "float64", "ratio", True, "null"),
    ("imp_chg5", "ivdyn", "float64", "ratio", True, "null"),
    ("imp_chg10", "ivdyn", "float64", "ratio", True, "null"),
    ("iv30_chg5", "ivdyn", "float64", "ratio", True, "null"),
    ("iv30_chg20", "ivdyn", "float64", "ratio", True, "null"),
    ("iv10_chg5", "ivdyn", "float64", "ratio", True, "null"),
    ("own_rvol5", "ownvol", "float64", "percent", True, "null"),
    ("own_rvol10", "ownvol", "float64", "percent", True, "null"),
    ("own_rvol60", "ownvol", "float64", "percent", True, "null"),
    ("own_absret1", "ownvol", "float64", "percent", True, "null"),
    ("own_rvol5_60", "ownvol", "float64", "ratio", True, "null"),
    ("cal_amc", "calendar", "int64", "flag", False, "refuse"),
    ("cal_weekday", "calendar", "int64", "weekday", False, "refuse"),
    ("cal_month", "calendar", "int64", "month", False, "refuse"),
    ("cal_days_since_prior", "calendar", "int64", "days", True, "null"),
)


def _dependencies(block, name):
    role, policy = _SOURCE_DEPS[block]
    dependencies = [{"role": role, "policy": dict(policy)}]
    if name == "cal_days_since_prior":
        tail_role, tail_policy = _SOURCE_DEPS["tail"]
        dependencies.append({"role": tail_role, "policy": dict(tail_policy)})
    return dependencies


def x_fields() -> list[dict]:
    """Return fresh HMOSX descriptors in X_COLUMNS order."""
    fields = []
    for name, block, physical_type, unit, nullable, on_missing in _FIELDS:
        schema_version, function = _DERIVATIONS[block]
        fields.append({"name": name, "block": block, "source_alias": name,
            "native_output_alias": name, "physical_type": physical_type, "unit": unit,
            "nullable": nullable, "on_missing": on_missing,
            "source_dependencies": _dependencies(block, name),
            "derivation_policy": {"schema_version": schema_version,
                "function": function, "output": name}})
    return fields
