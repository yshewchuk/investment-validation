/**
 * Display-only metadata mirroring `engine/v2/serving/bridge.py`'s
 * `LEGACY_DISPLAY_MAPPING_V1` (`DisplayFieldSpec`: field, source_path, unit,
 * nullable, derived, exact) -- transcribed by hand, field name and unit for
 * field name and unit, from that module's `_BOARD_FIELD_NAMES`/`_UNITS`/
 * `_EXACT_FIELDS`/`_DERIVED_SOURCES`. This is presentation metadata only:
 * a label to group an already-rendered `display_record` field under, and a
 * flag noting the renderer (not this client) computed it. No value is ever
 * read, converted or computed from this table.
 *
 * Guide P3-3b deliverable 2: "show display_record fields grouped by the
 * mapping spec's categories if available, otherwise alphabetically." The
 * bridge's spec has no field literally named "category" -- `unit` is the
 * closest analog (identity/date/usd/ratio/percent/... groupings) and is
 * what this module uses as the category. A `display_record` field this
 * table does not know about (schema drift, or a mock-only field) falls into
 * the synthetic "other" category rather than being dropped, and groupings
 * with no field present in the record are simply omitted.
 *
 * Keep in sync with `engine/v2/serving/bridge.py::LEGACY_DISPLAY_MAPPING_V1`
 * by hand; this file is read-only presentation data outside `engine/`, so a
 * drift here does not affect field identity, only which group heading a
 * value is displayed under.
 */

const OTHER_CATEGORY = "other";

/** Mirrors bridge.py `_UNITS`. */
const CATEGORY_BY_FIELD: Readonly<Record<string, string>> = {
  row_id: "identity",
  ticker: "identity",
  strategy: "identity",
  as_of: "date",
  event_date: "date",
  session: "string",
  entry_date: "date",
  exit_date: "date",
  expiry: "date",
  quote_date: "date",
  forecast_fold: "date",
  model_input_as_of: "date",
  chain_last_obs: "date",
  strike: "usd",
  requested_strike: "usd",
  spot: "usd",
  entry_cost: "usd",
  structure_width: "usd",
  strike_offset: "ratio",
  cost_over_width: "ratio",
  premium_vs_fair: "ratio",
  model_vs_market: "ratio",
  rel_spread: "ratio",
  entry_cost_pct: "percent",
  model_fair_pct: "percent",
  exp_pnl_model: "percent_of_spot",
  exp_pnl_analog: "percent_of_spot",
  exp_pnl_sim: "percent_of_spot",
  model_p10: "percent_of_spot",
  model_p90: "percent_of_spot",
  ci_low: "percent_of_spot",
  ci_high: "percent_of_spot",
  forecast_abs_move: "percent_of_spot",
  forecast_p10: "percent_of_spot",
  forecast_p90: "percent_of_spot",
  forecast_sd: "percent_of_spot",
  driver_prediction: "percent_of_spot",
  driver_p10: "percent_of_spot",
  driver_p90: "percent_of_spot",
  runup_move_prediction: "percent_of_spot",
  runup_move_p10: "percent_of_spot",
  runup_move_p90: "percent_of_spot",
  implied_move: "percent_of_spot",
  implied_move_at_entry: "percent_of_spot",
  gate_score: "raw_score",
  gate_threshold: "raw_score",
  chosen_margin: "raw_score",
  win_model: "probability",
  win_analog: "probability",
  win_sim: "probability",
  n_analogs: "count",
  analog_widened: "count",
  menu_size: "count",
  dte_entry: "days",
  quote_age_sessions: "days",
  quote_max_age_sessions: "days",
  chain_age_days: "days",
  runup_move_days: "days",
  runup_move_scale: "ratio",
  rank: "count",
  gate_pass: "bool",
  extrapolated: "bool",
  scored: "bool",
  forecast_model: "string",
  chosen_strategy: "string",
  driver_name: "string",
  detail: "string",
  digest: "hash",
  legs: "object",
  structure_params: "object",
  model_versions: "object",
  payoff_curve: "object",
  flags: "list",
  fill: "ratio",
};

/** Mirrors bridge.py `_DERIVED_SOURCES` keys -- fields the renderer computed
 * itself rather than copying from the engine record. Informational only: it
 * never changes how a value is displayed, only whether a "derived" note is
 * shown next to the field label. */
const DERIVED_FIELDS: ReadonlySet<string> = new Set([
  "row_id",
  "payoff_curve",
  "cost_over_width",
  "entry_cost_pct",
  "model_fair_pct",
  "premium_vs_fair",
  "model_vs_market",
  "scored",
  "rank",
  "digest",
]);

/** Mirrors bridge.py `_EXACT_FIELDS` -- kept at full precision in the
 * rendered row. Informational only. */
const EXACT_FIELDS: ReadonlySet<string> = new Set([
  "structure_params",
  "requested_strike",
  "strike",
  "strike_offset",
  "fill",
  "quote_max_age_sessions",
]);

export interface DisplayFieldMeta {
  field: string;
  category: string;
  derived: boolean;
  exact: boolean;
  knownField: boolean;
}

export function displayFieldMeta(field: string): DisplayFieldMeta {
  const category = CATEGORY_BY_FIELD[field];
  return {
    field,
    category: category ?? OTHER_CATEGORY,
    derived: DERIVED_FIELDS.has(field),
    exact: EXACT_FIELDS.has(field),
    knownField: category !== undefined,
  };
}

export interface DisplayFieldGroup {
  category: string;
  fields: DisplayFieldMeta[];
}

/**
 * Groups the keys actually present in a `display_record` by category
 * (guide: "grouped by the mapping spec's categories if available, otherwise
 * alphabetically"). Every field is placed in exactly one group -- known
 * fields under their mapped category, unknown ones under "other" -- and
 * fields are sorted alphabetically within each group. Groups are sorted
 * alphabetically, with "other" always last so a genuinely categorized
 * record reads as categorized first.
 */
export function groupDisplayRecordFields(record: Record<string, unknown>): DisplayFieldGroup[] {
  const byCategory = new Map<string, DisplayFieldMeta[]>();
  for (const field of Object.keys(record)) {
    const meta = displayFieldMeta(field);
    const bucket = byCategory.get(meta.category);
    if (bucket) {
      bucket.push(meta);
    } else {
      byCategory.set(meta.category, [meta]);
    }
  }
  const categories = [...byCategory.keys()].sort((a, b) => {
    if (a === OTHER_CATEGORY) return 1;
    if (b === OTHER_CATEGORY) return -1;
    return a.localeCompare(b);
  });
  return categories.map((category) => ({
    category,
    fields: (byCategory.get(category) ?? []).sort((a, b) => a.field.localeCompare(b.field)),
  }));
}
