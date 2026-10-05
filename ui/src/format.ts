/**
 * Presentation-only helpers. Guide §7: "The UI may format units/dates/
 * percentages ... It cannot compute fair premium, ratios, gates, expected
 * returns, strategy selection or book totals." Nothing here derives a new
 * number from two others — every function takes one already-rendered value
 * and returns a string or a CSS class.
 */

/** Null is unavailable and renders as an em dash; a real zero renders as "0". */
export function fmtNumber(value: number | null, digits = 2): string {
  if (value === null) {
    return "—";
  }
  return value.toFixed(digits);
}

export function fmtPercent(value: number | null, digits = 1): string {
  if (value === null) {
    return "—";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

export function fmtText(value: string | null): string {
  return value === null ? "—" : value;
}

/**
 * For values that ALREADY arrive in percentage points (e.g. a 5.0 meaning
 * "5.0 percentage points"): print the given number with one decimal place
 * followed by `%`. The value is never multiplied by 100 — that scaling is
 * `fmtPercent`'s job for fractional ratios.
 */
export function fmtPercentagePoint(value: number | null, digits = 1): string {
  if (value === null) {
    return "—";
  }
  return `${fmtNumber(value, digits)}%`;
}

/** Coverage-summary keys that are whole-number counts, not fractional ratios. */
const COVERAGE_COUNT_KEYS: ReadonlySet<string> = new Set([
  "planned_population",
  "compared_population",
]);

/**
 * One `coverage_summary` entry from the serving release: count keys
 * (`planned_population`, `compared_population`) render as whole numbers with
 * no `%`; every other entry is a fractional ratio and renders via
 * `fmtPercent` (fraction × 100). Null is "missing" either way.
 */
export function fmtCoverageEntry(key: string, value: number | null): string {
  if (value === null) {
    return "—";
  }
  if (COVERAGE_COUNT_KEYS.has(key)) {
    return fmtNumber(value, 0);
  }
  return fmtPercent(value);
}

/** Builds the immutable compatibility-surface link (§6): `/release/<id>/...`. */
export function compatibilityLink(releaseId: string, path = "index.html"): string {
  return `/release/${encodeURIComponent(releaseId)}/${path}`;
}

export interface Headline {
  value: number | null;
  sim: boolean;
}

/**
 * Ports the legacy board's headline expected-return choice, display-only —
 * NOT a financial computation. Exact source:
 * `engine/dashboard/static/assets/app.js::pnlCell` (~line 487): show
 * `exp_pnl_model` if present, else `exp_pnl_sim`, **never** `exp_pnl_analog`.
 * `EventScoreSummary.expected_return` is always null (v1.1); this picks
 * between the two honest per-producer reads the summary actually carries
 * (`expected_return_model`/`expected_return_sim`) the same way the legacy
 * cell does, so the client shows one headline number without inventing one.
 */
/**
 * Generic `display_record` value formatting for the score-detail view
 * (guide P3-3b deliverable 2). Every `display_record` field is an
 * already-rendered legacy value of unknown shape (`Record<string,
 * unknown>`) — this only decides how to print a value that is already
 * there; it never combines two fields or computes a new one. Null/absent is
 * "missing", never blank-as-zero (guide: "Nulls show as missing, never
 * 0."); a real boolean `false` or number `0` prints as given.
 */
export function fmtUnknown(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "number") {
    return String(value);
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function headlineExpectedReturn(score: {
  expected_return_model: number | null;
  expected_return_sim: number | null;
}): Headline {
  if (score.expected_return_model !== null) {
    return { value: score.expected_return_model, sim: false };
  }
  if (score.expected_return_sim !== null) {
    return { value: score.expected_return_sim, sim: true };
  }
  return { value: null, sim: false };
}
