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
