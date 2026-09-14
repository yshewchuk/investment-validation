/**
 * Payoff-curve geometry: SCALING ONLY. Guide P3-3b deliverable 2: "Render
 * `payoff_curve` as a simple SVG line from the given points only, with no
 * interpolation or computed values." `engine/dashboard/render.py::
 * payoff_curve` already computed the shape (parallel `x`/`y` arrays); this
 * module linearly maps those SAME numbers into pixel space so they can be
 * drawn (guide §7: "The UI may ... map stored chart values to pixels"). It
 * never adds, removes or interpolates a point, and never derives a new
 * financial quantity (no premium, ratio, or payoff arithmetic here).
 */

export interface PayoffCurveData {
  x: number[];
  y: number[];
}

export interface PayoffPixelPoint {
  dataX: number;
  dataY: number;
  pixelX: number;
  pixelY: number;
}

/** True only for a genuine `{x, y}` shape with matching non-empty arrays;
 * anything else (missing field, wrong shape, mismatched lengths) is treated
 * as "no curve to draw" rather than guessed at. */
export function isPayoffCurve(value: unknown): value is PayoffCurveData {
  if (value === null || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return Array.isArray(record.x) && Array.isArray(record.y);
}

/** Pairs the curve's own `x[i]`/`y[i]` values one-to-one, dropping only a
 * trailing length mismatch (never fabricating a missing coordinate), then
 * scales that exact point set into a `width`x`height` pixel box. Returns an
 * empty array for an empty curve -- callers render an explicit "no points"
 * state rather than an empty chart. */
export function payoffPixelPoints(
  curve: PayoffCurveData,
  width: number,
  height: number,
  padding = 6,
): PayoffPixelPoint[] {
  const n = Math.min(curve.x.length, curve.y.length);
  const pairs: [number, number][] = [];
  for (let i = 0; i < n; i += 1) {
    const xv = curve.x[i];
    const yv = curve.y[i];
    if (xv === undefined || yv === undefined) continue;
    pairs.push([xv, yv]);
  }
  if (pairs.length === 0) return [];

  const xs = pairs.map(([x]) => x);
  const ys = pairs.map(([, y]) => y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  const innerW = Math.max(width - 2 * padding, 1);
  const innerH = Math.max(height - 2 * padding, 1);

  return pairs.map(([dataX, dataY]) => ({
    dataX,
    dataY,
    pixelX: padding + ((dataX - xMin) / xSpan) * innerW,
    pixelY: height - padding - ((dataY - yMin) / ySpan) * innerH,
  }));
}

export function pointsAttribute(points: PayoffPixelPoint[]): string {
  return points.map((p) => `${p.pixelX.toFixed(2)},${p.pixelY.toFixed(2)}`).join(" ");
}
