import { isPayoffCurve, payoffPixelPoints, pointsAttribute } from "../payoff";

interface Props {
  value: unknown;
}

const WIDTH = 320;
const HEIGHT = 160;

/**
 * Renders `display_record.payoff_curve` as a plain SVG polyline from the
 * curve's own `x`/`y` arrays only — no interpolation, no computed values
 * (guide P3-3b deliverable 2). Three distinct states, never conflated:
 * missing (field absent/null — no curve was saved for this score), empty
 * (a real curve object with zero points), and a drawn curve.
 */
export function PayoffChart({ value }: Props) {
  if (!isPayoffCurve(value)) {
    return (
      <p data-testid="payoff-missing">No payoff curve is available for this score.</p>
    );
  }
  const points = payoffPixelPoints(value, WIDTH, HEIGHT);
  if (points.length === 0) {
    return <p data-testid="payoff-empty">The payoff curve has no points.</p>;
  }
  return (
    <svg
      data-testid="payoff-svg"
      width={WIDTH}
      height={HEIGHT}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="img"
      aria-label="payoff curve"
    >
      <polyline points={pointsAttribute(points)} fill="none" stroke="#345" strokeWidth={1.5} />
      {points.map((p, i) => (
        <circle
          key={`${p.dataX}-${p.dataY}-${i}`}
          data-testid="payoff-point"
          data-x={p.dataX}
          data-y={p.dataY}
          cx={p.pixelX}
          cy={p.pixelY}
          r={2}
          fill="#345"
        />
      ))}
    </svg>
  );
}
