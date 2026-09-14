import type { EventPageItem, EventScoreSummary } from "../api/types";
import { compatibilityLink, fmtNumber, fmtPercent, fmtText, headlineExpectedReturn } from "../format";
import { eventHash } from "../routes";

interface Props {
  items: EventPageItem[];
  releaseId: string;
}

function isRefusal(score: EventScoreSummary): boolean {
  return score.refusal_reason !== null;
}

function ScoreRow({
  item,
  score,
  releaseId,
}: {
  item: EventPageItem;
  score: EventScoreSummary;
  releaseId: string;
}) {
  const refusal = isRefusal(score);
  const headline = headlineExpectedReturn(score);
  return (
    <tr
      className={refusal ? "score-row score-row-refusal" : "score-row"}
      data-testid="score-row"
      data-score-id={score.score_id}
    >
      <td>
        <a href={eventHash(releaseId, item.event_ref.event_id)} data-testid="open-event-link">
          {item.ticker}
        </a>
      </td>
      <td>{item.event_date}</td>
      <td>{fmtText(item.session)}</td>
      <td>{score.strategy}</td>
      <td data-testid="verdict-cell">
        {refusal ? (
          <span className="badge badge-refusal" data-testid="refusal-badge">
            REFUSED — {score.refusal_reason}
          </span>
        ) : (
          // Raw gate_pass, not the legacy gatePill decision tree (ui/README.md
          // gap list) — a compatibility-view link sits in the last column.
          <span data-testid="raw-verdict">gate_pass: {fmtText(score.verdict)}</span>
        )}
      </td>
      <td data-testid="driver-forecast-cell">{fmtPercent(score.driver_forecast)}</td>
      <td>{fmtPercent(score.market_implied_move)}</td>
      <td data-testid="entry-premium-cell">{fmtNumber(score.entry_premium)}</td>
      <td data-testid="expected-return-cell">
        {fmtPercent(headline.value)}
        {headline.sim && (
          <span className="badge badge-sim" data-testid="sim-badge" title="entry-rule simulated expected return, not a payoff-map forecast">
            sim
          </span>
        )}
      </td>
      <td>
        {score.chosen_strategy === null
          ? "—"
          : `${score.chosen_strategy}${
              score.menu_size !== null ? ` (of ${score.menu_size})` : ""
            }${score.chosen_margin !== null ? ` margin ${score.chosen_margin.toFixed(3)}` : ""}`}
      </td>
      <td>
        <a
          href={compatibilityLink(releaseId)}
          target="_blank"
          rel="noreferrer"
          data-testid="compat-link"
        >
          legacy view
        </a>
      </td>
    </tr>
  );
}

/**
 * §7 board columns: "Event date/session, ticker, strategy, verdict/refusal,
 * driver forecast, market implied move, entry premium, available expected-
 * return fields, and DYN-SV choice." Every strategy/refusal row in the
 * accepted population is shown — no passing/non-null filter (§7: "Preserve
 * every strategy/refusal in the accepted population.").
 */
export function EventTable({ items, releaseId }: Props) {
  if (items.length === 0) {
    return (
      <p data-testid="no-matches">No events match the current filters.</p>
    );
  }
  return (
    <table className="event-table" data-testid="event-table">
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Event date</th>
          <th>Session</th>
          <th>Strategy</th>
          <th>Verdict</th>
          <th>Driver forecast</th>
          <th>Market implied move</th>
          <th>Entry premium</th>
          <th>Expected return</th>
          <th>DYN-SV choice</th>
          <th>Legacy</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) =>
          item.scores.length === 0 ? (
            <tr key={`${item.event_ref.event_id}-empty`} data-testid="no-scores-row">
              <td>{item.ticker}</td>
              <td>{item.event_date}</td>
              <td>{fmtText(item.session)}</td>
              <td colSpan={8}>no scores for this event</td>
            </tr>
          ) : (
            item.scores.map((score) => (
              <ScoreRow
                key={score.score_id}
                item={item}
                score={score}
                releaseId={releaseId}
              />
            ))
          ),
        )}
      </tbody>
    </table>
  );
}
