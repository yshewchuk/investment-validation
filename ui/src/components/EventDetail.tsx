import type { DataClient } from "../api/client";
import { compatibilityLink, fmtPercent, fmtNumber, fmtText, headlineExpectedReturn } from "../format";
import { useEventScores } from "../hooks";
import { boardHash, scoreHash } from "../routes";

export interface EventMeta {
  ticker: string;
  event_date: string;
  session: string | null;
}

interface Props {
  client: DataClient;
  releaseId: string;
  eventId: string;
  /** Populated when the event was opened from an already-loaded board page
   * (App.tsx's `eventMetaCache`); null on a fresh deep link, since
   * `GET /events/{id}/scores` (§6) returns score summaries only, no event
   * metadata (judgement call, recorded in ui/README.md). */
  meta: EventMeta | null;
}

/**
 * Guide P3-3b deliverable 1: "open an event from the board, lazily fetch
 * its scores for the pinned release, and list score summaries ... Handle
 * loading, error and empty states."
 */
export function EventDetail({ client, releaseId, eventId, meta }: Props) {
  const state = useEventScores(client, releaseId, eventId);

  return (
    <div className="detail-view" data-testid="event-detail">
      <p className="breadcrumbs">
        <a href={boardHash(releaseId)} data-testid="back-to-board">
          ← Back to board
        </a>
      </p>
      <h2 data-testid="event-detail-header">
        {meta !== null ? `${meta.ticker} — ${meta.event_date}${meta.session ? ` (${meta.session})` : ""}` : `Event ${eventId}`}
      </h2>
      <p className="event-id-line">
        event <code>{eventId}</code>, release <code>{releaseId}</code>
      </p>

      {state.status === "loading" && <p data-testid="event-scores-loading">Loading scores…</p>}

      {state.status === "not_found" && (
        <p data-testid="event-not-found" className="error-banner">
          Unknown event: no scores were found for <code>{eventId}</code> under this release.
        </p>
      )}

      {state.status === "error" && (
        <p data-testid="event-scores-error" className="error-banner">
          Could not load scores: {state.error.message} ({state.error.code}).
        </p>
      )}

      {state.status === "ready" && state.scores.length === 0 && (
        <p data-testid="event-scores-empty">No scores are recorded for this event.</p>
      )}

      {state.status === "ready" && state.scores.length > 0 && (
        <table className="event-table" data-testid="event-scores-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Verdict</th>
              <th>Driver forecast</th>
              <th>Market implied move</th>
              <th>Entry premium</th>
              <th>Expected return</th>
              <th>DYN-SV choice</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {state.scores.map((score) => {
              const refusal = score.refusal_reason !== null;
              const headline = headlineExpectedReturn(score);
              return (
                <tr
                  key={score.score_id}
                  data-testid="event-score-row"
                  data-score-id={score.score_id}
                  className={refusal ? "score-row score-row-refusal" : "score-row"}
                >
                  <td>{score.strategy}</td>
                  <td data-testid="verdict-cell">
                    {refusal ? (
                      <span className="badge badge-refusal" data-testid="refusal-badge">
                        REFUSED — {score.refusal_reason}
                      </span>
                    ) : (
                      <span data-testid="raw-verdict">gate_pass: {fmtText(score.verdict)}</span>
                    )}
                  </td>
                  <td>{fmtPercent(score.driver_forecast)}</td>
                  <td>{fmtPercent(score.market_implied_move)}</td>
                  <td data-testid="entry-premium-cell">{fmtNumber(score.entry_premium)}</td>
                  <td data-testid="expected-return-cell">
                    {fmtPercent(headline.value)}
                    {headline.sim && (
                      <span className="badge badge-sim" data-testid="sim-badge">
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
                      href={scoreHash(releaseId, score.score_id, eventId)}
                      data-testid="open-score-link"
                    >
                      view detail
                    </a>{" "}
                    <a href={compatibilityLink(releaseId)} target="_blank" rel="noreferrer" data-testid="compat-link">
                      legacy view
                    </a>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
