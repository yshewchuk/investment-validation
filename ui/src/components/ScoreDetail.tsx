import type { DataClient } from "../api/client";
import type { LegacyScoreBridge } from "../api/types";
import { compatibilityLink, fmtUnknown } from "../format";
import { useScoreDetail } from "../hooks";
import { boardHash, eventHash } from "../routes";
import { DisplayRecordFields } from "./DisplayRecordFields";
import { PayoffChart } from "./PayoffChart";

interface Props {
  client: DataClient;
  releaseId: string;
  scoreId: string;
  /** Present when reached via the event-detail flow (board → event →
   * score); null on a bare `#/release/<id>/scores/<id>` deep link. When
   * null but the fetched bridge names its own `event_ref.event_id`, that
   * is used instead — the bridge always knows which event it belongs to
   * even when the URL that opened it did not. */
  eventId: string | null;
}

/**
 * Guide P3-3b deliverable 2: "lazily fetch the score and show
 * `display_record` fields grouped by the mapping spec's categories ...
 * Render `payoff_curve` as a simple SVG line ... Put `engine_record` in a
 * collapsed 'engine evidence' section as raw JSON. Show the score_id and
 * release_id." Display rule (Facts): "The legacy gatePill tree isn't
 * reproduced. Show the raw verdict plus a compatibility link."
 */
export function ScoreDetail({ client, releaseId, scoreId, eventId }: Props) {
  const state = useScoreDetail(client, releaseId, scoreId);
  const backEventId =
    eventId ?? (state.status === "ready" ? state.score.event_ref.event_id : null);

  return (
    <div className="detail-view" data-testid="score-detail">
      <p className="breadcrumbs">
        <a href={boardHash(releaseId)} data-testid="back-to-board">
          ← Back to board
        </a>
        {backEventId !== null && (
          <>
            {" "}
            |{" "}
            <a href={eventHash(releaseId, backEventId)} data-testid="back-to-event">
              ← Back to event
            </a>
          </>
        )}
      </p>

      {state.status === "loading" && <p data-testid="score-loading">Loading score…</p>}

      {state.status === "not_found" && (
        <p data-testid="score-not-found" className="error-banner">
          Unknown score: <code>{scoreId}</code> was not found under this release.
        </p>
      )}

      {state.status === "error" && (
        <p data-testid="score-error" className="error-banner">
          Could not load score: {state.error.message} ({state.error.code}).
        </p>
      )}

      {state.status === "ready" && <ScoreDetailBody releaseId={releaseId} score={state.score} />}
    </div>
  );
}

function ScoreDetailBody({ releaseId, score }: { releaseId: string; score: LegacyScoreBridge }) {
  const { display_record, engine_record } = score;
  const { payoff_curve, ...restDisplay } = display_record;
  const rawVerdict = "gate_pass" in display_record ? display_record["gate_pass"] : undefined;

  return (
    <>
      <h2 data-testid="score-detail-header">Score detail</h2>
      <dl className="field-list score-identity">
        <div className="field-row">
          <dt>score_id</dt>
          <dd data-testid="score-id-value">
            <code>{score.score_id}</code>
          </dd>
        </div>
        <div className="field-row">
          <dt>release_id</dt>
          <dd data-testid="score-release-id-value">
            <code>{releaseId}</code>
          </dd>
        </div>
      </dl>

      <section className="verdict-section" data-testid="score-verdict-section">
        <span data-testid="score-raw-verdict">raw verdict (gate_pass): {fmtUnknown(rawVerdict)}</span>{" "}
        <a href={compatibilityLink(releaseId)} target="_blank" rel="noreferrer" data-testid="compat-link">
          legacy view
        </a>
      </section>

      <section data-testid="payoff-section">
        <h3>Payoff curve</h3>
        <PayoffChart value={payoff_curve} />
      </section>

      <section data-testid="display-record-section">
        <h3>Display fields</h3>
        <DisplayRecordFields record={restDisplay} />
      </section>

      <section data-testid="provenance-section">
        <h3>Provenance</h3>
        <dl className="field-list">
          <div className="field-row">
            <dt>legacy_row_id</dt>
            <dd>{score.legacy_row_id}</dd>
          </div>
          <div className="field-row">
            <dt>clock_id</dt>
            <dd>{score.clock_id}</dd>
          </div>
          <div className="field-row">
            <dt>event</dt>
            <dd>
              {score.event_ref.event_id} (calendar rev. {score.event_ref.calendar_revision})
            </dd>
          </div>
          <div className="field-row">
            <dt>score_batch_ref</dt>
            <dd>{score.score_batch_ref}</dd>
          </div>
          <div className="field-row">
            <dt>snapshot_ref</dt>
            <dd>{score.snapshot_ref}</dd>
          </div>
          <div className="field-row">
            <dt>model_registry_artifact_refs</dt>
            <dd>{score.model_registry_artifact_refs.length > 0 ? score.model_registry_artifact_refs.join(", ") : "(none)"}</dd>
          </div>
          {score.unavailable_detail_reasons.length > 0 && (
            <div className="field-row">
              <dt>unavailable_detail_reasons</dt>
              <dd data-testid="unavailable-detail-reasons">{score.unavailable_detail_reasons.join(", ")}</dd>
            </div>
          )}
        </dl>
      </section>

      <details data-testid="engine-evidence">
        <summary>Engine evidence (raw)</summary>
        <pre data-testid="engine-evidence-json">{JSON.stringify(engine_record, null, 2)}</pre>
      </details>
    </>
  );
}
