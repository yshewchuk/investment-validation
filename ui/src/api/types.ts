/**
 * Types mirroring `engine/v2/contracts/serving.py` field-for-field (guide
 * §6/§7: "Use these field names exactly in TS types"). This module declares
 * shapes only — no fetch, no formatting, no derived arithmetic. Every
 * nullable Python field (`X | None = None`) is `X | null` here, never
 * optional-with-undefined, so a client that forgets a field fails a type
 * check instead of quietly treating "absent" as "not sent" (§6: "Null means
 * unavailable, never zero.").
 */

export type SchemaVersion = string;

export interface PreviewCapabilities {
  read: boolean;
  submit_jobs: boolean;
  collect_live: boolean;
}

/** `PreviewRelease` (preview_release.v1.0), contracts/serving.py §5.2. */
export interface PreviewRelease {
  release_id: string;
  source_release_id: string;
  projection_manifest_ref: string;
  snapshot_ref: string;
  score_batch_ref: string;
  bundle_manifest_ref: string;
  model_registry_artifact_refs: string[];
  model_evidence_ref: string | null;
  comparison_receipt_refs: string[];
  source_code_hash: string;
  projection_code_hash: string;
  requested_as_of: string;
  resolved_as_of: string;
  clock_ids: string[];
  coverage_summary: Record<string, number>;
  stale_or_degraded_reasons: string[];
  score_format: "legacy_score_bridge.v1.0";
  producer: "legacy_via_v2";
  capabilities: PreviewCapabilities;
  schema_version: SchemaVersion;
}

/** `EventRef`, contracts/data.py. */
export interface EventRef {
  event_id: string;
  calendar_revision: string;
  schema_version: SchemaVersion;
}

/**
 * `EventScoreSummary` (event_score_summary.v1.1), contracts/serving.py §6.
 *
 * v1.1 (P3-1b review fix, `engine/v2/serving/README.md` "Summary-field gap"):
 * the rendered row carries no single merged headline "expected return", so
 * `expected_return` is always `null` here — `expected_return_model`/
 * `_analog`/`_sim` are the honest per-producer reads (`exp_pnl_model`/
 * `exp_pnl_analog`/`exp_pnl_sim` copied through unchanged). `verdict` is the
 * row's raw `gate_pass` as JSON (`"true"`/`"false"`/`null`), never an
 * invented TRADE/REFUSED word — the real board's richer `gatePill` decision
 * tree (`engine/dashboard/static/assets/app.js` ~line 235) is not
 * reproduced here; see ui/README.md's gap list.
 */
export interface EventScoreSummary {
  score_id: string;
  strategy: string;
  verdict: string | null;
  refusal_reason: string | null;
  driver_forecast: number | null;
  market_implied_move: number | null;
  entry_premium: number | null;
  expected_return: number | null;
  expected_return_model: number | null;
  expected_return_analog: number | null;
  expected_return_sim: number | null;
  chosen_strategy: string | null;
  chosen_margin: number | null;
  menu_size: number | null;
  schema_version: SchemaVersion;
}

/** `EventPageItem` (event_page_item.v1.0), contracts/serving.py §6. */
export interface EventPageItem {
  event_ref: EventRef;
  ticker: string;
  event_date: string;
  session: string | null;
  clock_id: string;
  readiness: string;
  scores: EventScoreSummary[];
  schema_version: SchemaVersion;
}

/**
 * `EventPage` (event_page.v1.0), contracts/serving.py §6:
 * `schema_version, release_id, query_hash, items, next_cursor, total_matching`.
 */
export interface EventPage {
  release_id: string;
  query_hash: string;
  items: EventPageItem[];
  next_cursor: string | null;
  total_matching: number;
  schema_version: SchemaVersion;
}

/** `LegacyScoreBridge` (legacy_score_bridge.v1.0), contracts/serving.py §5.2. */
export interface LegacyScoreBridge {
  score_id: string;
  event_ref: EventRef;
  clock_id: string;
  legacy_row_id: string;
  score_batch_ref: string;
  source_row_key: string;
  source_record_hash: string;
  request_provenance_refs: string[];
  snapshot_ref: string;
  model_registry_artifact_refs: string[];
  engine_record: Record<string, unknown>;
  display_record: Record<string, unknown>;
  detail_refs: string[];
  unavailable_detail_reasons: string[];
  schema_version: SchemaVersion;
}

/**
 * Operations health sidecar (`operations_health.v1.0`), separately dated
 * from frozen release data — §6's `GET /api/v1/operations`. Shape matches
 * what `engine/v2/serving/operations.py` already serves at `/health.json`.
 */
export interface OperationsHealth {
  schema_version: "operations_health.v1.0";
  generated_at: string;
  withheld_release: string | null;
  code_budgets?: { consecutive_nights?: number };
}

/**
 * The real `Problem` envelope (`problem.v1.0`, component_contracts.md
 * §2.4), read directly -- `engine/v2/serving/api.py` (P3-2, in review) uses
 * exactly these field names (`_problem()`: code, category, retryable,
 * message, stage, trace_id, dependency_refs, retry_after_seconds,
 * diagnostic_ref, details, schema_version), with no `title`/`status`
 * aliases on the body; the HTTP status code itself is read from the
 * response, not this envelope (`ApiError.status` in `client.ts`).
 */
export interface ProblemEnvelope {
  schema_version: string;
  code: string;
  category: string;
  retryable: boolean;
  message: string;
  stage: string | null;
  trace_id: string | null;
  dependency_refs: string[];
  retry_after_seconds: number | null;
  diagnostic_ref: string | null;
  details: Record<string, unknown>;
}

export interface EventQuery {
  release_id: string;
  date_from?: string;
  date_to?: string;
  ticker?: string;
  strategy?: string;
  verdict?: string;
  cursor?: string;
  limit?: number;
}
