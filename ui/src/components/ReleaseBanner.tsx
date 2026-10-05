import type { ApiError } from "../api/client";
import type { OperationsStatus, PreviewRelease } from "../api/types";
import { fmtCoverageEntry } from "../format";

interface Props {
  /** The pinned release id — always known, even when `release` metadata is
   * not (a deep link whose own `GET /api/v1/releases/{id}` fetch failed for
   * a reason other than 404; see hooks.ts `useResolvedRelease`). */
  releaseId: string;
  /** This release's own metadata — for an unpinned load, from `current`;
   * for a deep link (P3-3c), from `GET /api/v1/releases/{id}` directly, so
   * a non-current pin still shows its own id/as-of/coverage, not just
   * current's. */
  release: PreviewRelease | null;
  /** False when the pin differs from `current` (deep link to an older
   * release) — P3-3b deliverable 3: "It shows a notice if it isn't
   * current, and never silently switches." */
  isCurrent: boolean;
  /** The id `current` resolves to right now, when known — lets the "not
   * current" notice link back to it (P3-3c deliverable 2). */
  currentReleaseId: string | null;
  /** Set when `current` itself could not be resolved at all (distinct from
   * simply resolving to a different id). */
  currentError: ApiError | null;
  changedReleaseId: string | null;
  /** The operations-status sidecar (`GET /api/v1/operations`); null before
   * the first reply and whenever the read failed. */
  operationsStatus: OperationsStatus | null;
  /** True while the first sidecar fetch is still in flight. */
  operationsLoading: boolean;
  /** True when the sidecar could not be read at all. */
  operationsUnavailable: boolean;
}

/**
 * §7: "Persistent shadow/producer label, resolved as-of date, quote/input
 * freshness, coverage, and current failure/withheld banner." Plus two
 * distinct notices, never conflated:
 *
 * - `release-not-current-notice`: the pin (deep link or otherwise) was
 *   never `current` to begin with (P3-3b deliverable 3).
 * - `release-changed-notice`: the pin WAS `current` at load time, but the
 *   background poll later saw `current` move away from it (guide §6
 *   requires: "on a release mismatch, show that the release changed and
 *   offer a reload. Never mix releases silently.").
 */
/** A real page load (never a same-document hash change, which would leave
 * `useResolvedRelease`'s pin untouched per "R1 readers retain R1") that
 * drops any pinned release segment from the address bar, so the fresh load
 * resolves whatever `current` is at that moment — the same technique the
 * `release-changed-notice` reload button below already uses. */
export function reloadToCurrent(): void {
  window.location.href = window.location.pathname + window.location.search;
}

export function ReleaseBanner({
  releaseId,
  release,
  isCurrent,
  currentReleaseId,
  currentError,
  changedReleaseId,
  operationsStatus,
  operationsLoading,
  operationsUnavailable,
}: Props) {
  const stale = (release?.stale_or_degraded_reasons.length ?? 0) > 0;
  // Operations-status presentation (display-only): fixed wording plus
  // release/session/occurrence identifiers — never a raw `detail`,
  // `stale_reason`, `withheld_reason` or `failed_update_reason`.
  const now = Date.now();
  const publishedId = changedReleaseId ?? currentReleaseId;
  const published = publishedId !== null && publishedId !== "" ? publishedId : "unavailable";
  const status = operationsStatus;
  const described = status !== null && typeof status.release_id === "string" && status.release_id !== "" ? status.release_id : "unavailable";
  const at = status === null ? Number.NaN : Date.parse(status.generated_at);
  const timed = !Number.isNaN(at) && at <= now;
  const hours = Math.floor((now - at) / 3_600_000);
  const age = !timed ? "age unavailable" : hours >= 24 ? `${Math.floor(hours / 24)}d ${hours % 24}h ago` : now === at ? "0h ago" : hours > 0 ? `${hours}h ago` : "1h ago";
  const rows = status !== null && Array.isArray(status.engineering_history) ? status.engineering_history : [];
  const unknownDates: string[] = [], failDates: string[] = [];
  let malformed = status === null || rows.length === 0;
  for (const row of rows as Array<{ occurrence: unknown; status: unknown } | null>) {
    const occurrence = row !== null && typeof row === "object" && typeof row.occurrence === "string" && row.occurrence !== "" ? row.occurrence : "unavailable";
    const outcome = row !== null && typeof row === "object" ? row.status : undefined;
    if (row === null || typeof row !== "object" || occurrence === "unavailable" || (outcome !== "pass" && outcome !== "unknown" && outcome !== "fail")) { malformed = true; continue; }
    if (outcome === "unknown" && !unknownDates.includes(occurrence)) unknownDates.push(occurrence);
    if (outcome === "fail" && !failDates.includes(occurrence)) failDates.push(occurrence);
  }
  const missing = operationsLoading || operationsUnavailable || status === null;
  const requested = status !== null && typeof status.requested_session === "string" && status.requested_session !== "" ? status.requested_session : "unavailable";
  const resolved = status !== null && typeof status.resolved_session === "string" && status.resolved_session !== "" ? status.resolved_session : "unavailable";
  const attempted = status !== null && typeof status.attempted_release_id === "string" && status.attempted_release_id !== "" ? status.attempted_release_id : "unavailable";
  const attempt = missing || status === null ? "no observation available" : status.failed_update === true ? `failed for release ${attempted}` : `no failed update recorded (attempted release ${attempted})`;
  const reasons: string[] = [];
  let unknownClass = false;
  let label = operationsLoading ? "loading" : "unknown / unavailable";
  let observed = "observation unavailable";
  if (status !== null && !operationsLoading && !operationsUnavailable) {
    observed = timed ? `observed ${age}` : "observation time unknown";
    if (status.schema_version !== "operations_status.v1.0") reasons.push("unexpected operations status schema");
    if (!timed) reasons.push("observation time is invalid or in the future");
    else if (now - at > 86_400_000) reasons.push(`observation is older than 24 hours (${age})`);
    if (described === "unavailable") reasons.push("health/status-described release identity unavailable");
    if (published === "unavailable") reasons.push("latest published release identity unavailable");
    const agree = described !== "unavailable" && published !== "unavailable" && described === published && published === releaseId;
    if (!agree && described !== "unavailable" && published !== "unavailable") reasons.push(`release identity mismatch: board ${releaseId}, status ${described}, published ${published}`);
    if (status.withheld === true) reasons.push("release withheld");
    if (status.failed_update === true) reasons.push("latest update attempt failed");
    if (status.stale === true) reasons.push("operations status flagged stale");
    if (typeof status.withheld !== "boolean" || typeof status.stale !== "boolean" || typeof status.failed_update !== "boolean") reasons.push("operations status flags are absent or non-boolean");
    if (requested === "unavailable" || resolved === "unavailable" || attempted === "unavailable") reasons.push("operations status attempt/session identifiers unavailable");
    if (malformed) reasons.push("scheduled observations missing or malformed");
    if (unknownDates.length > 0) reasons.push(`scheduled observations missing: ${unknownDates.join(", ")}`);
    if (failDates.length > 0) reasons.push(`scheduled observations failed: ${failDates.join(", ")}`);
    unknownClass = status.schema_version !== "operations_status.v1.0" || !timed || malformed || described === "unavailable" || published === "unavailable" || !agree || status.withheld === true || status.failed_update === true || unknownDates.length > 0 || typeof status.withheld !== "boolean" || typeof status.stale !== "boolean" || typeof status.failed_update !== "boolean" || requested === "unavailable" || resolved === "unavailable" || attempted === "unavailable";
    label = reasons.length === 0 ? "current" : unknownClass ? "unknown" : failDates.length > 0 ? "failed" : "stale";
  }
  const boardStale = publishedId !== null && publishedId !== releaseId;
  return (
    <div className="release-banner" data-testid="release-banner">
      <div className="release-banner-row">
        <span className="badge badge-producer">{release?.producer ?? "legacy_via_v2"}</span>
        <span data-testid="release-id">
          release <code>{releaseId}</code>
        </span>
        {release !== null && <span data-testid="release-as-of">as of {release.resolved_as_of}</span>}
        {stale && release !== null && (
          <span className="badge badge-stale" data-testid="release-stale">
            stale/degraded: {release.stale_or_degraded_reasons.join(", ")}
          </span>
        )}
      </div>
      {release !== null && Object.keys(release.coverage_summary).length > 0 && (
        <div className="release-coverage" data-testid="release-coverage">
          {Object.entries(release.coverage_summary).map(([key, value]) => (
            <span key={key} className="coverage-item">
              {key}: {fmtCoverageEntry(key, value)}
            </span>
          ))}
        </div>
      )}
      <div className="release-operations">
        <div className="release-banner-row" data-testid="operations-status">
          <span className={`badge ${label === "current" ? "badge-producer" : "badge-stale"}`}>operations: {label}</span>
          <span>{observed}</span>
          {reasons.map((reason) => <span className="badge badge-refusal" key={reason}>{reason}</span>)}
        </div>
        <div className="release-coverage" data-testid="operations-identities">
          <span>board pin <code>{releaseId}</code></span>
          <span>health/status-described <code>{described}</code></span>
          <span>latest published <code>{published}</code></span>
        </div>
        <div className="release-coverage" data-testid="operations-sessions">
          <span>requested session <code>{requested}</code></span>
          <span>resolved session <code>{resolved}</code></span>
        </div>
        <div className="release-coverage" data-testid="operations-attempt">
          <span>latest attempt {attempt}</span>
        </div>
      </div>
      {!isCurrent && (
        <div className="release-not-current-notice" data-testid="release-not-current-notice">
          {currentError !== null
            ? `The current release could not be resolved (${currentError.message}); showing the ` +
              `pinned release ${releaseId} directly, as saved.`
            : `This is a deep link to release ${releaseId}, which is not the current release. ` +
              `Data for ${releaseId} is shown as saved; it is never silently swapped for the current one.`}
          {currentReleaseId !== null && (
            <>
              {" "}
              <button type="button" onClick={reloadToCurrent} data-testid="current-release-link">
                Go to the current release ({currentReleaseId})
              </button>
            </>
          )}
        </div>
      )}
      {changedReleaseId !== null && (
        <div className="release-changed-notice" data-testid="release-changed-notice">
          The current release changed to <code>{changedReleaseId}</code>. This page keeps
          showing the pinned release <code>{releaseId}</code>.{" "}
          <button type="button" onClick={reloadToCurrent}>
            Reload to see {changedReleaseId}
          </button>
        </div>
      )}
      {boardStale && (
        <div className="release-changed-notice" data-testid="operations-board-stale">
          The displayed board is stale: this page shows release <code>{releaseId}</code> while the latest published release is{" "}
          <code>{published}</code> ({age}). The board is never swapped automatically.{" "}
          <button type="button" onClick={reloadToCurrent}>Reload to see {published}</button>
        </div>
      )}
    </div>
  );
}
