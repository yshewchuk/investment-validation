import type { ApiError } from "../api/client";
import type { PreviewRelease } from "../api/types";

interface Props {
  /** The pinned release id — always known, even when `release` metadata is
   * not (a deep link to a non-current release; see hooks.ts
   * `useResolvedRelease`). */
  releaseId: string;
  /** Full metadata, only available when the pin equals `current`. */
  release: PreviewRelease | null;
  /** False when the pin differs from `current` (deep link to an older
   * release) — P3-3b deliverable 3: "It shows a notice if it isn't
   * current, and never silently switches." */
  isCurrent: boolean;
  /** Set when `current` itself could not be resolved at all (distinct from
   * simply resolving to a different id). */
  currentError: ApiError | null;
  changedReleaseId: string | null;
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
export function ReleaseBanner({ releaseId, release, isCurrent, currentError, changedReleaseId }: Props) {
  const stale = (release?.stale_or_degraded_reasons.length ?? 0) > 0;
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
              {key}: {(value * 100).toFixed(1)}%
            </span>
          ))}
        </div>
      )}
      {!isCurrent && (
        <div className="release-not-current-notice" data-testid="release-not-current-notice">
          {currentError !== null
            ? `The current release could not be resolved (${currentError.message}); showing the ` +
              `pinned release ${releaseId} directly, as saved.`
            : `This is a deep link to release ${releaseId}, which is not the current release. ` +
              `Data for ${releaseId} is shown as saved; it is never silently swapped for the current one.`}
        </div>
      )}
      {changedReleaseId !== null && (
        <div className="release-changed-notice" data-testid="release-changed-notice">
          The current release changed to <code>{changedReleaseId}</code>. This page keeps
          showing the pinned release <code>{releaseId}</code>.{" "}
          <button
            type="button"
            onClick={() => {
              // A plain `location.reload()` would reload the SAME URL --
              // which, once pinned, names THIS (now-stale) release
              // explicitly (App.tsx's address-bar rewrite, deliverable 3).
              // Drop the release segment first so the fresh load re-resolves
              // whatever is current at that moment, not this pinned one.
              window.location.href = window.location.pathname + window.location.search;
            }}
          >
            Reload to see {changedReleaseId}
          </button>
        </div>
      )}
    </div>
  );
}
