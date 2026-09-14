import type { PreviewRelease } from "../api/types";

interface Props {
  release: PreviewRelease;
  changedReleaseId: string | null;
}

/**
 * §7: "Persistent shadow/producer label, resolved as-of date, quote/input
 * freshness, coverage, and current failure/withheld banner." Plus the
 * release-mismatch notice this guide's §6 requires: "on a release mismatch,
 * show that the release changed and offer a reload. Never mix releases
 * silently."
 */
export function ReleaseBanner({ release, changedReleaseId }: Props) {
  const stale = release.stale_or_degraded_reasons.length > 0;
  return (
    <div className="release-banner" data-testid="release-banner">
      <div className="release-banner-row">
        <span className="badge badge-producer">{release.producer}</span>
        <span data-testid="release-id">
          release <code>{release.release_id}</code>
        </span>
        <span data-testid="release-as-of">as of {release.resolved_as_of}</span>
        {stale && (
          <span className="badge badge-stale" data-testid="release-stale">
            stale/degraded: {release.stale_or_degraded_reasons.join(", ")}
          </span>
        )}
      </div>
      {Object.keys(release.coverage_summary).length > 0 && (
        <div className="release-coverage" data-testid="release-coverage">
          {Object.entries(release.coverage_summary).map(([key, value]) => (
            <span key={key} className="coverage-item">
              {key}: {(value * 100).toFixed(1)}%
            </span>
          ))}
        </div>
      )}
      {changedReleaseId !== null && (
        <div className="release-changed-notice" data-testid="release-changed-notice">
          The current release changed to <code>{changedReleaseId}</code>. This page keeps
          showing the pinned release <code>{release.release_id}</code>.{" "}
          <button type="button" onClick={() => window.location.reload()}>
            Reload to see {changedReleaseId}
          </button>
        </div>
      )}
    </div>
  );
}
