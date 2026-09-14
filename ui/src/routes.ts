/**
 * Hash routing for the board/event/score views (guide P3-3b deliverable 3:
 * "hash or path routes that include release_id, so a deep link reopens the
 * same pinned release"). No router library -- three view shapes do not need
 * one (guide §7: "ordinary components, fetch and a small SVG chart
 * suffice").
 *
 * Every route carries an explicit release_id segment. A bare `#/` or an
 * empty hash means "no release pinned by the URL yet" -- the app resolves
 * `current` and then rewrites the address bar to name it (App.tsx), so a
 * copied link always reopens the same pinned release rather than whatever
 * is current when the link is later opened.
 */

export type Route =
  | { name: "board"; releaseId: string | null }
  | { name: "event"; releaseId: string; eventId: string }
  | { name: "score"; releaseId: string; scoreId: string; eventId: string | null };

function decodePart(part: string): string {
  try {
    return decodeURIComponent(part);
  } catch {
    return part;
  }
}

/** Parses `window.location.hash` (with or without the leading `#`). Any
 * shape this function does not recognize falls back to the unpinned board
 * route rather than throwing -- a malformed hash is not a crash. */
export function parseHash(hash: string): Route {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const parts = raw.split("/").filter((p) => p.length > 0);
  if (parts[0] !== "release" || parts.length < 2 || parts[1] === undefined) {
    return { name: "board", releaseId: null };
  }
  const releaseId = decodePart(parts[1]);
  if (parts[2] === "events" && parts[3] !== undefined) {
    const eventId = decodePart(parts[3]);
    if (parts[4] === "scores" && parts[5] !== undefined) {
      return { name: "score", releaseId, scoreId: decodePart(parts[5]), eventId };
    }
    return { name: "event", releaseId, eventId };
  }
  if (parts[2] === "scores" && parts[3] !== undefined) {
    return { name: "score", releaseId, scoreId: decodePart(parts[3]), eventId: null };
  }
  return { name: "board", releaseId };
}

export function boardHash(releaseId: string): string {
  return `#/release/${encodeURIComponent(releaseId)}`;
}

export function eventHash(releaseId: string, eventId: string): string {
  return `#/release/${encodeURIComponent(releaseId)}/events/${encodeURIComponent(eventId)}`;
}

export function scoreHash(releaseId: string, scoreId: string, eventId: string | null): string {
  return eventId === null
    ? `#/release/${encodeURIComponent(releaseId)}/scores/${encodeURIComponent(scoreId)}`
    : `#/release/${encodeURIComponent(releaseId)}/events/${encodeURIComponent(eventId)}` +
        `/scores/${encodeURIComponent(scoreId)}`;
}
