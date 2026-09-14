import { useEffect, useRef, useState } from "react";
import { ApiAborted, ApiError, clientProblem, type DataClient } from "./api/client";
import type {
  EventPage,
  EventQuery,
  EventScoreSummary,
  LegacyScoreBridge,
  PreviewRelease,
} from "./api/types";
import { parseHash, type Route } from "./routes";

/**
 * Read-only test affordance: `?pollMs=200` shortens the release re-check
 * interval so a browser test does not have to wait out the production
 * default. Never reads a credential, never changes what is fetched — only
 * how often. Falls back to `defaultMs` for any missing/invalid value.
 */
export function pollIntervalMs(defaultMs: number): number {
  const raw = new URLSearchParams(window.location.search).get("pollMs");
  const parsed = raw === null ? NaN : Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : defaultMs;
}

export type ReleaseState =
  | { status: "loading" }
  | {
      status: "ready";
      /** The pinned release id used for every subsequent request/cache key.
       * Equal to `current`'s id unless the page was opened with an explicit
       * `#/release/<id>/...` deep link (P3-3b deliverable 3). */
      releaseId: string;
      /** False when the pinned id differs from `current`, OR the pin came
       * from a deep link and `current` could not be resolved at all. */
      isCurrent: boolean;
      /** Full metadata is only available when the pin equals `current` —
       * there is no "get an arbitrary past release's metadata" route (§6
       * only exposes `current`). A pinned-but-not-current view still fetches
       * events/scores fine (those accept an explicit `release_id`); it just
       * cannot show `resolved_as_of`/coverage for that release. */
      release: PreviewRelease | null;
      /** Set when `current` itself could not be resolved (e.g. 503 "no
       * current release published") but an explicit deep-link pin still
       * lets the page proceed. Null when `current` resolved normally. */
      currentError: ApiError | null;
    }
  | { status: "unauthenticated" }
  | { status: "unavailable"; error: ApiError };

/**
 * Resolves the pinned release exactly once per page load (guide §6/§7:
 * "resolve current once; carry `release_id` in every request"; P3-3b
 * deliverable 3: "A deep link to a release that is no longer current still
 * loads that release ... never silently switches"), then holds it in state
 * for the rest of the session:
 *
 * - No `routeReleaseId` (bare `#/` or no hash): pins to whatever `current`
 *   resolves to, exactly like P3-3a.
 * - An explicit `routeReleaseId` (a deep link): pins to THAT id regardless
 *   of what `current` turns out to be. `current` is still fetched once, only
 *   to learn whether the pin is current and, if so, to source its metadata.
 *
 * `routeReleaseId`'s value at the FIRST render is what gets pinned; guide
 * §9 L02's "R1 readers retain R1" rule means a later change to the route
 * (which this app never makes to its own pinned segment; only a fresh
 * page load can) must not silently repin.
 *
 * A background poll (`pollIntervalMs`) keeps checking `current` without
 * ever moving the pin — it only flags `changedReleaseId` so the UI can
 * offer a reload.
 */
export function useResolvedRelease(
  client: DataClient,
  pollMs: number,
  routeReleaseId: string | null,
) {
  const [state, setState] = useState<ReleaseState>({ status: "loading" });
  const [changedReleaseId, setChangedReleaseId] = useState<string | null>(null);
  const pinnedIdRef = useRef<string | null>(null);
  const routePinRef = useRef(routeReleaseId);

  useEffect(() => {
    let cancelled = false;
    const pin = routePinRef.current;
    client
      .getRelease()
      .then((release) => {
        if (cancelled) return;
        const releaseId = pin ?? release.release_id;
        const isCurrent = releaseId === release.release_id;
        pinnedIdRef.current = releaseId;
        setState({
          status: "ready",
          releaseId,
          isCurrent,
          release: isCurrent ? release : null,
          currentError: null,
        });
      })
      .catch((error: unknown) => {
        if (cancelled || error instanceof ApiAborted) return;
        if (error instanceof ApiError && error.status === 401) {
          setState({ status: "unauthenticated" });
          return;
        }
        const apiError =
          error instanceof ApiError
            ? error
            : new ApiError(0, clientProblem("NETWORK_ERROR", "network error"));
        if (pin !== null) {
          // `current` is broken, but a deep link still names a specific,
          // presumably-still-readable release (guide §5.4: "the previous
          // release remains readable"). Proceed pinned to it.
          pinnedIdRef.current = pin;
          setState({
            status: "ready",
            releaseId: pin,
            isCurrent: false,
            release: null,
            currentError: apiError,
          });
          return;
        }
        setState({ status: "unavailable", error: apiError });
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (pinnedIdRef.current === null) return;
      client
        .getRelease()
        .then((release) => {
          if (release.release_id !== pinnedIdRef.current) {
            setChangedReleaseId(release.release_id);
          }
        })
        .catch(() => {
          // A transient poll failure is not the pinned session's problem;
          // the board keeps showing the last good pinned release.
        });
    }, pollMs);
    return () => window.clearInterval(timer);
  }, [client, pollMs]);

  return { state, changedReleaseId };
}

/**
 * Parses `window.location.hash` into a `Route` and re-parses on every
 * `hashchange` — the only navigation event this app relies on; there is no
 * router library (guide §7: three view shapes do not need one).
 */
export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    function onHashChange() {
      setRoute(parseHash(window.location.hash));
    }
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);
  return route;
}

export type PageState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; page: EventPage }
  | { status: "error"; error: ApiError };

/**
 * Fetches one event page for the given (pinned) query, discarding any
 * reply whose release/query no longer matches the latest request (§7:
 * "Abort/discard replies whose release/query no longer matches the
 * selected state.").
 */
export function useEventPage(client: DataClient, query: EventQuery | null): PageState {
  const [state, setState] = useState<PageState>({ status: "idle" });
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (query === null) {
      setState({ status: "idle" });
      return;
    }
    const requestId = ++requestIdRef.current;
    const controller = new AbortController();
    setState({ status: "loading" });
    client
      .listEvents(query, controller.signal)
      .then((page) => {
        if (requestIdRef.current !== requestId) return; // stale reply, discard
        if (page.release_id !== query.release_id) return; // belt and suspenders
        setState({ status: "ready", page });
      })
      .catch((error: unknown) => {
        if (requestIdRef.current !== requestId || error instanceof ApiAborted) return;
        setState({
          status: "error",
          error:
            error instanceof ApiError
              ? error
              : new ApiError(0, clientProblem("NETWORK_ERROR", "network error")),
        });
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, JSON.stringify(query)]);

  return state;
}

export type EventScoresState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; scores: EventScoreSummary[] }
  | { status: "not_found" }
  | { status: "error"; error: ApiError };

/**
 * Lazy per-event score fetch (guide P3-3b deliverable 1: "open an event
 * from the board, lazily fetch its scores for the pinned release"; §7:
 * "Load only after selection"). Cache key is `(client, releaseId, eventId)`
 * — the pinned release id is always part of it (deliverable 4), and a stale
 * reply for a since-abandoned event/release is discarded exactly like
 * `useEventPage` (§7: "Abort/discard replies whose release/query no longer
 * matches the selected state.").
 */
export function useEventScores(
  client: DataClient,
  releaseId: string | null,
  eventId: string | null,
): EventScoresState {
  const [state, setState] = useState<EventScoresState>({ status: "idle" });
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (releaseId === null || eventId === null) {
      setState({ status: "idle" });
      return;
    }
    const requestId = ++requestIdRef.current;
    const controller = new AbortController();
    setState({ status: "loading" });
    client
      .getEventScores(eventId, releaseId, controller.signal)
      .then((scores) => {
        if (requestIdRef.current !== requestId) return; // stale reply, discard
        setState({ status: "ready", scores });
      })
      .catch((error: unknown) => {
        if (requestIdRef.current !== requestId || error instanceof ApiAborted) return;
        if (error instanceof ApiError && error.status === 404) {
          setState({ status: "not_found" });
          return;
        }
        setState({
          status: "error",
          error:
            error instanceof ApiError
              ? error
              : new ApiError(0, clientProblem("NETWORK_ERROR", "network error")),
        });
      });
    return () => controller.abort();
  }, [client, releaseId, eventId]);

  return state;
}

export type ScoreDetailState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; score: LegacyScoreBridge }
  | { status: "not_found" }
  | { status: "error"; error: ApiError };

/**
 * Lazy score-detail fetch (guide P3-3b deliverable 2; §7: "Load only after
 * selection; a detail failure leaves the board usable."). Cache key is
 * `(client, releaseId, scoreId)` — `releaseId` is always the pinned release
 * (deliverable 4/6: "switching the release mid-session keeps detail
 * fetches on the pinned release"), never `current`. `releaseId` is required,
 * not optional: P3-2's `GET /api/v1/scores/{id}` requires `release_id` (400
 * if missing) -- the caller always has the pinned release by the time this
 * runs (`ScoreDetail`'s own `releaseId` prop is a plain `string`).
 */
export function useScoreDetail(
  client: DataClient,
  releaseId: string,
  scoreId: string | null,
): ScoreDetailState {
  const [state, setState] = useState<ScoreDetailState>({ status: "idle" });
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (scoreId === null) {
      setState({ status: "idle" });
      return;
    }
    const requestId = ++requestIdRef.current;
    const controller = new AbortController();
    setState({ status: "loading" });
    client
      .getScore(scoreId, releaseId, controller.signal)
      .then((score) => {
        if (requestIdRef.current !== requestId) return; // stale reply, discard
        setState({ status: "ready", score });
      })
      .catch((error: unknown) => {
        if (requestIdRef.current !== requestId || error instanceof ApiAborted) return;
        if (error instanceof ApiError && error.status === 404) {
          setState({ status: "not_found" });
          return;
        }
        setState({
          status: "error",
          error:
            error instanceof ApiError
              ? error
              : new ApiError(0, clientProblem("NETWORK_ERROR", "network error")),
        });
      });
    return () => controller.abort();
  }, [client, releaseId, scoreId]);

  return state;
}
