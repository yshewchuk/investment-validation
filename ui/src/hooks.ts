import { useEffect, useRef, useState } from "react";
import { ApiAborted, ApiError, type DataClient } from "./api/client";
import type { EventPage, EventQuery, PreviewRelease } from "./api/types";

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
  | { status: "ready"; release: PreviewRelease }
  | { status: "unauthenticated" }
  | { status: "unavailable"; error: ApiError };

/**
 * Resolves `current` exactly once (guide §6/§7: "resolve current once;
 * carry `release_id` in every request"), then pins it in state for the rest
 * of the session. A background poll (`pollIntervalMs`) keeps checking
 * `current` without ever moving the pin — it only flags `changedReleaseId`
 * so the UI can offer a reload, matching the compatibility preview's own
 * "R1 readers retain R1; a new session resolves R2" rule (§9 L02).
 */
export function usePinnedRelease(client: DataClient, pollMs: number) {
  const [state, setState] = useState<ReleaseState>({ status: "loading" });
  const [changedReleaseId, setChangedReleaseId] = useState<string | null>(null);
  const pinnedIdRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    client
      .getRelease()
      .then((release) => {
        if (cancelled) return;
        pinnedIdRef.current = release.release_id;
        setState({ status: "ready", release });
      })
      .catch((error: unknown) => {
        if (cancelled || error instanceof ApiAborted) return;
        if (error instanceof ApiError && error.status === 401) {
          setState({ status: "unauthenticated" });
        } else if (error instanceof ApiError) {
          setState({ status: "unavailable", error });
        } else {
          setState({
            status: "unavailable",
            error: new ApiError({ status: 0, code: "NETWORK_ERROR", title: "network error" }),
          });
        }
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
              : new ApiError({ status: 0, code: "NETWORK_ERROR", title: "network error" }),
        });
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, JSON.stringify(query)]);

  return state;
}
