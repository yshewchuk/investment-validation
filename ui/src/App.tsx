import { useEffect, useMemo, useRef, useState } from "react";
import type { DataClient } from "./api/client";
import type { EventQuery, EventPageItem } from "./api/types";
import { EMPTY_FILTERS, EventFilters, type FilterValues } from "./components/EventFilters";
import { EventDetail, type EventMeta } from "./components/EventDetail";
import { EventTable } from "./components/EventTable";
import { Pagination } from "./components/Pagination";
import { ReleaseBanner } from "./components/ReleaseBanner";
import { ScoreDetail } from "./components/ScoreDetail";
import { pollIntervalMs, useResolvedRelease, useEventPage, useHashRoute } from "./hooks";
import { boardHash } from "./routes";

const DEFAULT_POLL_MS = 4000;
const DEFAULT_LIMIT = 50;

function toQuery(releaseId: string, filters: FilterValues, cursor: string | null): EventQuery {
  const query: EventQuery = { release_id: releaseId, limit: DEFAULT_LIMIT };
  if (filters.ticker) query.ticker = filters.ticker;
  if (filters.strategy) query.strategy = filters.strategy;
  if (filters.verdict) query.verdict = filters.verdict;
  if (filters.date_from) query.date_from = filters.date_from;
  if (filters.date_to) query.date_to = filters.date_to;
  if (cursor) query.cursor = cursor;
  return query;
}

interface Props {
  client: DataClient;
}

export function App({ client }: Props) {
  const pollMs = useMemo(() => pollIntervalMs(DEFAULT_POLL_MS), []);
  const route = useHashRoute();
  // The route's release segment is only ever consulted for the FIRST pin
  // resolution (guide §9 L02: "R1 readers retain R1") -- capture it once so
  // a later in-app navigation to a different route shape never repins.
  const routeReleaseIdRef = useRef(route.releaseId);
  const { state: releaseState, changedReleaseId } = useResolvedRelease(
    client,
    pollMs,
    routeReleaseIdRef.current,
  );
  const pinnedReleaseId = releaseState.status === "ready" ? releaseState.releaseId : null;

  const [filters, setFilters] = useState<FilterValues>(EMPTY_FILTERS);
  const [cursorStack, setCursorStack] = useState<(string | null)[]>([null]);
  const cursor = cursorStack[cursorStack.length - 1] ?? null;

  // Ticker/date/session for events seen on a loaded board page, so
  // EventDetail can show a header without its own fetch (the dedicated
  // per-event route, §6, returns score summaries only). Never used as a
  // source of score values -- only display labels for an id already known.
  const [eventMetaCache, setEventMetaCache] = useState<Record<string, EventMeta>>({});

  // The board page is only fetched while actually viewing the board --
  // deliverable 4/§7: detail views must not keep the board's own request
  // alive, and going back to "board" re-issues the SAME (filters, cursor)
  // query, reproducing the same page ("keeps the page and cursor state").
  const query =
    pinnedReleaseId !== null && route.name === "board"
      ? toQuery(pinnedReleaseId, filters, cursor)
      : null;
  const pageState = useEventPage(client, query);

  // Once the pin resolves, name it explicitly in the address bar if the
  // page was opened without one (bare `#/` or no hash) -- deliverable 3: "a
  // deep link reopens the same pinned release." `replaceState` (not a hash
  // assignment) so this does not add a spurious back-button history entry.
  useEffect(() => {
    if (pinnedReleaseId === null) return;
    if (route.name === "board" && route.releaseId === null) {
      window.history.replaceState(null, "", boardHash(pinnedReleaseId));
    }
  }, [pinnedReleaseId, route]);

  useEffect(() => {
    if (pageState.status !== "ready") return;
    const items: EventPageItem[] = pageState.page.items;
    setEventMetaCache((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const item of items) {
        const id = item.event_ref.event_id;
        if (next[id] === undefined) {
          next[id] = { ticker: item.ticker, event_date: item.event_date, session: item.session };
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [pageState]);

  function applyFilters(next: FilterValues) {
    setFilters(next);
    setCursorStack([null]);
  }

  function goNext() {
    if (pageState.status === "ready" && pageState.page.next_cursor !== null) {
      setCursorStack((stack) => [...stack, pageState.page.next_cursor]);
    }
  }

  function goPrev() {
    setCursorStack((stack) => (stack.length > 1 ? stack.slice(0, -1) : stack));
  }

  if (releaseState.status === "loading") {
    return (
      <main className="app">
        <p data-testid="loading-release">Resolving current release…</p>
      </main>
    );
  }

  if (releaseState.status === "unauthenticated") {
    return (
      <main className="app">
        <p data-testid="unauthenticated">
          Not authenticated. This is a private shadow dashboard; sign in with the
          operations session to continue.
        </p>
      </main>
    );
  }

  if (releaseState.status === "unavailable") {
    return (
      <main className="app">
        <p data-testid="no-release" className="error-banner">
          {releaseState.error.status === 503
            ? "No current release is published yet."
            : `Could not resolve the current release: ${releaseState.error.message}.`}
        </p>
      </main>
    );
  }

  const { releaseId, release, isCurrent, currentError } = releaseState;

  return (
    <main className="app">
      <h1>v2 board (shadow)</h1>
      <ReleaseBanner
        releaseId={releaseId}
        release={release}
        isCurrent={isCurrent}
        currentError={currentError}
        changedReleaseId={changedReleaseId}
      />

      {route.name === "event" && (
        <EventDetail
          client={client}
          releaseId={releaseId}
          eventId={route.eventId}
          meta={eventMetaCache[route.eventId] ?? null}
        />
      )}

      {route.name === "score" && (
        <ScoreDetail client={client} releaseId={releaseId} scoreId={route.scoreId} eventId={route.eventId} />
      )}

      {route.name === "board" && (
        <>
          <EventFilters value={filters} onApply={applyFilters} />

          {pageState.status === "loading" && <p data-testid="loading-page">Loading events…</p>}

          {pageState.status === "error" && (
            <p data-testid="page-error" className="error-banner">
              Could not load events: {pageState.error.message} ({pageState.error.code}).
            </p>
          )}

          {pageState.status === "ready" && (
            <>
              <EventTable items={pageState.page.items} releaseId={releaseId} />
              <Pagination
                shownCount={pageState.page.items.length}
                totalMatching={pageState.page.total_matching}
                hasNext={pageState.page.next_cursor !== null}
                hasPrev={cursorStack.length > 1}
                onNext={goNext}
                onPrev={goPrev}
              />
            </>
          )}
        </>
      )}
    </main>
  );
}
