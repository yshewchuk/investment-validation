import { useMemo, useState } from "react";
import type { DataClient } from "./api/client";
import type { EventQuery } from "./api/types";
import { EMPTY_FILTERS, EventFilters, type FilterValues } from "./components/EventFilters";
import { EventTable } from "./components/EventTable";
import { Pagination } from "./components/Pagination";
import { ReleaseBanner } from "./components/ReleaseBanner";
import { pollIntervalMs, usePinnedRelease, useEventPage } from "./hooks";

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
  const { state: releaseState, changedReleaseId } = usePinnedRelease(client, pollMs);

  const [filters, setFilters] = useState<FilterValues>(EMPTY_FILTERS);
  const [cursorStack, setCursorStack] = useState<(string | null)[]>([null]);
  const cursor = cursorStack[cursorStack.length - 1] ?? null;

  const query =
    releaseState.status === "ready"
      ? toQuery(releaseState.release.release_id, filters, cursor)
      : null;
  const pageState = useEventPage(client, query);

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
            : `Could not resolve the current release: ${releaseState.error.title}.`}
        </p>
      </main>
    );
  }

  const release = releaseState.release;

  return (
    <main className="app">
      <h1>v2 board (shadow)</h1>
      <ReleaseBanner release={release} changedReleaseId={changedReleaseId} />
      <EventFilters value={filters} onApply={applyFilters} />

      {pageState.status === "loading" && <p data-testid="loading-page">Loading events…</p>}

      {pageState.status === "error" && (
        <p data-testid="page-error" className="error-banner">
          Could not load events: {pageState.error.title} ({pageState.error.code}).
        </p>
      )}

      {pageState.status === "ready" && (
        <>
          <EventTable items={pageState.page.items} releaseId={release.release_id} />
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
    </main>
  );
}
