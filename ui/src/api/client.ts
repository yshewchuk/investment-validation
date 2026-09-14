/**
 * The one module in this app that calls `fetch` (guide §7: "Components do
 * not read raw bundles directly"; §8 P3-3 task 1: "A typed API client
 * module that is the only place fetch is used.").
 *
 * Auth (§6/§7): the read API is same-origin and reads a session cookie the
 * server sets; this client never reads, stores or forwards a credential
 * itself — no `Authorization` header is built here, no token is put in a
 * URL, and nothing is written to `localStorage`. Every request carries
 * `credentials: "same-origin"` so the browser attaches the cookie
 * automatically. A 401 response surfaces as `ApiError` with `status: 401`;
 * the caller renders the "unauthenticated" state (§7).
 */
import type {
  EventPage,
  EventQuery,
  EventScoreSummary,
  LegacyScoreBridge,
  OperationsHealth,
  PreviewRelease,
  ProblemEnvelope,
} from "./types";

export class ApiError extends Error implements ProblemEnvelope {
  status: number;
  code: string;
  title: string;
  detail?: string;

  constructor(problem: ProblemEnvelope) {
    super(problem.title);
    this.name = "ApiError";
    this.status = problem.status;
    this.code = problem.code;
    this.title = problem.title;
    if (problem.detail !== undefined) {
      this.detail = problem.detail;
    }
  }
}

/** Raised when a caller's own `AbortSignal` cancels a request in flight. */
export class ApiAborted extends Error {
  constructor() {
    super("request aborted");
    this.name = "ApiAborted";
  }
}

async function readProblem(response: Response): Promise<ProblemEnvelope> {
  try {
    const body = (await response.json()) as Partial<ProblemEnvelope>;
    return {
      status: response.status,
      code: body.code ?? String(response.status),
      title: body.title ?? response.statusText,
      ...(body.detail !== undefined ? { detail: body.detail } : {}),
    };
  } catch {
    return { status: response.status, code: String(response.status), title: response.statusText };
  }
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    const init: RequestInit = { credentials: "same-origin" };
    if (signal !== undefined) {
      init.signal = signal;
    }
    response = await fetch(path, init);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiAborted();
    }
    throw error;
  }
  if (!response.ok) {
    throw new ApiError(await readProblem(response));
  }
  return (await response.json()) as T;
}

function queryString(params: Record<string, string | number | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) {
      search.set(key, String(value));
    }
  }
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
}

export interface DataClient {
  getRelease(signal?: AbortSignal): Promise<PreviewRelease>;
  listEvents(query: EventQuery, signal?: AbortSignal): Promise<EventPage>;
  getEventScores(
    eventId: string,
    releaseId: string,
    signal?: AbortSignal,
  ): Promise<EventScoreSummary[]>;
  getScore(scoreId: string, releaseId?: string, signal?: AbortSignal): Promise<LegacyScoreBridge>;
  getOperations(signal?: AbortSignal): Promise<OperationsHealth>;
}

/** The only implementation today: same-origin HTTP against §6's routes. */
export function createHttpDataClient(basePath = "/api/v1"): DataClient {
  return {
    getRelease(signal) {
      return getJson<PreviewRelease>(`${basePath}/releases/current`, signal);
    },
    listEvents(query, signal) {
      const { release_id, ...rest } = query;
      const search = queryString({ release_id, ...rest });
      return getJson<EventPage>(`${basePath}/events${search}`, signal);
    },
    getEventScores(eventId, releaseId, signal) {
      const search = queryString({ release_id: releaseId });
      return getJson<EventScoreSummary[]>(
        `${basePath}/events/${encodeURIComponent(eventId)}/scores${search}`,
        signal,
      );
    },
    getScore(scoreId, releaseId, signal) {
      const search = queryString({ release_id: releaseId });
      return getJson<LegacyScoreBridge>(`${basePath}/scores/${encodeURIComponent(scoreId)}${search}`, signal);
    },
    getOperations(signal) {
      return getJson<OperationsHealth>(`${basePath}/operations`, signal);
    },
  };
}
