import { Fragment, useEffect, useRef, useState } from "react";
import { ApiError, clientProblem, type DataClient } from "../api/client";
import type { NativeParityMismatchItem, NativeParitySummary } from "../api/types";
import { fmtNumber, fmtText } from "../format";

type Loaded<T> =
  | { status: "loading" }
  | { status: "ready"; data: T }
  | { status: "error"; error: ApiError };

function toApiError(error: unknown): ApiError {
  return error instanceof ApiError ? error : new ApiError(0, clientProblem("NETWORK_ERROR", "network error"));
}

type Stated<T> = { key: string | null; loaded: Loaded<T> };

function useRead<T>(key: string | null, read: (signal: AbortSignal) => Promise<T>): Loaded<T> {
  const [stated, setStated] = useState<Stated<T>>({ key, loaded: { status: "loading" } });
  const readRef = useRef(read);
  readRef.current = read;
  useEffect(() => {
    if (key === null) return;
    let obsolete = false;
    const controller = new AbortController();
    setStated({ key, loaded: { status: "loading" } });
    readRef
      .current(controller.signal)
      .then(
        (data) => !obsolete && setStated({ key, loaded: { status: "ready", data } }),
        (error: unknown) => !obsolete && setStated({ key, loaded: { status: "error", error: toApiError(error) } }),
      );
    return () => {
      obsolete = true;
      controller.abort();
    };
  }, [key]);
  return stated.key === key ? stated.loaded : { status: "loading" };
}

function summaryFields(summary: Exclude<NativeParitySummary, { status: "no_report" }>): [string, string, string][] {
  return [
    ["Status", "parity-status", summary.status],
    ["As of", "parity-as-of", fmtText(summary.as_of)],
    ["Generated at", "parity-generated-at", fmtText(summary.generated_at)],
    ["Tolerance policy", "parity-tolerance", fmtText(summary.tolerance_policy_id)],
    ["Compared rows", "parity-compared", fmtNumber(summary.compared_count, 0)],
    ["Matched rows", "parity-matched", fmtNumber(summary.matched_row_count, 0)],
    ["Mismatched rows", "parity-mismatched", fmtNumber(summary.mismatched_row_count, 0)],
    ["Only legacy", "parity-only-legacy", fmtNumber(summary.only_legacy_count, 0)],
    ["Only native", "parity-only-native", fmtNumber(summary.only_native_count, 0)],
    ["Native refused", "parity-native-refused", fmtNumber(summary.native_refused_count, 0)],
    ["Native refused, unmatched", "parity-native-refused-unmatched",
      fmtNumber(summary.native_refused_unmatched_count, 0)],
  ];
}

function ReadError({ error, prefix, name, label }: {
  error: ApiError;
  prefix: string;
  name: string;
  label: string;
}) {
  return error.status === 401 ? (
    <p data-testid={`${prefix}unauthenticated`}>
      Not authenticated. Sign in with the operations session to continue.
    </p>
  ) : (
    <p className="error-banner" data-testid={`${prefix}${name}`}>
      Could not load {label}: {error.message} ({error.code}).
    </p>
  );
}

function fmtMismatchValue(value: unknown): string {
  if (value === undefined || value === null) return "—";
  if (typeof value === "string") return value;
  if (typeof value === "number") return String(value);
  return JSON.stringify(value);
}

function MismatchTable({ client }: { client: DataClient }) {
  const [cursors, setCursors] = useState<(string | null)[]>([null]);
  const index = cursors.length - 1;
  const cursor = cursors[index] ?? null;
  const page = useRead(`mismatch:${cursors.length}:${cursor}`, (signal) =>
    client.getNativeParityMismatches(50, cursor, signal),
  );
  const data = page.status === "ready" && page.data.status !== "no_report" ? page.data : null;
  const rows: { item: NativeParityMismatchItem; field: string; legacy: unknown; native: unknown }[] = [];
  if (data !== null) {
    for (const item of data.items) {
      for (const [field, cell] of Object.entries(item.fields)) {
        if (cell.status === "differ") rows.push({ item, field, legacy: cell.legacy, native: cell.native });
      }
    }
  }
  const showPager = data !== null && (data.items.length > 0 || index > 0);
  const nextCursor = data?.next_cursor ?? null;
  return (
    <>
      <h2>Mismatched fields</h2>
      {page.status === "loading" && <p data-testid="parity-mismatch-loading">Loading mismatches…</p>}
      {page.status === "error" && (
        <ReadError error={page.error} prefix="parity-mismatch-" name="unavailable" label="mismatches" />
      )}
      {page.status === "ready" && page.data.status === "no_report" && (
        <p data-testid="parity-mismatch-no-report">No native parity report is published yet.</p>
      )}
      {data !== null && data.items.length === 0 && (
        <p data-testid="parity-mismatch-empty">No mismatched rows.</p>
      )}
      {data !== null && data.items.length > 0 && (
        <table data-testid="parity-mismatch-table">
          <thead>
            <tr><th>Row</th><th>Dimension</th><th>Field</th><th>Legacy</th><th>Native</th></tr>
          </thead>
          <tbody>
            {rows.map(({ item, field, legacy, native }) => (
              <tr key={`${item.row_key}|${item.dimension}|${field}`} data-testid="parity-mismatch-row">
                <td>{item.row_key}</td>
                <td>{item.dimension}</td>
                <td>{field}</td>
                <td data-testid="parity-mismatch-legacy">{fmtMismatchValue(legacy)}</td>
                <td data-testid="parity-mismatch-native">{fmtMismatchValue(native)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {showPager && (
        <div data-testid="parity-mismatch-pager">
          <button type="button" data-testid="parity-mismatch-prev" disabled={index === 0}
            onClick={() => setCursors((stack) => stack.slice(0, -1))}>Previous</button>
          <button type="button" data-testid="parity-mismatch-next" disabled={nextCursor === null}
            onClick={() => setCursors((stack) => [...stack, nextCursor])}>Next</button>
          <span data-testid="parity-mismatch-page">Page {index + 1}</span>
        </div>
      )}
    </>
  );
}

export function NativeParity({ client }: { client: DataClient }) {
  const summary = useRead("summary", (signal) => client.getNativeParity(signal));
  const refusalReasons: [string, number][] = summary.status === "ready" && summary.data.status !== "no_report" ? Object.entries(summary.data.native_refused_reasons) : [];
  const refusedTickers: { ticker: string; reason: string }[] = summary.status === "ready" && (summary.data.status === "available" || summary.data.status === "stale") ? summary.data.native_refused_tickers : [];
  return (
    <main className="app">
      <h1>v2 native parity (shadow)</h1>
      <p><a href="#/" data-testid="board-link">Back to board</a></p>

      {summary.status === "loading" && <p data-testid="parity-loading">Loading native parity…</p>}
      {summary.status === "error" && (
        <ReadError error={summary.error} prefix="parity-" name="unavailable" label="native parity" />
      )}
      {summary.status === "ready" && summary.data.status === "no_report" && <p data-testid="parity-no-report">No native parity report is published yet.</p>}
      {summary.status === "ready" && summary.data.status !== "no_report" && (
        <>
          {summary.data.status === "stale" && <p className="error-banner" data-testid="parity-stale">The native parity report is stale; its saved comparison is shown below.</p>}
          {summary.data.partial === true && <p className="error-banner" role="alert" data-testid="parity-partial">This saved report is partial: its schema has no refusal breakdown, so the refusal counts below are incomplete, not confirmed zero.</p>}
          <dl data-testid="parity-summary">{summaryFields(summary.data).map(([label, testid, value]) => <Fragment key={testid}><dt>{label}</dt><dd data-testid={testid}>{value}</dd></Fragment>)}</dl>
          <h2>Refusal reasons</h2>
          {refusalReasons.length === 0 && <p data-testid="parity-refusal-reasons-empty">No refusal reasons.</p>}
          {refusalReasons.length > 0 && <ul>{refusalReasons.map(([reason, count]) => <li key={reason} data-testid="parity-refusal-reason">{reason}: {fmtNumber(count, 0)}</li>)}</ul>}
          {refusedTickers.length > 0 && (
            <>
              <h2 data-testid="parity-refused-tickers-heading">Refused tickers ({fmtNumber(refusedTickers.length, 0)})</h2>
              <ul>{refusedTickers.map(({ ticker, reason }, index) => <li key={`${ticker}-${index}`} data-testid="parity-refused-ticker">{ticker}: {reason}</li>)}</ul>
            </>
          )}
        </>
      )}
      {summary.status === "ready" && summary.data.status !== "no_report" && <MismatchTable client={client} />}
    </main>
  );
}
