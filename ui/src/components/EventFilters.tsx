import { useState } from "react";

export interface FilterValues {
  ticker: string;
  strategy: string;
  verdict: string;
  date_from: string;
  date_to: string;
}

export const EMPTY_FILTERS: FilterValues = {
  ticker: "",
  strategy: "",
  verdict: "",
  date_from: "",
  date_to: "",
};

interface Props {
  value: FilterValues;
  onApply: (value: FilterValues) => void;
}

/** §7: "Date/ticker/strategy/verdict filters, sort, paging, count". Sort
 * stays the API's documented default (event date, ticker, event ID; §6) —
 * no sort control is shipped in this slice (recorded as a judgement call
 * in ui/README.md). */
export function EventFilters({ value, onApply }: Props) {
  const [draft, setDraft] = useState(value);

  return (
    <form
      className="event-filters"
      data-testid="event-filters"
      onSubmit={(event) => {
        event.preventDefault();
        onApply(draft);
      }}
    >
      <label>
        Ticker
        <input
          value={draft.ticker}
          onChange={(event) => setDraft({ ...draft, ticker: event.target.value })}
          placeholder="e.g. MRVL"
        />
      </label>
      <label>
        Strategy
        <input
          value={draft.strategy}
          onChange={(event) => setDraft({ ...draft, strategy: event.target.value })}
          placeholder="e.g. STR-THRU"
        />
      </label>
      <label>
        Verdict
        <input
          value={draft.verdict}
          onChange={(event) => setDraft({ ...draft, verdict: event.target.value })}
          placeholder="e.g. enter"
        />
      </label>
      <label>
        From
        <input
          type="date"
          value={draft.date_from}
          onChange={(event) => setDraft({ ...draft, date_from: event.target.value })}
        />
      </label>
      <label>
        To
        <input
          type="date"
          value={draft.date_to}
          onChange={(event) => setDraft({ ...draft, date_to: event.target.value })}
        />
      </label>
      <button type="submit">Apply</button>
      <button
        type="button"
        onClick={() => {
          setDraft(EMPTY_FILTERS);
          onApply(EMPTY_FILTERS);
        }}
      >
        Reset
      </button>
    </form>
  );
}
