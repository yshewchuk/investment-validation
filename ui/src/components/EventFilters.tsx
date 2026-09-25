import { useState } from "react";

export interface FilterValues {
  ticker: string;
  strategy: string;
  verdict: string;
  gate: string;
  outOfDomain: boolean;
  disabled: boolean;
  date_from: string;
  date_to: string;
}

export const EMPTY_FILTERS: FilterValues = {
  ticker: "",
  strategy: "",
  verdict: "",
  gate: "",
  outOfDomain: false,
  disabled: false,
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
        Gate
        <select
          value={draft.gate}
          onChange={(event) => setDraft({ ...draft, gate: event.target.value })}
        >
          <option value="">All</option>
          <option value="pass">Pass</option>
          <option value="fail">Fail</option>
          <option value="na">N/A</option>
        </select>
      </label>
      <label>
        Show out-of-domain
        <input
          type="checkbox"
          checked={draft.outOfDomain}
          onChange={(event) => setDraft({ ...draft, outOfDomain: event.target.checked })}
        />
      </label>
      <label>
        Show disabled structures
        <input
          type="checkbox"
          checked={draft.disabled}
          onChange={(event) => setDraft({ ...draft, disabled: event.target.checked })}
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
