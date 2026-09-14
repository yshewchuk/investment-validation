import { groupDisplayRecordFields } from "../displayFieldSpec";
import { fmtUnknown } from "../format";

interface Props {
  record: Record<string, unknown>;
}

/**
 * Renders every `display_record` field grouped by the mapping spec's
 * category, falling back to "other" (alphabetical) for anything unmapped
 * (guide P3-3b deliverable 2). Only prints already-rendered values (via
 * `fmtUnknown` — no computation); a "derived" tag notes fields the legacy
 * renderer computed itself (`payoff_curve`, `row_id`, ...), which is
 * informational only, not a recomputation performed here.
 */
export function DisplayRecordFields({ record }: Props) {
  const groups = groupDisplayRecordFields(record);
  if (groups.length === 0) {
    return <p data-testid="display-record-empty">No display fields on this score.</p>;
  }
  return (
    <div data-testid="display-record-fields">
      {groups.map((group) => (
        <section key={group.category} className="field-group" data-testid="field-group">
          <h3 data-testid="field-group-category">{group.category}</h3>
          <dl className="field-list">
            {group.fields.map((meta) => (
              <div className="field-row" key={meta.field} data-testid="field-row" data-field={meta.field}>
                <dt>
                  {meta.field}
                  {meta.derived && (
                    <span className="badge badge-derived" title="rendered by the legacy renderer, not recomputed here">
                      derived
                    </span>
                  )}
                  {!meta.knownField && (
                    <span className="badge badge-unmapped" title="not in the mapping spec transcription">
                      unmapped
                    </span>
                  )}
                </dt>
                <dd data-testid="field-value">{fmtUnknown(record[meta.field])}</dd>
              </div>
            ))}
          </dl>
        </section>
      ))}
    </div>
  );
}
