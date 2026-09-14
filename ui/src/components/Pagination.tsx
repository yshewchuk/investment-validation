interface Props {
  shownCount: number;
  totalMatching: number;
  hasNext: boolean;
  hasPrev: boolean;
  onNext: () => void;
  onPrev: () => void;
}

/** §7: "paging, count". `total_matching` is the complete filtered
 * population (§6), never just the visible page. */
export function Pagination({ shownCount, totalMatching, hasNext, hasPrev, onNext, onPrev }: Props) {
  return (
    <div className="pagination" data-testid="pagination">
      <span data-testid="pagination-count">
        showing {shownCount} of {totalMatching}
      </span>
      <button type="button" onClick={onPrev} disabled={!hasPrev} data-testid="page-prev">
        Previous
      </button>
      <button type="button" onClick={onNext} disabled={!hasNext} data-testid="page-next">
        Next
      </button>
    </div>
  );
}
