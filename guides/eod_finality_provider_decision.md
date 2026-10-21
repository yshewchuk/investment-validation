# EOD availability/finality: provider-decision memo (issue #260)

Status: decision input for the repo owner. Docs only; nothing here is implemented or chosen.
Everything marked **(mine)** is the author's judgement, not a recorded user decision.

## Requirements (from #260 and the merged #497 design)

R-A. A request-scoped completion/finality marker for the exact requested partition, including a
provider-stated completed-empty answer. Session date, endpoint name, mtime, row presence and ratio
floors do not count.
R-B. Exact coverage: complete domain membership and genuinely completed-empty domains.
R-C. A trusted readiness clock, never a caller or model clock. #497 requires a post-publication
attestation by the coordinator (after source and normalized outputs are durably published) bounding
readiness. A raw-byte receipt time only shows when bytes were first seen; it cannot bound later
normalization or publication, so it is an input to that bound, not the bound itself.

Today `verify_eod_availability` (`engine/v2/ops/source_availability.py`) validates the request form,
the pinned snapshot and the evidence bytes, then unconditionally refuses. Malformed requests raise `INVALID_REQUEST`, corrupt evidence can
raise `INTEGRITY_FAILED`, and `VALIDATION_FAILED` is the terminal outcome after a successful
preflight. It never parses evidence. The target table is `option_chains` only.

## Finding that reframes the choice

Neither candidate currently supplies native `option_chains` bytes.

- The native ORATS fetcher (`engine/v2/ops/providers/orats_daily_market.py`) acquires only
  `daily_market` (`hist/summaries` plus `hist/cores`, one `tradeDate` per unit). The ORATS
  `option_chains` table is produced by legacy code (`engine/data/normalize/n_chains.py`, fed by
  `strikes` pulls) and reaches v2 through legacy materialization.
- Massive pinned acquisitions (`massive_plan.DATASETS = ("daily", "splits", "dividends")`) are stock
  daily aggregates and corporate actions. Massive has no pinned option-chain acquisition in this repo's production path; the only option-chain fetch is the manual measurement probe `tools/finality_probe.py` (see `guides/finality_probe_runbook.md`), which stores private captures outside the repository and feeds no table. A recorded probe (2026-08-30, Polygon naming,
  `engine/data/normalize/n_option_daily.py`) found the plan covers option daily aggregates,
  reference and snapshots, while trades, quotes and intraday bars return NOT_AUTHORIZED, so
  there is no bid/ask. Whether the current Massive account matches that probe is unconfirmed.

So "which provider supplies the marker" is really "which provider gets a native option-chain
acquisition path". Massive would need a new dataset (and plan, quota and pin work) first.

## ORATS fetch path

1. **Marker.** None request-scoped. `_classify` treats 404 as not-final and a 2xx with no rows as
   not-final; a 2xx with rows is "final" by assumption. The market-wide input to the legacy finality
   barrier (`legacy_adapter.finality_market_wide_complete`) only checks that both endpoints have a
   cached status-200 entry for the date; the full barrier also applies daily and chain coverage
   shares, which is a heuristic, not a marker. There is no completion watermark or end-of-data field in what the repo reads.
   Whether ORATS publishes one (response header, `strikes` metadata, status page): **unknown**, not
   verified against provider docs or a captured response.
2. **Exact coverage.** Partly. The fetcher knows the expected ticker set and flags
   missing tickers as `partial` with a paired retry. That proves "every requested ticker returned a
   row", not that each ticker's strike/expiry chain is complete. A completed-empty ticker cannot be
   told from an unpublished one, because empty 2xx is mapped to not-final.
3. **Clock.** `data_raw_receipts.received_at` is sampled from `SystemClock` at acquisition
   (`_fetch_unit` in `engine/v2/data/incremental.py`), and that `cache_raw_receipt` call passes no
   `fence_check`, so these raw receipts are not attempt-fenced (snapshot publication fencing is a
   separate step). It is a "bytes first seen by" time, not a provider "complete as of" time or a
   post-publication attestation. It bounds raw-byte observation only and cannot upper-bound
   publication readiness; that bound needs the later coordinator attestation.
4. **Slice 1 of #497.** Add an `option_chains` ORATS acquisition (new `strikes` fetch unit), a
   scoped completion receipt, and a completed-empty rule. Without a provider marker it can only
   assert "all requested tickers returned, N rows". Roughly the stated 170 code lines, plus the
   marker research, which is unsized and may block it.
5. **Gaps/risks.** Retroactive proof is impossible for existing legacy chains: no raw receipts bind
   them and `capture_inputs` records empty finality and availability refs. An ORATS day can be
   revised after first publication; a receipt shows only what was served at that moment. The one
   known defect-class (silent partials behind a 2xx) is what this requirement exists to prevent, and
   absent a marker we would be re-deriving it from ratios.

## Massive pinned acquisitions (#715, #731)

1. **Marker.** No finality marker either, but the strongest structural evidence of the two. Each
   body must have `status == "OK"` and, when `resultsCount` is present (it is optional), a matching
   count; the envelope ticker/adjusted match applies to the daily dataset only. Exact-mode
   completion needs `next_url` absent (`massive_plan._complete_body`), whereas generation mode
   accepts validated continuations before the terminal page; cursors and row dates are
   scope-checked (`stores/massive/pages.py`). That proves complete pagination of one request, not
   that the provider has finished the session. Finality is a local rule: `closed_day` requires the
   range end to be before the Eastern date of the observation, and its own docstring says it is
   "not source finality evidence". A provider completion field: **unknown**.
2. **Exact coverage.** Strong for what it covers. Saved coverage mosaics are bound to the producing
   job and fence (`pinned.acquisition_input`), and rows are scoped per request. An empty result is
   rejected (`SOURCE_EMPTY`), so completed-empty is not representable today. All of this is for
   daily bars and actions, not chains.
3. **Clock.** Best of the two: receipts carry `received_at` from the production `SystemClock` under a
   fence check (`verify_fence`, Massive path only), and `observed_at` is recorded separately from
   `first_received_at`. Same limitation: it bounds raw-byte observation only, not publication readiness.
4. **Slice 1.** Add an options dataset to `massive_plan`, request descriptor, page validation and
   an empty-domain rule, then reuse the existing acquisition/pin machinery. Larger than ORATS
   slice 1 despite the reuse: roughly 200+ code lines **(mine)**, likely needing a split, and it depends on
   the probed entitlement (daily aggregates yes; trades, quotes and intraday no). Option B therefore
   lacks bid/ask, so it cannot replace ORATS chain quotes; whether a chain-level completion marker
   exists stays **unknown**.
5. **Gaps/risks.** Same retroactive gap (reconstructed snapshots "prove neither historical
   availability nor source finality", per the ops architecture note). New vendor surface for the
   most important table. Switching `option_chains` source changes price semantics and would need
   its own parity work.

## Options

| Option | What it buys | Cost / risk |
|---|---|---|
| A. ORATS | Same source as the existing table; no semantics change | No known marker; slice 1 is blocked on provider research; may end as ratio-based and so not genuine |
| B. Massive | Cleanest receipts, fencing, pinning and clock already built | No option-chain dataset yet; probed plan has no bid/ask; source change for `option_chains` |
| C. Neither (keep closed) | Zero false admission; honest status | Non-shadow callers stay blocked; cutover critical path unmoved |

## Recommendation (mine)

1. **Do not start #497 slice 1 yet.** Spend one small, bounded investigation first: confirm from
   ORATS and Massive documentation or one captured response whether either exposes a request-scoped
   completion field for option chains (ORATS `strikes`; Massive options). The repo cannot answer this.
2. If ORATS exposes the same two-sided evidence (non-empty and completed-empty), choose **A**: lowest blast radius, same table, and the receipt/clock
   plumbing (`cache_raw_receipt`, `received_at`; no attempt fence on that path today) already applies.
3. If ORATS has none but Massive offers options with request-scoped completion evidence for both
   non-empty results and completed-empty results (R-A and R-B; neither a marker nor explicit empty
   responses alone suffices), choose **B**, accepting the larger slice and the source-change parity work.
4. If neither does, choose **C** and leave admission closed. Do not substitute a ratio or row-count
   heuristic; that is the defect this issue exists to remove.

The earlier reviewer suggestion (#260, planning review) to prefer Massive because its acquisitions
are already pinned natively holds for receipt quality but not for scope: Massive pins today cover no
option-chain data.

## Follow-ups

- Provider-documentation check (above) as a separate issue if the owner agrees.
- Not covered: spot and quote-expiry sourcing, and daily-state reuse (see #497 slice 5).
