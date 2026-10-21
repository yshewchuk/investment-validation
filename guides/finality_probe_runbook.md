# EOD finality probe: runbook and sufficiency rule (#877, #260)

Status: measurement design. Nothing here chooses a provider. Anything marked **(mine)** is the author's judgement.

## Question

The docs check found no documented request-scoped end-of-day completion marker for ORATS `hist/strikes` or the Massive option chain snapshot. This probe asks whether an **analog** exists in the bytes the providers actually serve: does a payload stop changing after some time T, and does `updatedAt`, a response header or the record count signal that it has?

## In-repo analog: none usable

- Legacy `engine.data.finality.session_finality` (`engine/data/finality.py`) is ratio-based. It passes when both ORATS market-wide files (`hist/summaries`, `hist/cores`) have a cached status-200 entry for the date and the covered-universe share is at least 0.80 in daily state and option chains (`MIN_FINAL_DAILY_SHARE`, `MIN_FINAL_CHAIN_SHARE`). It is the heuristic that #260 and `guides/eod_finality_provider_decision.md` reject: it shows that rows arrived, not that the provider has finished the session.
- Native `engine/v2/ops/finality.py` binds the same reads through `legacy_adapter`, so it inherits the rule. `verify_eod_availability` refuses unconditionally.
- Native acquisition receipts (`received_at`) bound raw-byte observation only, not publication readiness.

So the native side has nothing to reuse. If the probe supports a rule, the rule is new.

## Candidate sufficiency rule R **(mine)**

A refetch at T+X whose content hash equals the fetch at T, with max(`updatedAt`) at T earlier than T, counts as final at T.

Terms: content hash is the sha256 of the response rows sorted canonically (policy `rows-sorted-canonical-json-v1`; no field is excluded), so row order does not count. `compare` recomputes it from each stored body and skips captures whose digest or policy does not match. Only metadata, never the bodies, goes into reports. For Massive, `max_updated_at` is the larger of `day.last_updated` and `last_quote.last_updated` over the results. Comparisons are per trade date, and only successful captures (HTTP 2xx with a JSON body) count; failed ones are listed as `excluded_not_ok`.

## Evidence that R is sufficient, or not

Sufficient (for the tickers and sessions sampled): over at least 3 sessions and at least 5 tickers, including one liquid ETF and one thin name:
1. `false_final` is 0. R never declares a capture final that a later capture changes.
2. R is not vacuous: at least one `true_not_final` shows an early capture whose content a later capture changed, or the first capture after the close is already identical to all later ones (then X can be short).
3. The earliest label from which every group is `stable_from` is the same across sessions, so X can be fixed.

Not sufficient:
- Any `false_final`, or content changing between the later labels (`next_morning` to `plus_1d`).
- `max_updated_at` is null or later than the fetch time (clock skew): the `updatedAt` leg of R cannot be used.
- Stability only on the evening labels but a revision on the next day (a restatement window): R cannot certify a same-day verdict.
- Fewer than 3 sessions: report "inconclusive", not "sufficient".

Only `etag` and `last-modified` are evaluated as marker candidates (`age`, `date` and `content-length` are displayed only). A header counts as a completion marker only if every group reports `header_tracking: yes` for it (the overall line then says "header tracks content"; a group with no usable pair counts as not tracking), or the row count is the same at every label and differs from a known-incomplete capture. Otherwise report it as not a marker.

Limit: a sample cannot prove the provider never revises. A pass supports R as an evidence-backed heuristic for admission, and the design must still record the revision window it observed.

## Commands for the supervisor

Run from a checkout of this branch. The output directory must be outside any repository, for example `~/finality-probe` (directories are created 0700 and files 0600, a response over 64 MiB is recorded as a failed unit, and the directory is never mirrored). Keys come from the environment (`ORATS_API_KEY`, and `MASSIVE_API_KEY` for `--massive`); the script never prints or stores them. Source the nightly env file in the same shell, do not paste keys into commands.

Pick a recent normal full session `S` (not a half day). Eastern times:

| Label | When (ET) |
|---|---|
| `close_plus_15m` | S 16:20 |
| `evening` | S 21:00 |
| `next_morning` | S+1 08:00 |
| `plus_1d` | S+2 (next trading day after S+1) 08:00 |

```
TICKERS=SPY,AAPL,MSFT,NVDA,<one thin name>
python3 tools/finality_probe.py capture --session S --tickers $TICKERS --label LABEL --out ~/finality-probe \
    --orats-endpoints hist/strikes,hist/summaries [--massive]
python3 tools/finality_probe.py compare --out ~/finality-probe --session S
```

Repeat for at least 3 sessions (one capture series each). `--massive` adds the option chain snapshot, which is a current-state endpoint: it can only show `day.last_updated` and `last_quote.last_updated` freezing after the close (`compare` uses the larger of the two; all pages of a ticker are compared as one snapshot, so an added or changed page counts as a content change), so run it on the same schedule but treat it as secondary. If `--massive` returns a plan error, record that and skip it. Post the `compare` output (metadata only) to #877.

`capture` exits 1 if any request failed or a Massive result was truncated at the 20-page cap, but still writes every unit; rerun a failed label rather than comparing around it. Cost: about 4 labels x 5 tickers x 2 endpoints = 40 ORATS GETs per session; no retries. Not a heavy job.
