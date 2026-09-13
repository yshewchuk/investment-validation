# Rearchitecture tech debt

Deferral rule (user decision, 2026-09-13): work that any later
rearchitecture phase needs is finished in its own phase. Only nice-to-haves
that can safely wait until **after Phase 6** are recorded here. Nothing on
this list may be a prerequisite of Phases 3–6. If a phase guide comes to need
an entry, move it into that phase and delete it here.

Each entry: what was deferred, why it can wait, the risk while it waits, and
what should trigger revisiting it.

| ID | Item | Origin | Why deferrable | Risk while deferred | Revisit when |
|---|---|---|---|---|---|
| TD-1 | Process-local object-verification cache keyed by `(device, inode, size, mtime_ns)` (Phase 2 guide §8.2 step 6). | P2-4 planning | The repository re-hashes every object it opens. Correctness is unchanged; only speed is lost. | Slower scans of large `option_chains` fragments. | A read path measurably exceeds its time budget, or live latency matters (Phase 7). |
| TD-2 | `TimeInterval` start < end compares canonical strings. | P2-1a | Correct for dates, naive timestamps, and UTC timestamps produced by `format_timestamp`. | A non-UTC offset written by hand would compare wrongly. | Any contract accepts caller-supplied offsets other than UTC. |
| TD-3 | `artifact_check`'s `CheckParameters` carries an optional `input_bindings` field used only by the cache-identity tests. | P2-5 B1a | A test seam on a production kind; no production caller sets it. | A production caller could start depending on it. | A non-legacy production kind exists that those tests can use instead. |
| TD-4 | Code shaped to fit the eight-module import fan-out budget. `engine/v2/data/legacy_adapter.py` builds contracts in two passes and hard-codes three legacy paths (tier-0 tested against `engine.paths`). `engine/v2/data/objects.py` uses `os`-only equivalents of `stat`/`uuid`/`math`. | P2-1b, P2-2b | Behaviour is correct and tested; the cost is readability. | Future edits keep contorting around the budget. | The fan-out budget is revisited, or these modules are split. |
