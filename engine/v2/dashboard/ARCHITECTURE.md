# `engine/v2/dashboard` — architecture

Layer 8.0 in the root `/ARCHITECTURE.md` layer table, alongside `ui/`.
Replaces the formatting half of legacy `dashboard/render.py` and
`dashboard/static/`. See the companion legacy doc,
`engine/dashboard/ARCHITECTURE.md`, for the board this package does not
replace yet. This package never writes `dashboard/published/**` — it only
composes and starts the serving preview; the legacy package owns
publication to that path (`publish_bundle`), which is excluded from
CodeRabbit review by `.coderabbit.yaml`'s `path_filters` and from the
secret scan by `checks/repo_hygiene.py`.

## Purpose

UI composition only: the native operations-server preview launcher
(`preview.py`) and its server-composition helper (`_server.py`). It
formats and serves what `engine/v2/serving` already computed; it never
computes a gate, a financial ratio, a return estimate, or portfolio
accounting itself.

## Primary contracts and public interfaces

- `preview.py` — the preview launcher entrypoint (see its own module
  docstring/CLI for exact flags).
- `_server.py::build_server` — composes `engine.v2.serving.operations`'s
  `create_server` with this launcher's health/release/calibration paths
  and (optionally) a refresh callback. Package-internal, not itself a
  cross-package public name.

## Inputs

Release root, model-release root, calibration health path and serving
index path — all read via `engine.v2.serving`'s bounded/paginated reads,
never a direct file read of scoring/evaluation/ledger data.

## Outputs

The operations-server HTTP responses `create_server` builds (health,
release data, calibration health) and, when a refresh root is configured,
queued refresh job ids from `POST /actions/refresh`.

## Dependencies

`only_imports=(7.0,)` (root doc §2): this package may import anything on
layer 7.0 — `engine.v2.serving` (used directly, `_server.py`'s
`create_server` import) and `engine.v2.ops`/`engine.v2.research` (allowed
by the layer rule, but not otherwise imported here) — and nothing else in
`engine.v2`: no direct import of `scoring`, `evaluation`, `ledger`,
`domain`, `models`, or `data` (layers 0-6), and no import of
`engine.v2.diagnosis` (7.5, the sink; imported by nothing).

The one ops import in this package is `_server.py`'s lazy
`from engine.v2.ops.cli import refresh_action`, taken only inside
`_refresh_callback` and only when an explicit `ops_root` is configured —
a plain read-only preview never loads the supervisor. It calls
`refresh_action` directly (never starts the supervisor loop, never runs a
refresh inline); the call queues a shadow nightly plan submission and
returns job ids. This is the one documented exception the root doc §4
cites: no other production package imports `engine.v2.ops` Python modules
directly.

Callers: the preview launcher is invoked by an operator (CLI/manual), not
by another production package; `preview.py` itself imports
`_server.build_server`.

## External systems and libraries

An HTTP server (`engine.v2.serving.operations.create_server`'s listener);
no direct filesystem, database or third-party API access of its own.

## Failure semantics

- **Missing input** — a missing release/calibration/serving-index path is
  handled by `engine.v2.serving`'s own typed responses (e.g. the read-only
  503 "refresh not configured" `create_server` returns when no refresh
  callback is wired); this package adds no new missing-input handling.
- **Cache / retry / transaction / partial write** — none: this package
  holds no durable state of its own; every read goes through
  `engine.v2.serving`'s own semantics.
- **Idempotency** — `POST /actions/refresh` submissions are idempotent at
  the `engine.v2.ops` layer (job identity keyed on session/scope/stage);
  this package passes the payload through unchanged.

## Invariants

- Layer 8 `only_imports=(7.0,)` (above), enforced by `checks/import_layers.py`.
- No gate, financial ratio, return estimate, or portfolio accounting
  computed here — read from `engine.v2.serving` instead (root doc §5,
  native-vs-legacy provenance's sibling rule for this layer).
- Free-text fields that can reach the published bundle (a
  `degraded_reason`, a flag's `detail` string, an exception's `str()`) are
  a leak path and must be sanitised before they are written — the same
  rule as "nothing published carries a local path or raw exception text"
  (root doc §5), not exempt just because a string looks short.
- Any hardcoded or developer-local filesystem path is disallowed; paths
  resolve through `engine.paths`/the v2 foundation, never a module's own
  `Path(__file__)`-derived root.
