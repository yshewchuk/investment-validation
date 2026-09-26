# `engine/v2/dashboard` — architecture

Layer 8.0 in the root `/ARCHITECTURE.md` layer table, alongside `ui/`.
Replaces the formatting half of legacy `dashboard/render.py` and
`dashboard/static/`. See the companion legacy doc,
`engine/dashboard/ARCHITECTURE.md`, for the board this package does not
replace yet. This package never writes `dashboard/published/**` — it only
composes and starts the serving preview; the legacy package owns
publication to that path (`publish_bundle`), which `.coderabbit.yaml`'s
`path_filters` excludes from CodeRabbit review (it is a runtime artifact,
not committed source). It is not excluded from a secret scan, but two
distinct scans are involved, over two distinct artifacts, and neither is
`check_files` (the ordinary committed-source scan): legacy `publish_bundle`
calls `engine/dashboard/publish.py`'s `secret_scan` on the built bundle
before writing it to `dashboard/published/**`; separately, the v2
publication security gate (`engine/v2/ops/effects_graph.py`'s
`_security_gate`) calls `checks/repo_hygiene.py`'s `check_bundle` (via
`engine/v2/ops/legacy_adapter.py`'s `run_security_scan`) on its own bound
`bundle.tar` artifact before that generation's commit — not on
`dashboard/published/**`, which the v2 gate never writes.

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

Launcher contract (`preview.py::_parse_args`/`run`): `--release-root` and
`--health-path` are required by argparse — omitting either exits with
code 2 before any server is built. The `V2_DASHBOARD_TOKEN` env var is
required by `run()` — a missing token raises `SystemExit` before
`build_server`/`create_server` runs at all. The launcher is loopback-only
by default (`is_loopback`, true for a literal loopback address or
`localhost`; `0.0.0.0` — every interface, not one — is treated as NOT
loopback): `run()` refuses a non-loopback `--host` with `SystemExit`
("refusing non-loopback host ... without --allow-non-loopback") unless
`--allow-non-loopback` is also passed, mirroring `engine/v2/serving/api.py`'s
own `--allow-non-loopback` guard on its server entrypoint. `--model-release-root`,
`--calibration-health-path`, `--ops-root` and `--serving-index-path` are
all optional; each unlocks exactly one route (`/models/release.json`,
`/calibration-health.json`, `POST /actions/refresh`, `GET /analogs.json`
respectively) and none is ever inferred from `--release-root` or
`--health-path` — omitting one keeps that route's own explicit
"not configured" refusal (see Failure semantics). `--release-root`,
`--health-path`, `--model-release-root`, `--calibration-health-path` and
`--serving-index-path` are serving roots: they are read via
`engine.v2.serving`'s bounded/paginated reads, never by directly reading
scoring/evaluation/ledger data. `--ops-root` is not a serving read at
all — it names a job root, not a data root: when configured, `_server.py`'s
`_refresh_callback` passes it straight to `engine.v2.ops.cli.refresh_action`
to submit a shadow nightly plan (see Dependencies for what that call does
and does not do).

## Outputs

The operations-server HTTP responses `create_server` builds — health,
release data, calibration health, and, when `--serving-index-path` is
configured, `GET /analogs.json`'s per-event analog document (each
score/strategy's persisted analog row ids and count, keyed by
`release_id`/`event_id` query params, or a refusal — not configured,
unreadable, outdated, no rows for that event, or missing query params) —
and, when a refresh root is configured, queued refresh job ids from
`POST /actions/refresh`.

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

- **Missing input** — two distinct layers:
  - **Launcher-time, in this package**: a missing `--release-root` or
    `--health-path` is rejected by argparse (`SystemExit(2)`) before any
    server object exists; a missing `V2_DASHBOARD_TOKEN` raises
    `SystemExit` in `run()`, again before `build_server`/`create_server`
    is ever called. Neither reaches the serving layer.
  - **Serving-time, in `engine.v2.serving` (not this package)**: an
    omitted *optional* root (`--model-release-root`,
    `--calibration-health-path`, `--serving-index-path`) is passed
    straight through to `create_server`, whose own typed responses
    answer the request at call time — e.g. the read-only 503
    "refresh not configured" when no refresh callback is wired.
    `--ops-root` is the one exception: this package's own
    `_server.py::_refresh_callback` converts it to a bound
    `submit_refresh` callable, or to `None` when the root is missing or
    empty, before `create_server` ever sees it — `create_server` still
    answers the same read-only 503 when that callback is `None`. This
    package adds no other missing-input handling of its own beyond the
    two launcher checks above.
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
