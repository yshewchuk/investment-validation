# `engine/v2/dashboard`

## Ownership

Implements the **UI — navigation, formatting, tables, charts, loading/error states** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**8** of §4.1.

Replaces (§4.4): `the formatting half of dashboard/render.py`, `dashboard/static/`.

## Responsibilities

- Navigation, formatting, tables, charts, loading and error states.

## Non-responsibilities

- **Compute gates, financial ratios or return estimates** — `engine/v2/serving` does it instead.
- **Do portfolio accounting** — `engine/v2/evaluation` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

`preview.main`, `preview.run`, `preview.is_loopback`, `preview.resolve_release_id`
and `preview.TOKEN_ENV_VAR` — the P3-0 compatibility preview launcher. It
composes only `engine.v2.serving.operations.create_server` (this package's "7
only" import rule); it does not implement serving or rendering itself.

<!-- public-interface: preview -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

_Nothing yet — no package imports this one. The first importer is added here in the same commit._

<!-- consumers: none -->

## Usage

Start the compatibility preview against a release root (real or synthetic)
and a health artifact, loopback-only:

    V2_DASHBOARD_TOKEN=secret python3 -m engine.v2.dashboard.preview \
        --host 127.0.0.1 --port 8765 \
        --release-root /path/to/release_root --health-path /path/to/health.json

It refuses to start with no `V2_DASHBOARD_TOKEN` set, and refuses a
non-loopback `--host` unless `--allow-non-loopback` is also passed. It never
prints the token; it prints the URL and the release id resolved once from the
server's own `/release/current.json`.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes. `tests/test_v2_dashboard_preview.py` builds its own
synthetic release bundles under `tmp_path` rather than using that corpus, since
P3-0 has no real published release yet.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
