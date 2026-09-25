"""UI navigation, formatting, tables, charts, loading/error states

Layer 8 of `system_rearchitecture.md` §4.1. Replaces `the formatting half of dashboard/render.py`, `dashboard/static/`.

P3-0 (`guides/rearchitecture_phase3_parity_launch.md` §8) adds the first
production code: `preview.py`, the compatibility preview launcher, which
composes `engine.v2.serving` (via the package-internal `_server.py`) per this
package's "7 only" import rule, and — only for an explicit `--ops-root` — wires
`engine.v2.ops.cli.refresh_action` onto the operations server's refresh route.
See ``README.md`` for the public interface and which packages may import it.
"""
