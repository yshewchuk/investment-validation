"""Phase 3 — the monitoring dashboard pipeline (renderer-first).

One rendering path produces a self-contained bundle from scorer output; the
local FastAPI app serves that bundle; the publisher ships the same bundle to
the remote host. Phone view, desk view, and engine therefore cannot drift:

    engine/dashboard/render.py      score_calendar() output -> snapshot bundle
    engine/dashboard/selfcheck.py   bundle vs direct engine.score re-score
    engine/dashboard/publish.py     bundle -> target, atomic, access-checked
    engine/dashboard/nightly.py     refresh -> validate -> score -> ledger ->
                                    render -> selfcheck -> publish -> flags

The bundle layout the renderer writes (and the server serves)::

    dashboard/earnings/
      index.html  assets/app.js  assets/app.css
      data/board.json   data/board.js
      data/meta.json    data/meta.js
      data/health.json  data/health.js
      data/flags.json   data/flags.js
      data/tickers/{T}.json  data/tickers/{T}.js

Every datum travels twice — as ``.json`` (the contract the self-check and the
API read) and as a ``.js`` wrapper (so the same bundle opens from ``file://``
where ``fetch()`` is blocked). The wrapper is generated from the JSON bytes,
never written independently, so the two cannot disagree.
"""
from engine.dashboard.render import (
    BOARD_MAX_BYTES,
    build_health,
    build_meta,
    compact_row,
    freshness_summary,
    quota_state,
    render_bundle,
    size_model_mae_from_ledger,
)
# `selfcheck` (the function) is deliberately NOT re-exported: it would shadow
# the submodule of the same name, and then `import engine.dashboard.selfcheck`
# hands back a function. Import it as
# `from engine.dashboard.selfcheck import selfcheck`. Same rule everywhere in
# this package — no export may collide with a submodule name, which is why the
# publisher's entry point is `publish_bundle` rather than `publish`.
from engine.dashboard.selfcheck import SelfCheckReport
from engine.dashboard.publish import (
    LocalPublisher,
    PublishError,
    access_probe,
    publish_bundle,
    secret_scan,
)
# NOT imported here: engine.dashboard.nightly. The orchestrator is invoked as
# `python3 -m engine.dashboard.nightly`, and a package that has already imported
# the module being run makes Python warn about it on every cron night. Import it
# by module path (`from engine.dashboard.nightly import run_nightly`).

__all__ = [
    "BOARD_MAX_BYTES",
    "LocalPublisher",
    "PublishError",
    "SelfCheckReport",
    "access_probe",
    "build_health",
    "build_meta",
    "compact_row",
    "freshness_summary",
    "publish_bundle",
    "quota_state",
    "render_bundle",
    "secret_scan",
    "size_model_mae_from_ledger",
]
