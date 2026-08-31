"""Earnings-vol monitoring board — FastAPI, port 8712.

Renderer-first: this server is a convenience around the SAME bundle the
nightly job renders and the publisher ships. It serves ``dashboard/earnings/``
statically and adds desk-only endpoints (manual re-score, refresh) that never
exist in the published snapshot — the static bundle has no mutating surface by
construction.

Hard rules from the guide, enforced here:

* **Binds 127.0.0.1 by default.** The cloudflared tunnel is the sole remote
  path to this app. ``DASHBOARD_HOST`` / ``DASHBOARD_PORT`` override it, which
  a containerised desk needs — a published Docker port forwards to the
  container's bridge interface, not its loopback, so a 127.0.0.1 bind inside a
  container is unreachable from the host. Overriding is a deliberate act with a
  real consequence: the board discloses position intent and redistributes
  licensed ORATS-derived quotes, so whoever can reach the bind can read both.
  The default stays loopback so that is never the accident.
* **Quota-spending actions are local-only.** ``POST /api/refresh`` shells out
  to the nightly job with ``--no-publish``.
* The semis scanner on 8711 is untouched.

Run::

    python3 dashboard/earnings_app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

BUNDLE = HERE / "earnings"
BUNDLE.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optionally warm the scorer at startup — OFF by default.

    Building a ``Scorer`` loads the panel, the trades, the champions and the
    calendar: a couple of minutes and ~1.4 GB, held for the life of the process.
    Warming it makes the first ad-hoc ``/api/score`` instant instead of
    appearing to hang.

    It is off by default because the cost lands on the wrong job. Serving the
    board needs NO scorer — the bundle is already rendered on disk — while the
    nightly needs to build one of its own, and two at once exceeded this
    machine's memory and the nightly was OOM-killed with an empty log. The
    board is the product; an ad-hoc re-score paying two minutes on first use is
    the cheaper trade. Set ``DASHBOARD_WARM=1`` on a box with room to spare.
    """
    if os.environ.get("DASHBOARD_WARM") == "1":
        def warm():
            try:
                from engine.score import _scorer

                _scorer()
            except Exception:  # a cold cache must never stop the server serving
                pass

        threading.Thread(target=warm, name="scorer-warmup", daemon=True).start()
    yield


app = FastAPI(title="Earnings-Vol Monitoring Board", lifespan=lifespan)


def _data(name: str) -> JSONResponse:
    path = BUNDLE / "data" / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"{name} missing — run `python3 -m engine.dashboard.nightly` first")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/meta")
def meta():
    return _data("meta")


@app.get("/api/board")
def board():
    return _data("board")


@app.get("/api/health")
def health():
    return _data("health")


@app.get("/api/flags")
def flags():
    return _data("flags")


@app.get("/api/ticker/{sym}")
def ticker(sym: str):
    sym = sym.upper()
    path = BUNDLE / "data" / "tickers" / f"{sym}.json"
    if not path.exists():
        raise HTTPException(404, f"no bundle data for {sym} — it may not be on the board")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/score")
def score(ticker: str, strategy: str, strike: float | None = None, expiry: str | None = None):
    """Desk-only ad-hoc re-score of one (ticker, strategy) through the engine.

    The published bundle cannot do this: it is static by design. This endpoint
    exists so the desk can probe a strike the nightly ladder did not carry.
    """
    from engine.score import UNSCORABLE
    from engine.score import score as engine_score

    try:
        result = engine_score(
            ticker.upper(),
            strategy.upper(),
            strike=strike,
            expiry=expiry,
        )
    except UNSCORABLE as exc:
        # The same exceptions the calendar turns into NO_CHAIN rows: an event
        # with no chain in the store, or a structure that cannot be built at
        # this strike. Not a server fault — there is nothing to price.
        raise HTTPException(404, f"{type(exc).__name__}: {exc}") from exc
    return JSONResponse(result.as_dict())


@app.post("/api/refresh")
def refresh():
    """Re-run the nightly pipeline without publishing (desk-only action)."""
    proc = subprocess.run(
        [sys.executable, "-m", "engine.dashboard.nightly", "--no-refresh", "--no-publish"],
        capture_output=True, text=True, timeout=1800, cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    if proc.returncode != 0:
        return JSONResponse(
            {"ok": False, "log": (proc.stdout + proc.stderr)[-2000:]}, status_code=500
        )
    return {"ok": True, "log": proc.stdout[-2000:]}


# The bundle itself, served LAST so /api/* routes win.
app.mount("/", StaticFiles(directory=str(BUNDLE), html=True), name="bundle")

#: Loopback and 8712 unless the environment says otherwise. 8711 belongs to the
#: semis scanner; the two dashboards do not share a port.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8712

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("DASHBOARD_HOST", DEFAULT_HOST),
        port=int(os.environ.get("DASHBOARD_PORT", DEFAULT_PORT)),
    )
