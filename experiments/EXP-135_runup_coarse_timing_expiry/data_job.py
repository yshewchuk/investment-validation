"""Offline planning and resumable paid acquisition for EXP-135."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine import paths
from engine.calendar import trading_calendar
from engine.data import store
from engine.data.fetch import Fetcher, CredentialRotated, FetchError, iter_cached
from engine.data.throttle import Throttle, SOURCES, latest_quota, QuotaExhausted
from engine.data.pulls.sep2026_plan import FIELDS
from experiments import lib

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OFFSETS = (21, 14, 7, 3)
BUDGET = 8500
START = time.monotonic()
def log(message):
    print(f"[EXP-135 {time.monotonic()-START:.0f}s] {message}", flush=True)

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)

def get_events():
    path = RESULTS / "events.parquet"
    if path.exists():
        return pd.read_parquet(path)
    log("Freezing existing run-up cohort")
    columns = ["ticker", "event_id", "event_date", "entry_date", "exit_date",
               "strategy", "variant", "provenance", "fill_alpha", "entry_cost",
               "exit_value", "ret", "legs"]
    t = store.read_table("trades", columns=columns)
    t = t[(t.strategy == "STR-RUNUP") & (t.provenance == "engine.replay")
          & (t.variant == "e-14_x+0_target_dte=30")].copy()
    t.to_parquet(RESULTS / "control.parquet", index=False)
    t = t[np.isclose(t.fill_alpha.astype(float), .5)].drop_duplicates(["ticker", "event_date"])
    assert len(t) == 6607, f"Frozen universe changed: {len(t)}"
    t = t[["ticker", "event_id", "event_date", "exit_date"]].sort_values(["event_date", "ticker"])
    # The existing replay exit identifies the session unambiguously without
    # a refreshed calendar silently changing this previously costed cohort.
    t["session"] = np.where(t.event_date == t.exit_date, "AMC", "BMO")
    cal = trading_calendar()
    for j in OFFSETS:
        t[f"d{j}"] = [cal.shift(pd.Timestamp(d), -j) for d in t.exit_date]
    t.to_parquet(path, index=False)
    manifests = []
    for p in sorted(paths.curated_table("trades").glob("year=*/*.parquet")):
        manifests.append({"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    write_json(RESULTS / "input_manifest.json", {"events": len(t), "inputs": manifests,
               "frozen_at": datetime.now(timezone.utc).isoformat()})
    spec = lib.load_spec(HERE / "spec.yaml")
    sha = lib.spec_hash(spec)
    ledger = lib.ledger_read()
    if not ((ledger.id == "EXP-135") & (ledger.spec_hash == sha)).any():
        lib.ledger_append([dict(id="EXP-135", spec_hash=sha,
             date=datetime.now(timezone.utc).strftime("%Y-%m-%d"), stage="planned",
             oos_mean_mid="", sharpe_trade="", promoted="False")])
    return t

def batches(pairs):
    by_date = defaultdict(set)
    for ticker, date in pairs:
        by_date[str(date)[:10]].add(ticker)
    jobs = []
    for date, tickers in sorted(by_date.items()):
        ts = sorted(tickers)
        for i in range(0, len(ts), 10):
            jobs.append({"ticker": ",".join(ts[i:i+10]), "tradeDate": date,
                         "dte": "1,100", "fields": FIELDS})
    return jobs

def plan():
    RESULTS.mkdir(parents=True, exist_ok=True)
    if (RESULTS / "pull_plan.json").exists():
        result = json.loads((RESULTS / "pull_plan.json").read_text())
        log(f"Frozen plan: {result['calls']} broad entry-chain requests")
        return result
    t = get_events()
    entries = {(r.ticker, str(getattr(r, f"d{j}").date())) for r in t.itertuples() for j in OFFSETS}
    exits = {(r.ticker, str(r.exit_date.date())) for r in t.itertuples()}
    # Request-range metadata is needed: presence of one expiry in the store
    # does not prove the early snapshot includes nearer expiries.
    have = set()
    log("Checking raw metadata for complete 1-100 DTE entry snapshots")
    for record in iter_cached("orats", "hist/strikes"):
        p = record.params
        dte = str(p.get("dte", "")).split(",")
        if len(dte) == 2 and float(dte[0]) <= 1 and float(dte[1]) >= 100 and not p.get("delta"):
            for ticker in str(p.get("ticker", "")).split(","):
                have.add((ticker, str(p.get("tradeDate", ""))[:10]))
    jobs = batches(entries - have)
    result = dict(events=len(t), entry_pairs=len(entries), reusable_pairs=len(entries & have),
                  calls=len(jobs), jobs=jobs, broad_all_five_dates_upper_calls=len(batches((entries|exits)-have)),
                  quota=latest_quota(), offsets=OFFSETS,
                  note="Exit top-ups planned after pinning actual contracts; existing exit snapshots reused.")
    write_json(RESULTS / "pull_plan.json", result)
    log(json.dumps({k:v for k,v in result.items() if k != "jobs"}))
    return result

def request_history():
    path = RESULTS / "requests.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return records

class BudgetThrottle(Throttle):
    def __init__(self, already):
        super().__init__()
        self.used = already
    def acquire(self, source):
        if source == "orats":
            if self.used >= BUDGET:
                raise QuotaExhausted("EXP-135 request ceiling reached")
            state = latest_quota()
            if state["remaining"] is not None and state["remaining"] <= 3006:
                raise QuotaExhausted("Preserving the 3,000-call operating reserve")
            self.used += 1
        return super().acquire(source)

def pull(jobs, limit=None):
    # A process-level lock prevents two invocations spending the same plan.
    import fcntl
    lock = open(RESULTS / ".pull.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    history = request_history()
    done = {r["signature"] for r in history if r["status"] in ("ok", "absent", "split")}
    prior_spend = sum(r.get("network_calls", 0) for r in history)
    throttle = BudgetThrottle(prior_spend)
    fetcher = Fetcher(throttle=throttle)
    journal = open(RESULTS / "requests.jsonl", "a", buffering=1)
    state = {"jobs_done":0, "jobs_total":len(jobs), "last":"starting"}
    stopping = threading.Event()
    def heartbeat():
        while not stopping.wait(30):
            log(f"Pull {state}; network calls this process={fetcher.network_calls}; quota={latest_quota()['remaining']}")
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    def save(value):
        journal.write(json.dumps(value, default=str) + "\n")
        journal.flush()
    def fetch_job(params):
        signature = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        if signature in done:
            return
        wanted = params["ticker"].split(",")
        before = fetcher.network_calls
        state["last"] = params["tradeDate"] + " " + params["ticker"]
        try:
            record = fetcher.fetch("orats", "hist/strikes", params, note="EXP-135")
            own_calls = fetcher.network_calls-before
            doc = record.json()
            if not isinstance(doc, dict) or not isinstance(doc.get("data"), list):
                raise ValueError("Unexpected ORATS response envelope")
            rows = doc["data"]
            if len(wanted) == 1:
                for row in rows:
                    row.setdefault("ticker", wanted[0])
            got = {r.get("ticker") for r in rows}
            missing = set(wanted) - got
            status = "ok"
            # Verify all requested symbols. Retry the whole partial batch in
            # smaller pieces, including the last returned ticker which might
            # itself have been truncated.
            if missing and len(wanted) > 1:
                log(f"Partial response: {len(got)}/{len(wanted)} tickers; splitting batch")
                middle = len(wanted)//2
                for subset in (wanted[:middle], wanted[middle:]):
                    child = dict(params, ticker=",".join(subset))
                    fetch_job(child)
                status = "split"
            elif missing:
                status = "absent"
            save(dict(signature=signature, status=status, key=record.key, params=params,
                      rows=len(rows), bytes=len(record.body), returned=sorted(got),
                      missing=sorted(missing), network_calls=own_calls,
                      own_network_call=int(not record.from_cache)))
            done.add(signature)
        except CredentialRotated:
            raise
        except FetchError as exc:
            own_calls = fetcher.network_calls-before
            # Never persist or print a transport exception containing an URL.
            if "HTTP 404" not in str(exc):
                save(dict(signature=signature, status="error", params=params,
                          error_type=type(exc).__name__, network_calls=fetcher.network_calls-before))
                raise
            if len(wanted) > 1:
                middle = len(wanted)//2
                for subset in (wanted[:middle], wanted[middle:]):
                    fetch_job(dict(params, ticker=",".join(subset)))
                status = "split"
            else:
                status = "absent"
            save(dict(signature=signature, status=status, params=params,
                      network_calls=own_calls, own_network_call=1))
            done.add(signature)
    try:
        for i, job in enumerate(jobs, 1):
            if limit is not None and i > limit:
                break
            fetch_job(job)
            state["jobs_done"] = i
            if i % 10 == 0 or i == 1:
                elapsed = time.monotonic()-START
                log(f"Pull {i}/{len(jobs)}; calls={fetcher.network_calls}; elapsed={elapsed:.0f}s; quota={latest_quota()['remaining']}")
    finally:
        stopping.set()
        journal.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        write_json(RESULTS / "quota_status.json", dict(last_quota=latest_quota(),
                   process_network_calls=fetcher.network_calls, state=state,
                   updated_at=datetime.now(timezone.utc).isoformat()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repair-plan", type=Path)
    args = parser.parse_args()
    result = plan()
    if args.confirm:
        jobs = json.loads(args.repair_plan.read_text()) if args.repair_plan else result["jobs"]
        pull(jobs, args.limit)
    else:
        log("Dry run only: no API calls")
if __name__ == "__main__":
    main()
