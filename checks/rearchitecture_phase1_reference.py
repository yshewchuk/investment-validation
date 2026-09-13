#!/usr/bin/env python3
"""Fresh-process direct legacy score reference for the private canary."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ["INVESTING_PLAN_ROOT"] = str(ROOT)
    import pandas as pd
    from engine.features import FeatureContext
    from engine.jsonio import json_safe
    from engine.score import UNSCORABLE, FillModel, Scorer, ScoreRequest, unscorable_result

    entries = json.loads(args.requests.read_text())
    requests = [entry["request"] for entry in entries]
    tickers = sorted({str(item["ticker"]) for item in requests})
    dates = [pd.Timestamp(item["as_of"]).year for item in requests]
    scorer = Scorer(context=FeatureContext.load(tickers, years=range(min(dates), max(dates) + 1)))
    fields = {field.name for field in dataclasses.fields(ScoreRequest)}
    date_fields = {"as_of", "event_date", "expiry", "chain_as_of"}
    rows = []
    for entry, request in zip(entries, requests):
        values = {key: value for key, value in request.items() if key in fields}
        for key in date_fields:
            if isinstance(values.get(key), str):
                values[key] = pd.Timestamp(values[key])
        if isinstance(values.get("fill"), dict):
            values["fill"] = FillModel(alpha=float(values["fill"]["alpha"]))
        parsed = ScoreRequest(**values)
        try:
            result = scorer.score(parsed)
        except UNSCORABLE as exc:
            result = unscorable_result(parsed, as_of=parsed.as_of,
                                       snapshot=scorer.snapshot, exc=exc)
        rows.append({"request_id": entry["canary_id"],
                     "record": json_safe(result.as_dict(), round_to=None)})
    args.output.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
