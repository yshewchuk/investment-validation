#!/usr/bin/env python3
"""EXP-132 figures. Separate from run.py so a redraw cannot re-fit anything."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "results" / "metrics.json").read_text())
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)

rows = []
for name, b in d.get("models", {}).items():
    for arm in ("orats", "drop_entirely"):
        m = b.get(arm)
        if m: rows.append((name, arm, m["delta_mae"], b["seed_noise_mae"], "MAE"))
i = d.get("implied_t1") or {}
for arm in ("orats", "drop_entirely"):
    if i.get(arm): rows.append(("opf_implied_t1_gbm", arm, i[arm]["delta_mae"], i["seed_noise_mae"], "MAE"))
for name, b in (d.get("gates") or {}).items():
    for arm in ("orats", "drop_entirely"):
        m = b.get(arm)
        if m: rows.append((name, arm, -m["delta_lift"], b["seed_noise_lift"], "lift"))

fig, ax = plt.subplots(figsize=(11, 5))
labels = [f"{n}\n{a}" for n, a, *_ in rows]
vals = [r[2] for r in rows]; noise = [r[3] for r in rows]
norm = [v / nz if nz else 0 for v, nz in zip(vals, noise)]
colours = ["#d62728" if v > 1 else ("#2ca02c" if v < -1 else "#999999") for v in norm]
ax.bar(range(len(rows)), norm, color=colours, alpha=0.85)
ax.axhline(1, ls="--", c="k", lw=0.8); ax.axhline(-1, ls="--", c="k", lw=0.8)
ax.axhline(0, c="k", lw=1)
ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("change ÷ this model's own seed noise\n(positive = worse)")
ax.set_title("EXP-132 — every champion, each against ITS OWN noise band\n"
             "grey = indistinguishable · green = better · red = worse", fontsize=10)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(FIG / "deltas_vs_noise.png", dpi=130); plt.close(fig)
print("figure:", FIG / "deltas_vs_noise.png")
