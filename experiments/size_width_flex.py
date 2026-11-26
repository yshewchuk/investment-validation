"""EXP-189 W6b: nearest-feasible fallback arm in both stages, and the T8/T9 markdown (design 12.3 to 12.4)."""
import resource
import sys
import time

from engine.pnl_sim import DRAWS
from experiments.size_pnl_report import md_table
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_band import PHIS, adopt, choose_phi
from experiments.size_width_fallback import FallbackSweep, accounts, recovery, t9
from experiments.size_width_sweep import E_ARM, INNER, WINDOWS

LABELS = ("production", "A", "B", "C", "D")  # arms with fixed per-structure factors (E has none)


def curves(tuned_b, tuned_c, anchor, min_trades):
    """Per structure the B, C (when tuned) and anchor curves band() reads; anchor cells under ``min_trades`` are not_scored."""
    return {name: {"B": tuned_b[name]["scores"], **({"C": tuned_c[name]["scores"]} if tuned_c is not None else {}),
                   "anchor": {k: {**cell, "not_scored": cell["trades"] < min_trades} for k, cell in anchor[name].items()}}
            for name in NAMES}


def target(rec):
    """The label whose factors design 12.3 uses: production if nothing is recommended, the arm if it has fixed factors, else None."""
    return "production" if rec is None else (rec if rec in LABELS else None)


def _accounts_ok(accounts, windows):
    return all(isinstance(accounts[w][k], (int, float)) and not isinstance(accounts[w][k], bool) for w in windows for k in ("fallback", "base"))


def _t8_ok(t8):
    return all(isinstance(t8[k], list) for k in ("quantiles", "buckets", "declines")) and "deviation_unavailable" in t8


def _inner_ok(got, factors):
    phis, choice = got["phis"], got["choice"]
    return (got["factors"] == {n: float(factors[n]) for n in NAMES} and (choice["phi"] is None or choice["phi"] in PHIS)
            and isinstance(choice["flags"], list)
            and all(isinstance(phis[str(p)]["recovery"], dict) and isinstance(phis[str(p)]["t9"], list)
                    and _accounts_ok(phis[str(p)]["accounts"], ("2023H2", "2024")) for p in PHIS))


def _set_ok(got, inner_label):
    return (got["factors"] == inner_label["factors"] and got["choice"] == inner_label["choice"]
            and all(isinstance(got["phis"][str(p)]["t9"], list) and {"fallback", "base"} <= set(got["phis"][str(p)]["alpha"])
                    and _accounts_ok(got["phis"][str(p)]["accounts"], ("2023H2", "2024", "2025")) for p in PHIS))


def admit(inner_summary, bands, frozen, outer_body=None):
    """Refuses unless the fallback summaries belong to the frozen receipts: the inner summary (and ``outer_body`` when given) holds
    the tuning_band ``bands`` and the T8 tables, and its flex has each fixed-factor label of ``frozen`` with matching factors and
    well-formed results; the outer body's sets match the inner labels."""
    try:
        flex_in = inner_summary["flex"]
        labels = {l for l in frozen if l in LABELS}
        ok = (inner_summary["bands"] == bands and _t8_ok(inner_summary["t8"]) and set(flex_in) == labels
              and all(_inner_ok(flex_in[l], frozen[l]) for l in labels))
        if outer_body is not None:
            fx = outer_body["flex"]
            ok = ok and (_t8_ok(outer_body["t8"]) and {"recommendation", "factors_label"} <= set(fx)
                         and isinstance(fx["adoption"], dict) and "adopt" in fx["adoption"]
                         and isinstance(fx["sets"], dict) and "production" in fx["sets"] and set(fx["sets"]) <= set(flex_in)
                         and all(_set_ok(fx["sets"][l], flex_in[l]) for l in fx["sets"]))
    except (KeyError, TypeError, AttributeError, ValueError):
        ok = False
    if not ok:
        raise Refusal("EXPERIMENT_IDENTITY_CONFLICT", "fallback summaries do not match the frozen receipts", {"reason": "tuning_receipt_changed"})


def _run(factor_sets, months, total, out, inputs_hash, sessions, pricing, ladder, tag, store=None):
    """{label: {phi: FallbackSweep}}; identical factor sets share their sweeps, and each month is simulated once for all of them."""
    built, keys = {}, {label: tuple(float(factors[n]) for n in NAMES) for label, factors in factor_sets.items()}
    for label, factors in factor_sets.items():
        if keys[label] not in built:
            built[keys[label]] = {phi: FallbackSweep(factors, out, inputs_hash, phi, pricing=pricing, ladder=ladder) for phi in PHIS}
    sweeps = [sweep for by_phi in built.values() for sweep in by_phi.values()]
    for sweep in sweeps: sweep.store = store
    started = time.monotonic()
    print(f"[width_flex] {tag} start sets={len(factor_sets)} sweeps={len(sweeps)} months={total}", file=sys.stderr)
    for done, (chunk, events, output, pool) in enumerate(months, start=1):
        sims = pricing.simulate(chunk, events, pool, draws=DRAWS)
        for sweep in sweeps:
            sweep.add_month(chunk, events, pool, output=output, sessions=sessions, sims=sims)
        if store is not None: store.flush(chunk)
        elapsed = max(time.monotonic() - started, 1e-9)
        print(f"[width_flex] {tag} month {done}/{total} rate={done / elapsed * 60:.2f}/min eta={(total - done) * elapsed / done:.0f}s "
              f"rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB", file=sys.stderr)
    print(f"[width_flex] {tag} end elapsed={time.monotonic() - started:.1f}s "
          f"rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB", file=sys.stderr)
    return {label: built[key] for label, key in keys.items()}


def inner(factor_sets, months, total, base, out, inputs_hash, sessions, pricing):
    """The 2023-24 stage per fixed-factor arm: per phi the recovery, accounts and T9, and the phi chosen from 2023-24 alone."""
    chosen = {label: f for label, f in factor_sets.items() if label in LABELS}
    result = {}
    for label, by_phi in _run(chosen, months, total, out, inputs_hash, sessions, pricing, False, "inner", getattr(base, "store", None)).items():
        rec = {phi: recovery(fb, base, INNER) for phi, fb in by_phi.items()}
        result[label] = {"factors": {n: float(k) for n, k in chosen[label].items()}, "choice": choose_phi(rec),
                         "phis": {str(phi): {"recovery": rec[phi], "accounts": accounts(fb, base, {w: WINDOWS[w] for w in ("2023H2", "2024")}),
                                             "t9": t9(fb, base, phi, INNER)} for phi, fb in by_phi.items()}}
    return result


def outer(frozen, flex_inner, rec, months, total, base, out, inputs_hash, sessions, pricing, stats):
    """The 2025 stage on the recommended factors, with the inner phi (flex_inner was admitted against the frozen factors by admit, before this)."""
    label_t = target(rec)
    labels = ["production"] + ([label_t] if label_t not in (None, "production") else [])
    window = WINDOWS["2025"]
    fbs = _run({label: frozen[label] for label in labels}, months, total, out, inputs_hash, sessions, pricing, True, "outer", getattr(base, "store", None))
    sets = {}
    for label in labels:
        phis = {}
        for phi in PHIS:
            fb = fbs[label][phi]
            phis[str(phi)] = {"accounts": {**flex_inner[label]["phis"][str(phi)]["accounts"], **accounts(fb, base, {"2025": window})},
                              "alpha": {"fallback": stats(fb, {n: E_ARM for n in NAMES}, window)["x"], "base": stats(base, fb.factors, window)["x"]},
                              "t9": t9(fb, base, phi, window)}
        sets[label] = {"factors": {n: float(frozen[label][n]) for n in NAMES}, "choice": flex_inner[label]["choice"], "phis": phis}
    phi = None if label_t is None else sets[label_t]["choice"]["phi"]
    cell = None if phi is None else sets[label_t]["phis"][str(phi)]
    adoption = ({"adopt": False, "phi": None, "reason": "recommended_arm_e_has_no_fixed_factors"} if label_t is None
                else adopt(phi, cell and cell["accounts"], cell and cell["alpha"]))
    return {"recommendation": rec or "no_change", "factors_label": label_t, "sets": sets, "adoption": adoption}


def _table(title, rows):
    """A markdown section: ``rows`` as a table over the first row's keys, or "none"."""
    if not rows:
        return f"#### {title}\n\nnone\n"
    headers = list(rows[0])
    return f"#### {title}\n\n" + md_table(headers, [[row.get(h) for h in headers] for row in rows]) + "\n"


def render(inner_summary, outer_body):
    """The T8-T9 markdown from the inner summary and the outer body (2025 NOT blind for widths)."""
    parts = ["### T8-T9 width flexibility (2025 NOT blind for widths)\n"]
    for key, title in (("quantiles", "T8 realised/target width quantiles"), ("buckets", "T8 return by deviation bucket (descriptive only)"),
                       ("declines", "T8 decline shares by reason and tier")):
        parts.append(_table(title, inner_summary["t8"][key] + outer_body["t8"][key]))
    parts.append(f"deviation_unavailable: 2023-24={inner_summary['t8']['deviation_unavailable']}, 2025={outer_body['t8']['deviation_unavailable']}\n")
    bands = [{"structure": n, "curve": label, "phi": phi, "lo": cell["lo"], "hi": cell["hi"], "peak_k": cell["peak_k"],
              "flags": ",".join(cell["flags"]), "primary": label == inner_summary["bands"][n]["primary"]}
             for n in NAMES for label, curve in inner_summary["bands"][n]["curves"].items() for phi, cell in curve.items()]
    parts.append(_table("T9 tolerance bands (k is a ratio to production; the anchor curve's k is a ratio to the actual move)", bands))
    fx = outer_body["flex"]
    for label, s in fx["sets"].items():
        t9_rows, acc_rows = [], []
        for phi in (str(p) for p in PHIS):
            t9_rows += [{"window": "2023-24", **r} for r in inner_summary["flex"][label]["phis"][phi]["t9"]]
            t9_rows += [{"window": "2025", **r} for r in s["phis"][phi]["t9"]]
            for w in ("2023H2", "2024", "2025"):
                a, alpha = s["phis"][phi]["accounts"][w], s["phis"][phi]["alpha"] if w == "2025" else {}
                acc_rows.append({"phi": phi, "window": w, "fallback": a["fallback"], "base": a["base"],
                                 "alpha_fallback": alpha.get("fallback"), "alpha_base": alpha.get("base")})
        parts.append(_table(f"T9 fallback on {label} factors", t9_rows))
        parts.append(_table("accounts and 2025 breakeven alpha", acc_rows))
        parts.append(f"{label}: phi chosen from 2023-24 is {s['choice']['phi']} (flags: {', '.join(s['choice']['flags']) or 'none'}).\n")
    ad = fx["adoption"]
    details = ", ".join(f"{k}={ad[k]}" for k in ("reason", "beats", "alpha_ok", "phi") if ad.get(k) is not None)
    verdict = f"Adoption: yes at phi={ad['phi']}." if ad["adopt"] else f"Adoption: no ({details})."
    parts.append(f"{verdict} Recommendation: {fx['recommendation']}; factors_label: {fx['factors_label']}.\n")
    return "\n".join(parts)
