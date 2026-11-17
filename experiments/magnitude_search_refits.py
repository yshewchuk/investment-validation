"""Per-candidate outer refit slots for three-winner search plans."""
from engine.v2.foundation import content_hash

__all__ = ["per_candidate", "refit_key", "refit_units", "possible_slots", "failed_terminal",
           "overall_key", "terminal_keys"]


def per_candidate(plan):
    """A plan uses one refit slot per candidate only when it seals winner rules."""
    return bool(plan.get("selection", {}).get("winner_rules"))


def refit_key(plan, year, candidate_id=None):
    """Legacy plans keep the fold key; winner plans add the refit candidate id."""
    if candidate_id is None:
        return content_hash([plan["plan_id"], year])
    return content_hash([plan["plan_id"], year, candidate_id])


def refit_units(plan, selection_item):
    """Ordered distinct refit units: overall winner, other winners, extras."""
    if not per_candidate(plan):
        return [None]
    overall = selection_item["selected_candidate_id"]
    units, seen = [overall], {overall}
    for candidate_id in sorted(selection_item.get("distinct_winner_ids", [])):
        if candidate_id not in seen:
            units.append(candidate_id)
            seen.add(candidate_id)
    for candidate_id in sorted(plan["selection"].get("extra_refits", [])):
        if candidate_id not in seen:
            units.append(candidate_id)
            seen.add(candidate_id)
    return units


def overall_key(plan, selection_item):
    """The overall winner's slot key for its fold year."""
    year = selection_item["outer_year"]
    if per_candidate(plan):
        return refit_key(plan, year, selection_item["selected_candidate_id"])
    return refit_key(plan, year)


def terminal_keys(plan, selection):
    """Every required outer-terminal key: each fold's units, overall first."""
    by_year = {item["outer_year"]: item for item in selection["selections"]}
    keys = []
    for fold in plan["folds"]:
        year = fold["outer"]["year"]
        for unit in refit_units(plan, by_year[year]):
            keys.append(refit_key(plan, year, unit))
    return keys


def possible_slots(plan):
    """Every outer slot a run could claim, for absent checks and orphan scans."""
    slots = []
    for item in plan["folds"]:
        year = item["outer"]["year"]
        if not per_candidate(plan):
            slots.append((year, None, refit_key(plan, year)))
            continue
        for candidate in plan["candidates"]:
            candidate_id = candidate["candidate_id"]
            slots.append((year, candidate_id, refit_key(plan, year, candidate_id)))
    return slots


def failed_terminal(key, year, candidate_id):
    """A failed non-overall refit unit is durable and is never retried."""
    return {"schema_version": "magnitude_search_outer_failed.v1", "outer_id": key,
            "outer_year": year, "candidate_id": candidate_id, "status": "failed",
            "code": "EXPERIMENT_VARIANT_FAILED"}
