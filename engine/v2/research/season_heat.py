from __future__ import annotations

import math
import numbers
from collections import namedtuple
from collections.abc import Mapping, Sequence
from datetime import date

from engine.v2.data.errors import fail
from engine.v2.research.progress import Progress, decision, stage

_WINDOWS = (5, 20, 60, 252)
_LABEL_KEYS = frozenset({"event_id", "ticker", "event_date", "available_as_of_date", "abs_move"})
_TARGET_KEYS = frozenset({"event_id", "ticker", "event_date", "decision_session"})
_SESSIONS = "trading_sessions must be a nonempty strictly ascending sequence of canonical ISO dates"
_SUM = "season heat sum is not finite"
_MEAN = "season heat mean is not finite"
_RATIO = "season heat ratio is not finite"
_Label = namedtuple("_Label", "event_id ticker event_index available move")
_Target = namedtuple("_Target", "event_id ticker decision_index decision_date decision_session")

def _contract(message):
    return fail("CONTRACT_MISMATCH", message)

def _variant(message):
    return fail("EXPERIMENT_VARIANT_FAILED", message)

def _finite(value, message):
    try:
        if not math.isfinite(value):
            raise _variant(message)
    except (TypeError, OverflowError, ValueError):
        raise _variant(message) from None
    return value

def _divide(total, count, message):
    try:
        value = total / count
    except (OverflowError, ZeroDivisionError, TypeError, ValueError):
        raise _variant(message) from None
    return _finite(value, message)

def _parse_date(value, *, availability):
    error = _variant if availability else _contract
    message = "value is not a canonical ISO date"
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise error(message)
    except (TypeError, ValueError):
        raise error(message) from None
    return parsed

def _rows(value, name):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _contract(f"{name} must be a sequence of mappings")
    return value

def _require_keys(row, expected, name):
    if not isinstance(row, Mapping):
        raise _contract(f"{name} row is not a mapping")
    if set(row.keys()) != expected:
        raise _contract(f"{name} row keys do not match the contract")
    return row

def _row_ids(row, expected, name):
    _require_keys(row, expected, name)
    event_id, ticker = row["event_id"], row["ticker"]
    if (not isinstance(event_id, str) or not event_id.strip()
            or not isinstance(ticker, str) or not ticker.strip() or ticker != ticker.strip()):
        raise _contract(f"{name} event_id must be nonblank and ticker a string")
    return event_id, ticker

def _parse_move(value):
    message = "abs_move must be a finite nonnegative number"
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or value < 0:
        raise _variant(message)
    return value if isinstance(value, numbers.Integral) else _finite(value, message)

def _session_positions(trading_sessions):
    if isinstance(trading_sessions, (str, bytes)) or not isinstance(trading_sessions, Sequence) or not trading_sessions:
        raise _contract(_SESSIONS)
    positions, previous = {}, None
    for position, raw in enumerate(trading_sessions):
        parsed = _parse_date(raw, availability=False)
        if previous is not None and parsed <= previous:
            raise _contract(_SESSIONS)
        positions[raw], previous = (position, parsed), parsed
    return positions

def _parse_labels(labels, positions):
    parsed, seen = [], set()
    for row in _rows(labels, "labels"):
        event_id, ticker = _row_ids(row, _LABEL_KEYS, "label")
        if event_id in seen:
            raise _contract("duplicate label event_id")
        seen.add(event_id)
        if not isinstance(event_date := row["event_date"], str) or event_date not in positions:
            raise _contract("label event_date is not a trading session")
        available = _parse_date(row["available_as_of_date"], availability=True)
        if available <= positions[event_date][1]:
            raise _variant("label available_as_of_date must be strictly after event_date")
        parsed.append(_Label(event_id, ticker, positions[event_date][0], available, _parse_move(row["abs_move"])))
    return sorted(parsed, key=lambda label: (label.available, label.event_index, label.event_id))

def _parse_targets(targets, positions):
    parsed, seen = [], set()
    for row in _rows(targets, "targets"):
        event_id, ticker = _row_ids(row, _TARGET_KEYS, "target")
        event_date, decision_session = row["event_date"], row["decision_session"]
        if not isinstance(event_date, str) or event_date not in positions:
            raise _contract("target event_date is not a trading session")
        if not isinstance(decision_session, str) or decision_session not in positions:
            raise _contract("target decision_session is not a trading session")
        key = (event_id, decision_session)
        if key in seen:
            raise _contract("duplicate target key")
        seen.add(key)
        decision_index, decision_date = positions[decision_session]
        parsed.append(_Target(event_id, ticker, decision_index, decision_date, decision_session))
    return sorted(parsed, key=lambda target: (target.decision_index, target.event_id))

class _Windows:
    def __init__(self):
        self.counts, self.sums = ({w: 0 for w in _WINDOWS}, {w: 0 for w in _WINDOWS})
        self.by_ticker, self.buckets = ({w: {} for w in _WINDOWS}, {w: {} for w in _WINDOWS})
        self.lows = {w: 0 for w in _WINDOWS}
    def start(self, decision_index):
        for window in _WINDOWS:
            lower = max(0, decision_index - window + 1)
            # Running window state: drop sessions that fell below `lower`.
            for index in range(self.lows[window], lower):
                for ticker, move in self.buckets[window].pop(index, ()):
                    self.counts[window], self.sums[window] = self.counts[window] - 1, self.sums[window] - move
                    cell = self.by_ticker[window][ticker]
                    cell[0], cell[1] = cell[0] - 1, cell[1] - move
                    if not cell[0]:
                        del self.by_ticker[window][ticker]
            self.lows[window] = max(self.lows[window], lower)
    def add(self, label, decision_index):
        for window in _WINDOWS:
            if label.event_index < decision_index - window + 1:
                continue
            self.counts[window], self.sums[window] = self.counts[window] + 1, self.sums[window] + label.move
            cell = self.by_ticker[window].get(label.ticker)
            if cell is None:
                self.by_ticker[window][label.ticker] = cell = [0, 0]
            cell[0], cell[1] = cell[0] + 1, cell[1] + label.move
            self.buckets[window].setdefault(label.event_index, []).append((label.ticker, label.move))
    def totals(self, target, label_by_id):
        values = {}
        for window in _WINDOWS:
            cell = self.by_ticker[window].get(target.ticker)
            ticker_count, ticker_sum = cell if cell else (0, 0)
            count = self.counts[window] - ticker_count
            total = self.sums[window] - ticker_sum
            label = label_by_id.get(target.event_id)
            if label is not None and label.ticker != target.ticker and label.available <= target.decision_date and label.event_index <= target.decision_index and label.event_index >= target.decision_index - window + 1:
                count, total = count - 1, total - label.move
            values[window] = (count, total)
        return values

def _payload(target, windows, label_by_id):
    totals = windows.totals(target, label_by_id)
    (count_5, sum_5), (count_20, sum_20), (count_60, sum_60), (count_252, sum_252) = (totals[w] for w in _WINDOWS)
    for total in (sum_5, sum_20, sum_60, sum_252):
        _finite(total, _SUM)
    mean_5 = _divide(sum_5, count_5, _MEAN) if count_5 >= 20 else None
    mean_20 = _divide(sum_20, count_20, _MEAN) if count_20 >= 20 else None
    mean_60 = _divide(sum_60, count_60, _MEAN) if count_60 >= 20 else None
    ratio = None
    if count_20 >= 20 and count_252 >= 20 and (mean_252 := _divide(sum_252, count_252, _MEAN)) != 0:
        ratio = _divide(mean_20, mean_252, _RATIO)
    return {"season_heat_mean_abs_5": mean_5, "season_heat_mean_abs_20": mean_20, "season_heat_mean_abs_60": mean_60, "season_heat_count_20": int(count_20), "season_heat_ratio_20_252": ratio}

def compute_season_heat(labels, targets, trading_sessions) -> dict[tuple[str, str], dict]:
    with stage("season heat calculation"):
        positions = _session_positions(trading_sessions)
        parsed_labels = _parse_labels(labels, positions)
        parsed_targets = _parse_targets(targets, positions)
        label_by_id = {label.event_id: label for label in parsed_labels}
        decision("season heat", f"{len(parsed_labels)} labels, {len(parsed_targets)} targets, windows {list(_WINDOWS)}")
        progress = Progress("season heat targets", len(targets))
        windows, pending, next_pending, label_position = _Windows(), {}, 0, 0
        results = {}
        for target in parsed_targets:
            decision_index, decision_date = target.decision_index, target.decision_date
            windows.start(decision_index)
            while label_position < len(parsed_labels):
                label = parsed_labels[label_position]
                if label.available > decision_date:
                    break
                label_position += 1
                if label.event_index <= decision_index:
                    windows.add(label, decision_index)
                else:
                    pending.setdefault(label.event_index, []).append(label)
            while next_pending <= decision_index:
                for label in pending.pop(next_pending, []):
                    windows.add(label, decision_index)
                next_pending += 1
            results[(target.event_id, target.decision_session)] = _payload(target, windows, label_by_id)
            progress.tick()
        return results
