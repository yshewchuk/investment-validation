"""Causal (leak) discipline, made mechanical.

The program's headline numbers are only worth the paper they are printed on if
every feature behind them was observable before the decision it informed. That
discipline was previously a convention maintained by care; here it is a runtime
assertion that every scoring and replay path runs through.

Two distinct claims get checked, and conflating them is how leaks survive
review:

**Feature causality** — no feature value may carry an as-of stamp later than
the decision. ``feature_as_of <= as_of``, per feature, with the offending
feature named. Equality is allowed on purpose: a feature read at the close you
also trade at is known to you at that close. What is *not* allowed is a feature
stamped even one session later.

**Decision causality** — the decision itself must be information-free about the
print. ``as_of <= last_pre_print(event)``. A feature vector can be perfectly
self-consistent and still be worthless if it was assembled the morning after
the announcement, so the event anchor is checked separately.

Both raise :class:`LeakError` rather than warning. A warning in a research
pipeline is a leak with extra steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "LeakError",
    "FeatureVector",
    "assert_causal",
    "assert_decision_causal",
    "audit_frame",
]


class LeakError(AssertionError):
    """A feature, or a decision, used information from after its cutoff."""


def _as_ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts is pd.NaT or pd.isna(ts):
        raise LeakError(f"un-timestamped value {value!r} cannot be causality-checked")
    return ts.normalize()


@dataclass(frozen=True)
class FeatureVector:
    """Feature values, each with the date its information was observable.

    ``as_of`` is the decision date — the close at which the trade would be
    placed. ``feature_as_of`` maps every entry of ``values`` to the date that
    value became knowable; a feature with no entry inherits ``as_of``, which is
    the conservative reading (it asserts nothing and passes trivially), so
    builders are expected to stamp every feature they produce and
    :meth:`assert_complete` is how a caller demands that.

    The vector is immutable. Feature builders construct one; consumers read it.
    Nothing mutates a feature vector after its causality has been asserted,
    because an assertion about a mutable object expires the moment it returns.
    """

    ticker: str
    as_of: pd.Timestamp
    values: Mapping[str, float]
    feature_as_of: Mapping[str, pd.Timestamp] = field(default_factory=dict)
    event_date: pd.Timestamp | None = None
    session: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", _as_ts(self.as_of))
        object.__setattr__(self, "values", dict(self.values))
        object.__setattr__(
            self,
            "feature_as_of",
            {k: _as_ts(v) for k, v in dict(self.feature_as_of).items()},
        )
        if self.event_date is not None:
            object.__setattr__(self, "event_date", _as_ts(self.event_date))
        unknown = sorted(set(self.feature_as_of) - set(self.values))
        if unknown:
            raise ValueError(f"feature_as_of stamps features that carry no value: {unknown}")

    # -- access ------------------------------------------------------------

    def stamp(self, name: str) -> pd.Timestamp:
        """As-of date of one feature, defaulting to the decision date."""
        return self.feature_as_of.get(name, self.as_of)

    def vector(self, names: Iterable[str]) -> np.ndarray:
        """Values for ``names`` in order, as a 1×n float array for a model.

        A name the vector does not carry is an error, not a NaN: a model
        silently scoring a missing feature as null is exactly the failure the
        registry's feature-list check exists to prevent.
        """
        names = list(names)
        missing = [n for n in names if n not in self.values]
        if missing:
            raise KeyError(f"feature vector for {self.ticker} is missing {missing}")
        return np.array([[float(self.values[n]) for n in names]], dtype=float)

    def missing(self, names: Iterable[str]) -> list[str]:
        """Requested names that are absent or non-finite."""
        out = []
        for name in names:
            if name not in self.values:
                out.append(name)
                continue
            value = self.values[name]
            if value is None or not np.isfinite(float(value)):
                out.append(name)
        return out

    def assert_complete(self, names: Iterable[str]) -> None:
        """Demand an as-of stamp on every one of ``names``."""
        unstamped = [n for n in names if n not in self.feature_as_of]
        if unstamped:
            raise LeakError(
                f"{self.ticker}: features carry no as-of stamp and therefore cannot "
                f"be audited: {sorted(unstamped)}"
            )

    def with_values(self, **overrides: float) -> "FeatureVector":
        """Copy carrying replaced values — used by tests to poison a feature."""
        return FeatureVector(
            ticker=self.ticker,
            as_of=self.as_of,
            values={**self.values, **overrides},
            feature_as_of=dict(self.feature_as_of),
            event_date=self.event_date,
            session=self.session,
            meta=dict(self.meta),
        )

    def with_stamps(self, **overrides) -> "FeatureVector":
        """Copy carrying replaced as-of stamps — the leak-poison test's tool."""
        return FeatureVector(
            ticker=self.ticker,
            as_of=self.as_of,
            values=dict(self.values),
            feature_as_of={**self.feature_as_of, **overrides},
            event_date=self.event_date,
            session=self.session,
            meta=dict(self.meta),
        )


def assert_causal(features: FeatureVector, as_of=None) -> None:
    """Raise unless every feature was observable at ``as_of``.

    ``as_of`` defaults to the vector's own decision date. Passing it explicitly
    is how a caller checks a vector against a decision date it did not build the
    vector for — the case that catches a re-used or cached feature vector being
    applied to an earlier decision.
    """
    if not isinstance(features, FeatureVector):
        raise TypeError(f"expected FeatureVector, got {type(features).__name__}")
    cutoff = _as_ts(as_of) if as_of is not None else features.as_of

    late = {
        name: stamp
        for name, stamp in features.feature_as_of.items()
        if stamp > cutoff
    }
    if late:
        detail = ", ".join(
            f"{name} @ {stamp.date()}" for name, stamp in sorted(late.items())
        )
        raise LeakError(
            f"{features.ticker}: {len(late)} feature(s) observed after the "
            f"{cutoff.date()} decision — {detail}"
        )


def assert_decision_causal(as_of, event_date, session: str, calendar=None) -> None:
    """Raise unless a decision at ``as_of`` precedes the print.

    The last information-free close is session-dependent: for a BMO print it is
    the previous trading day, for an AMC print the event date itself. Deciding
    on the event-date close for a BMO name means deciding after the
    announcement — the same one-day error that would make every backtest in the
    program look brilliant.
    """
    from engine.calendar import trading_calendar

    cal = calendar if calendar is not None else trading_calendar()
    as_of = _as_ts(as_of)
    event_date = _as_ts(event_date)
    last_ok = cal.last_pre_print(event_date, session)
    if as_of > last_ok:
        raise LeakError(
            f"decision date {as_of.date()} is after the last information-free "
            f"close {last_ok.date()} for the {session} print on {event_date.date()}"
        )


def audit_frame(
    df: pd.DataFrame,
    as_of_column: str,
    stamp_columns: Iterable[str],
    *,
    label: str = "frame",
) -> None:
    """Vectorized causality check over a whole feature frame.

    The per-row :class:`FeatureVector` check is the contract; this is the same
    assertion applied to a replay frame of hundreds of thousands of rows, where
    building one object per row would dominate the runtime.
    """
    cutoff = pd.to_datetime(df[as_of_column])
    offenders: dict[str, int] = {}
    for column in stamp_columns:
        if column not in df.columns:
            raise KeyError(f"{label}: no stamp column {column!r}")
        stamps = pd.to_datetime(df[column])
        bad = int((stamps > cutoff).sum())
        if bad:
            offenders[column] = bad
    if offenders:
        detail = ", ".join(f"{c}: {n} row(s)" for c, n in sorted(offenders.items()))
        raise LeakError(f"{label}: stamps after the decision date — {detail}")
