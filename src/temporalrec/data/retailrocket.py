"""Read-only validation utilities for the Retailrocket event log."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REQUIRED_COLUMNS = ("timestamp", "visitorid", "event", "itemid", "transactionid")
DEFAULT_ELIGIBLE_EVENTS = ("view", "addtocart", "transaction")


def load_events(path: str | Path) -> pd.DataFrame:
    """Load and schema-check the raw event log without changing it."""
    events = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"Missing Retailrocket columns: {sorted(missing)}")
    if events.empty:
        raise ValueError("Retailrocket events file contains no rows.")
    if events["timestamp"].isna().any() or events["visitorid"].isna().any() or events["itemid"].isna().any():
        raise ValueError("timestamp, visitorid, and itemid must be non-null.")
    return events


def select_eligible_events(
    events: pd.DataFrame,
    eligible_events: Iterable[str] = DEFAULT_ELIGIBLE_EVENTS,
    minimum_events_per_user: int = 5,
) -> pd.DataFrame:
    """Return stable chronological sequences for users meeting the threshold.

    The threshold counts target events too: a user with five events yields four
    next-item training examples before later temporal splitting. Equal timestamps
    are retained, because resolving them requires an explicit policy.
    """
    if minimum_events_per_user < 2:
        raise ValueError("minimum_events_per_user must be at least 2.")
    allowed = set(eligible_events)
    unknown = allowed.difference(events["event"].unique())
    if unknown:
        raise ValueError(f"Eligible events absent from data: {sorted(unknown)}")

    selected = events.loc[events["event"].isin(allowed)].copy()
    user_sizes = selected.groupby("visitorid", sort=False).size()
    retained_users = user_sizes.index[user_sizes >= minimum_events_per_user]
    selected = selected.loc[selected["visitorid"].isin(retained_users)].copy()
    selected = selected.sort_values(["visitorid", "timestamp"], kind="stable")
    selected["sequence_position"] = selected.groupby("visitorid", sort=False).cumcount()
    return selected


def inspect_events(
    events: pd.DataFrame,
    eligible_events: Iterable[str] = DEFAULT_ELIGIBLE_EVENTS,
    minimum_events_per_user: int = 5,
) -> dict[str, Any]:
    """Compute descriptive statistics required before sequence preprocessing."""
    selected = events.loc[events["event"].isin(set(eligible_events))].copy()
    ordered = selected.sort_values(["visitorid", "timestamp"], kind="stable")
    sizes = ordered.groupby("visitorid", sort=False).size()
    gaps_hours = ordered.groupby("visitorid", sort=False)["timestamp"].diff().dropna() / 3_600_000
    retained = select_eligible_events(events, eligible_events, minimum_events_per_user)

    def q(series: pd.Series, value: float) -> float:
        return round(float(series.quantile(value)), 4)

    return {
        "schema": {"columns": list(events.columns), "missing_values": events.isna().sum().to_dict()},
        "raw": {
            "events": int(len(events)),
            "users": int(events["visitorid"].nunique()),
            "items": int(events["itemid"].nunique()),
            "event_counts": {str(k): int(v) for k, v in events["event"].value_counts().items()},
            "timestamp_unit": "unix_ms",
            "start_utc": pd.to_datetime(events["timestamp"].min(), unit="ms", utc=True).isoformat(),
            "end_utc": pd.to_datetime(events["timestamp"].max(), unit="ms", utc=True).isoformat(),
            "duration_days": round((events["timestamp"].max() - events["timestamp"].min()) / 86_400_000, 3),
        },
        "eligible_policy": {
            "events": sorted(set(eligible_events)),
            "minimum_events_per_user": minimum_events_per_user,
            "eligible_events": int(len(selected)),
            "eligible_users": int(selected["visitorid"].nunique()),
            "retained_events": int(len(retained)),
            "retained_users": int(retained["visitorid"].nunique()),
            "retained_items": int(retained["itemid"].nunique()),
            "event_coverage": round(len(retained) / len(selected), 6) if len(selected) else 0.0,
            "user_coverage": round(retained["visitorid"].nunique() / selected["visitorid"].nunique(), 6) if len(selected) else 0.0,
        },
        "sequence_lengths": {
            "median": q(sizes, 0.5),
            "p75": q(sizes, 0.75),
            "p90": q(sizes, 0.90),
            "p95": q(sizes, 0.95),
            "p99": q(sizes, 0.99),
            "users_at_least": {str(n): int((sizes >= n).sum()) for n in (2, 3, 5, 10, 20)},
        },
        "time_gaps_hours": {
            "count": int(len(gaps_hours)),
            "median": q(gaps_hours, 0.5) if len(gaps_hours) else None,
            "p75": q(gaps_hours, 0.75) if len(gaps_hours) else None,
            "p90": q(gaps_hours, 0.9) if len(gaps_hours) else None,
            "p99": q(gaps_hours, 0.99) if len(gaps_hours) else None,
            "zero_gap_share": round(float((gaps_hours == 0).mean()), 6) if len(gaps_hours) else None,
            "negative_gap_count": int((gaps_hours < 0).sum()),
        },
        "ordering_diagnostics": {
            "duplicate_raw_rows": int(events.duplicated().sum()),
            "same_user_same_timestamp_rows": int(events.duplicated(["visitorid", "timestamp"], keep=False).sum()),
        },
    }

