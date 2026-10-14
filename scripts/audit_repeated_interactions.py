"""Measure repeated visitor--item interactions without changing source data.

An exact duplicate means every raw field is identical. This audit is different:
it finds cases where one visitor interacted with one item more than once after
exact duplicates have been removed. Repetition can be real user behaviour, so
this script only reports it; it does not remove any rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EVENT_COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--detail-output", type=Path, required=True)
    parser.add_argument("--pair-limit", type=int, default=250)
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    missing = set(EVENT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # Keep this link so any row shown in the report can be found in the raw CSV.
    events = events.copy()
    events["source_row_number"] = events.index + 2
    cleaned = events.drop_duplicates(EVENT_COLUMNS, keep="first").copy()

    pair_counts = (
        cleaned.groupby(["visitorid", "itemid"], as_index=False)
        .agg(
            interaction_count=("itemid", "size"),
            first_timestamp_ms=("timestamp", "min"),
            last_timestamp_ms=("timestamp", "max"),
            distinct_event_types=("event", "nunique"),
        )
    )
    repeated = pair_counts.loc[pair_counts["interaction_count"] > 1].copy()
    repeated["time_span_hours"] = (
        (repeated["last_timestamp_ms"] - repeated["first_timestamp_ms"]) / 3_600_000
    )
    repeated = repeated.sort_values(
        ["interaction_count", "visitorid", "itemid"], ascending=[False, True, True], kind="stable"
    )

    # A small detail file makes manual checking in Excel practical.  It contains
    # every event for the 250 most-repeated visitor--item pairs, not a random sample.
    selected_pairs = repeated[["visitorid", "itemid"]].head(args.pair_limit)
    detail = cleaned.merge(selected_pairs, on=["visitorid", "itemid"], how="inner")
    detail = detail.merge(
        repeated[["visitorid", "itemid", "interaction_count"]],
        on=["visitorid", "itemid"],
        how="left",
    )
    detail = detail.sort_values(
        ["interaction_count", "visitorid", "itemid", "timestamp", "source_row_number"],
        ascending=[False, True, True, True, True],
        kind="stable",
    )
    detail = detail[
        ["interaction_count", "source_row_number", *EVENT_COLUMNS]
    ]

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    repeated.to_csv(args.summary_output, index=False)
    detail.to_csv(args.detail_output, index=False)

    print(f"Events after exact-deduplication: {len(cleaned):,}")
    print(f"Visitor-item pairs repeated at least twice: {len(repeated):,}")
    print(f"Events belonging to repeated pairs: {int(repeated['interaction_count'].sum()):,}")
    print(f"Largest repeat count for one visitor-item pair: {int(repeated['interaction_count'].max()):,}")
    print(f"Summary written to: {args.summary_output}")
    print(f"Detail written to: {args.detail_output}")


if __name__ == "__main__":
    main()
