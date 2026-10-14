"""Audit events that share a visitor and an exact timestamp.

The raw file's row order is preserved as `source_row_number` only for review.
It is not treated as real chronological evidence.
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
    parser.add_argument("--group-limit", type=int, default=250)
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    missing = set(EVENT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    events = events.copy()
    events["source_row_number"] = events.index + 2
    events = events.drop_duplicates(EVENT_COLUMNS, keep="first")

    tied = events.loc[events.duplicated(["visitorid", "timestamp"], keep=False)].copy()
    tied = tied.sort_values(["visitorid", "timestamp", "source_row_number"], kind="stable")
    tied["tie_group_id"] = tied.groupby(["visitorid", "timestamp"], sort=False).ngroup() + 1
    tied["events_in_tie_group"] = tied.groupby("tie_group_id")["tie_group_id"].transform("size")
    tied["source_order_in_tie_group"] = tied.groupby("tie_group_id").cumcount() + 1

    summary = (
        tied[["tie_group_id", "visitorid", "timestamp", "events_in_tie_group"]]
        .drop_duplicates()
        .sort_values(["events_in_tie_group", "visitorid", "timestamp"], ascending=[False, True, True], kind="stable")
    )
    selected = summary.head(args.group_limit)[["tie_group_id"]]
    detail = tied.merge(selected, on="tie_group_id", how="inner")
    detail = detail[
        [
            "tie_group_id",
            "events_in_tie_group",
            "source_order_in_tie_group",
            "source_row_number",
            *EVENT_COLUMNS,
        ]
    ]

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    detail.to_csv(args.detail_output, index=False)

    print(f"Rows in timestamp ties: {len(tied):,}")
    print(f"Visitor-and-timestamp tie groups: {tied['tie_group_id'].nunique():,}")
    print(f"Largest tie group: {int(tied['events_in_tie_group'].max()):,} events")
    print(f"Summary written to: {args.summary_output}")
    print(f"Detail written to: {args.detail_output}")


if __name__ == "__main__":
    main()
