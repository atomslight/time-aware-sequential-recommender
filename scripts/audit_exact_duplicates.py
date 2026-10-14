"""Create a reviewable audit of exact duplicate Retailrocket event rows.

The raw source CSV is read only. Every exported audit record includes the
Excel row number from `events.csv`, so reviewers can trace it back to source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EVENT_COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument(
        "--clean-output",
        type=Path,
        help="Optional new CSV containing one copy of each exact event. The source file is never changed.",
    )
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    missing = set(EVENT_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    events = events.copy()
    events["source_row_number"] = events.index + 2  # CSV header occupies Excel row 1.
    is_duplicate_member = events.duplicated(EVENT_COLUMNS, keep=False)
    audit = events.loc[is_duplicate_member, ["source_row_number", *EVENT_COLUMNS]].copy()
    audit = audit.sort_values(EVENT_COLUMNS + ["source_row_number"], kind="stable")
    group = audit.groupby(EVENT_COLUMNS, dropna=False, sort=False)
    audit["duplicate_group_id"] = group.ngroup() + 1
    audit["duplicate_rank_in_group"] = group.cumcount() + 1
    audit["duplicate_group_size"] = group["source_row_number"].transform("size")
    audit["deduplication_action"] = audit["duplicate_rank_in_group"].map(
        lambda rank: "keep_first_source_row" if rank == 1 else "remove_exact_duplicate"
    )
    audit = audit[
        [
            "duplicate_group_id",
            "duplicate_rank_in_group",
            "duplicate_group_size",
            "deduplication_action",
            "source_row_number",
            *EVENT_COLUMNS,
        ]
    ]

    summary = (
        audit[["duplicate_group_id", "duplicate_group_size"]]
        .drop_duplicates()
        .groupby("duplicate_group_size", as_index=False)
        .size()
        .rename(columns={"size": "duplicate_groups"})
    )
    summary["rows_in_groups"] = summary["duplicate_group_size"] * summary["duplicate_groups"]

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    if args.clean_output:
        args.clean_output.parent.mkdir(parents=True, exist_ok=True)
        events.drop(columns="source_row_number").drop_duplicates(EVENT_COLUMNS, keep="first").to_csv(
            args.clean_output, index=False
        )

    rows_to_remove_if_keep_one = len(events) - len(events.drop_duplicates(EVENT_COLUMNS))
    print(f"Raw rows: {len(events):,}")
    print(f"Rows in duplicate groups: {len(audit):,}")
    print(f"Duplicate groups: {audit['duplicate_group_id'].nunique():,}")
    print(f"Rows removable by keeping one exact copy per group: {rows_to_remove_if_keep_one:,}")
    print(f"Audit written to: {args.audit_output}")
    print(f"Summary written to: {args.summary_output}")
    if args.clean_output:
        print(f"Cleaned copy written to: {args.clean_output}")


if __name__ == "__main__":
    main()
