"""Run non-destructive validity checks before sequence construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EVENT_COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]
EXPECTED_EVENTS = {"view", "addtocart", "transaction"}


def result(name: str, expected: str, observed: str, passed: bool, next_action: str) -> dict[str, str]:
    return {
        "assumption": name,
        "expected": expected,
        "observed": observed,
        "status": "PASS" if passed else "NEEDS_DECISION",
        "next_action": next_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-events", type=Path, required=True)
    parser.add_argument("--clean-events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_events)
    clean = pd.read_csv(args.clean_events)
    missing = set(EVENT_COLUMNS).difference(raw.columns) | set(EVENT_COLUMNS).difference(clean.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    raw_missing = int(raw[EVENT_COLUMNS].isna().sum().sum())
    required_missing = int(raw[["timestamp", "visitorid", "event", "itemid"]].isna().sum().sum())
    event_values = set(raw["event"].dropna().unique())
    bad_events = sorted(event_values.difference(EXPECTED_EVENTS))
    transaction_rows = raw["event"].eq("transaction")
    transaction_id_present = int(raw.loc[transaction_rows, "transactionid"].notna().sum())
    transaction_total = int(transaction_rows.sum())
    non_transaction_with_id = int(raw.loc[~transaction_rows, "transactionid"].notna().sum())
    ties = int(raw.duplicated(["visitorid", "timestamp"], keep=False).sum())
    clean_exact_duplicates = int(clean.duplicated(EVENT_COLUMNS, keep=False).sum())
    expected_clean_rows = len(raw) - int(raw.duplicated(EVENT_COLUMNS).sum())

    checks = [
        result(
            "Required fields are present",
            "No blank timestamp, visitorid, event, or itemid.",
            f"{required_missing:,} blank required cells.",
            required_missing == 0,
            "Safe to use these four fields for sequence construction." if required_missing == 0 else "Stop and resolve missing required fields.",
        ),
        result(
            "Transaction-ID blanks have the expected meaning",
            "Transaction rows have an ID; non-transaction rows can be blank.",
            f"{transaction_id_present:,}/{transaction_total:,} transaction rows have an ID; {non_transaction_with_id:,} other rows have an ID.",
            transaction_id_present == transaction_total and non_transaction_with_id == 0,
            "Keep transactionid blank outside transactions; do not fill it." if transaction_id_present == transaction_total and non_transaction_with_id == 0 else "Inspect transaction-ID rule before using it.",
        ),
        result(
            "Event names match the documented dataset",
            "Only view, addtocart, transaction.",
            ", ".join(sorted(event_values)) if not bad_events else f"Unexpected: {', '.join(bad_events)}",
            not bad_events,
            "Use these three event names in the candidate event-policy comparison." if not bad_events else "Update event policy or clean unexpected labels.",
        ),
        result(
            "Timestamps can order events",
            "Every timestamp is a non-negative number.",
            f"Minimum={int(raw['timestamp'].min())}; maximum={int(raw['timestamp'].max())}; invalid={int((raw['timestamp'] < 0).sum()):,}.",
            raw["timestamp"].notna().all() and bool((raw["timestamp"] >= 0).all()),
            "Use chronological ordering; convert milliseconds to time gaps later.",
        ),
        result(
            "Exact copies were removed only in the separate cleaned file",
            "Cleaned file has zero fully identical rows and expected row count.",
            f"raw={len(raw):,}; cleaned={len(clean):,}; expected cleaned={expected_clean_rows:,}; duplicate members remaining={clean_exact_duplicates:,}.",
            len(clean) == expected_clean_rows and clean_exact_duplicates == 0,
            "Use the cleaned copy as the next preprocessing input." if len(clean) == expected_clean_rows and clean_exact_duplicates == 0 else "Re-run exact-duplicate audit before continuing.",
        ),
        result(
            "One timestamp is enough to strictly order each visitor's events",
            "No visitor has two events at the exact same timestamp.",
            f"{ties:,} rows belong to a visitor-and-timestamp tie.",
            ties == 0,
            "Choose and document a deterministic tie policy before creating next-item labels.",
        ),
        result(
            "No future information has entered training examples",
            "No sequences, temporal split, labels, or models have been created yet.",
            "Not applicable yet: data checks only.",
            True,
            "When examples are built, verify each history timestamp is earlier than its target and split boundary.",
        ),
    ]

    report = pd.DataFrame(checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(report[["assumption", "status"]].to_string(index=False))
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
