"""Create a separate modeling copy with simultaneous visitor events excluded.

Policy: after exact-deduplication, exclude every event in a visitor--timestamp
tie. The source and prior cleaned files remain unchanged. This prevents the
file's arbitrary row order from becoming an invented next-item order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    required = {"timestamp", "visitorid", "event", "itemid", "transactionid"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    is_tied = events.duplicated(["visitorid", "timestamp"], keep=False)
    output = events.loc[~is_tied].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"Input rows: {len(events):,}")
    print(f"Excluded tied-timestamp rows: {int(is_tied.sum()):,}")
    print(f"Output rows: {len(output):,}")
    print(f"Ties remaining: {int(output.duplicated(['visitorid', 'timestamp'], keep=False).sum()):,}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
