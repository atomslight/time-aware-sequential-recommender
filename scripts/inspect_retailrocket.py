"""Create a read-only descriptive report for the Retailrocket event log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporalrec.data.retailrocket import inspect_events, load_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--minimum-events-per-user", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = inspect_events(load_events(args.events), minimum_events_per_user=args.minimum_events_per_user)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

