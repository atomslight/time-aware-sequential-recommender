from __future__ import annotations

import unittest

import pandas as pd

from temporalrec.data.retailrocket import inspect_events, select_eligible_events


class RetailrocketInspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = pd.DataFrame(
            {
                "timestamp": [3_000, 1_000, 2_000, 1_000, 4_000, 5_000],
                "visitorid": [1, 1, 1, 2, 2, 2],
                "event": ["view", "view", "transaction", "view", "addtocart", "view"],
                "itemid": [30, 10, 20, 40, 50, 60],
                "transactionid": [None, None, 9, None, None, None],
            }
        )

    def test_sequence_selection_is_chronological_and_thresholded(self) -> None:
        selected = select_eligible_events(self.events, minimum_events_per_user=3)
        self.assertEqual(selected[selected.visitorid == 1].itemid.tolist(), [10, 20, 30])
        self.assertEqual(selected[selected.visitorid == 1].sequence_position.tolist(), [0, 1, 2])
        self.assertEqual(selected.visitorid.nunique(), 2)

    def test_inspection_reports_time_and_coverage(self) -> None:
        report = inspect_events(self.events, minimum_events_per_user=3)
        self.assertEqual(report["raw"]["timestamp_unit"], "unix_ms")
        self.assertEqual(report["eligible_policy"]["retained_users"], 2)
        self.assertEqual(report["sequence_lengths"]["users_at_least"]["3"], 2)
        self.assertEqual(report["time_gaps_hours"]["negative_gap_count"], 0)


if __name__ == "__main__":
    unittest.main()

