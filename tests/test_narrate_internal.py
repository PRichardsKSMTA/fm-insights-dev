from typing import Any, Dict
from unittest import TestCase, mock

import narrate


class BuildDocsInternalTests(TestCase):
    def setUp(self) -> None:
        self.summary: Dict[str, Any] = {
            "meta": {
                "SCAC": "TEST",
                "A_label": "05/25/24",
                "B_label": "04/27/24",
                "weeks_A": 4,
                "weeks_B": 4,
                "network_total_A": 120,
                "network_total_B": 100,
                "weekly_total_A": 30.0,
                "weekly_total_B": 25.0,
            },
            "network": {
                "A": {
                    "loads": 120,
                    "or_pct": 0.92,
                    "rpl": 1234.5,
                    "overhead": 100.0,
                    "variable": 200.0,
                    "loh": 500.0,
                },
                "B": {
                    "loads": 100,
                    "or_pct": 0.95,
                    "rpl": 1000.0,
                    "overhead": 110.0,
                    "variable": 210.0,
                    "loh": 480.0,
                },
            },
            "tables": {"customers": []},
            "top10": [],
        }

    def test_internal_exec_summary_is_deterministic(self) -> None:
        with mock.patch("narrate._call_chat_completion", return_value=None) as mock_chat:
            client_doc, internal_doc = narrate.build_docs(self.summary, model="dummy")

        # Client generation still invokes OpenAI once for the client payload only
        self.assertTrue(mock_chat.called)
        purposes = [call.kwargs.get("purpose") for call in mock_chat.call_args_list]
        self.assertNotIn("internal_report", purposes)

        expected_exec = narrate.build_internal_exec_summary(self.summary)
        self.assertEqual(internal_doc["exec_summary"], expected_exec)
        self.assertEqual(internal_doc["network"], self.summary["network"])
        self.assertEqual(internal_doc["tables"], self.summary["tables"])
        self.assertEqual(internal_doc["top10"], self.summary["top10"])
        self.assertEqual(internal_doc["meta"]["report_type"], "internal")
        self.assertIn("tiny_exec_summary", internal_doc)
        self.assertIn("Period A", internal_doc["tiny_exec_summary"])

    def test_internal_payload_helper_returns_exec_summary(self) -> None:
        payload = narrate.build_internal_payload(self.summary, model="unused")
        expected_exec = narrate.build_internal_exec_summary(self.summary)
        self.assertEqual(payload, {"executive_summary": expected_exec})


if __name__ == "__main__":
    import unittest

    unittest.main()
