"""Tests for the shared HTML renderer."""

from __future__ import annotations

import unittest

from azure_func.shared.html_renderer import render_client, render_html, render_internal


class HtmlRendererTests(unittest.TestCase):
    def test_render_client_produces_html(self) -> None:
        doc = {
            "meta": {"report_type": "client", "A_label": "A (2 wk)", "B_label": "B (2 wk)"},
            "title": "Client Overview",
            "highlights": ["Growth in region"],
            "stories": [
                {"headline": "▲ Revenue increased", "driver_detail": "Demand up"},
                {"arrow": "→", "headline": "Loads stable", "driver_detail": "Seasonal"},
            ],
            "final_word": "Keep monitoring lane X.",
        }

        html = render_client(doc)

        self.assertTrue(html.startswith("<!DOCTYPE html>"))
        self.assertIn("Client Overview", html)
        # Ensure source document not mutated with derived keys
        self.assertNotIn("loads_A_display", str(doc))

    def test_render_internal_produces_html_without_mutation(self) -> None:
        doc = {
            "meta": {"report_type": "internal", "A_label": "A (3 wk)", "B_label": "B (3 wk)"},
            "title": "Internal Overview",
            "tables": {
                "customers": [
                    {
                        "BILLTO_NAME": "Customer 1",
                        "loads_A": 10,
                        "loads_B": 9,
                        "raw_loads_A": 9,
                        "raw_loads_B": 8,
                        "Core_OR_A": 0.91,
                        "Core_OR_B": 0.87,
                    }
                ]
            },
        }

        html = render_internal(doc)

        self.assertTrue(html.startswith("<!DOCTYPE html>"))
        self.assertIn("Internal Overview", html)
        self.assertIn("<table", html)
        # Deep copy behaviour ensures original is untouched
        self.assertNotIn("loads_A_display", doc["tables"]["customers"][0])

    def test_dispatcher_uses_report_type(self) -> None:
        doc = {
            "meta": {"report_type": "client"},
            "title": "Dispatcher",
            "highlights": ["A"],
        }

        html = render_html(doc)

        self.assertIn("Dispatcher", html)


if __name__ == "__main__":  # pragma: no cover - module test runner
    unittest.main()

