"""Tests for the HttpMmqbRun Azure Function helpers."""

import json
import os
from types import SimpleNamespace
from typing import Dict, List
import unittest
from unittest import mock

from azure_func.HttpMmqbRun import (
    PROC_DATA_CURRENT,
    PROC_PROMPT_TEXT,
    PROC_SAVE,
    _normalize_upload_id,
    main,
)
from azure_func.shared import ValidationError, generate_summary_artifacts
from azure_func.shared.validators import ensure_stored_procedure_env


def _assert_copy(testcase: unittest.TestCase, original, received) -> bool:
    testcase.assertIsNot(received, original)
    testcase.assertEqual(received, original)
    return False


class NormalizeUploadIdTests(unittest.TestCase):
    def test_accepts_plain_yyyymmdd(self) -> None:
        data = {"UPLOAD_ID": "20250921"}
        self.assertEqual(_normalize_upload_id(data), "20250921")

    def test_accepts_suffix_after_date(self) -> None:
        data = {"upload_id": "20250921_baseline"}
        self.assertEqual(_normalize_upload_id(data), "20250921_baseline")

    def test_rejects_invalid_format(self) -> None:
        data = {"UPLOAD_ID": "not-a-date"}
        with self.assertRaises(ValidationError) as ctx:
            _normalize_upload_id(data)
        self.assertIn("YYYYMMDD", str(ctx.exception))

    def test_rejects_invalid_calendar_date(self) -> None:
        data = {"UPLOAD_ID": "20250230"}
        with self.assertRaises(ValidationError) as ctx:
            _normalize_upload_id(data)
        self.assertIn("valid YYYYMMDD", str(ctx.exception))


class EnsureStoredProcedureEnvTests(unittest.TestCase):
    def test_requires_all_procedures(self) -> None:
        env = {
            "MMQB_PROC_DATA_CURRENT": "current_proc",
            "MMQB_PROC_PROMPT_TEXT": "prompt_proc",
            "MMQB_SAVE_PROC": "save_proc",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            procs = ensure_stored_procedure_env()

        self.assertEqual(
            procs,
            {
                "data_current": "current_proc",
                "prompt_text": "prompt_proc",
                "save": "save_proc",
            },
        )

    def test_raises_when_required_missing(self) -> None:
        env = {
            "MMQB_PROC_PROMPT_TEXT": "prompt_proc",
            "MMQB_SAVE_PROC": "save_proc",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                ensure_stored_procedure_env()

        self.assertIn("MMQB_PROC_DATA_CURRENT", str(ctx.exception))


class GenerateSummaryArtifactsTests(unittest.TestCase):
    def test_splits_single_dataset_into_periods(self) -> None:
        records = [
            {
                "DELIVERY_DT": "2024-01-06",
                "BILLTO_NAME": "Client A",
                "ORIG_AREA": "O1",
                "DEST_AREA": "D1",
                "LOADED_OP_MILES": 100,
                "TOTAL_REVENUE": 1000,
                "TOTAL_VARIABLE_COST": 400,
                "TOTAL_OVERHEAD_COST": 200,
            },
            {
                "DELIVERY_DT": "2024-01-13",
                "BILLTO_NAME": "Client A",
                "ORIG_AREA": "O1",
                "DEST_AREA": "D1",
                "LOADED_OP_MILES": 120,
                "TOTAL_REVENUE": 1200,
                "TOTAL_VARIABLE_COST": 500,
                "TOTAL_OVERHEAD_COST": 220,
            },
        ]

        artifacts = generate_summary_artifacts(records, operation_cd="TEST")

        meta = artifacts.summary["meta"]
        self.assertEqual(meta["SCAC"], "TEST")
        self.assertGreater(meta["weeks_A"], 0)
        self.assertGreater(meta["weeks_B"], 0)


class CurrentDatasetFetchTests(unittest.TestCase):
    def test_fetches_current_dataset_with_operation_only(self) -> None:
        request = mock.Mock()
        request.get_json.return_value = {
            "OPERATION_CD": "TEST_OP",
            "UPLOAD_ID": "20240101_upload",
        }

        procs = {
            PROC_DATA_CURRENT: "proc_current",
            PROC_PROMPT_TEXT: "proc_prompt",
            PROC_SAVE: "proc_save",
        }

        client_doc = {
            "title": "CLIENT Performance Report",
            "meta": {"report_type": "client"},
        }
        internal_doc = {"meta": {"report_type": "internal"}}

        class FakeDb:
            def __init__(self) -> None:
                self.calls = []

            def call_procedure(self, proc_name, params=None):
                self.calls.append(
                    (proc_name, None if params is None else list(params))
                )
                if proc_name == procs[PROC_DATA_CURRENT]:
                    return [{"ClientCode": "CLIENT"}]
                if proc_name == procs[PROC_PROMPT_TEXT]:
                    return [{"PROMPT_TEXT": "Prompt"}]
                raise AssertionError(f"Unexpected proc: {proc_name}")

        fake_db = FakeDb()
        artifacts = SimpleNamespace(summary={"meta": {}})

        captured_client_docs: List[Dict[str, Any]] = []

        with mock.patch(
            "azure_func.HttpMmqbRun.ensure_stored_procedure_env",
            return_value=procs,
        ), mock.patch(
            "azure_func.HttpMmqbRun.DatabaseClient",
            return_value=fake_db,
        ), mock.patch(
            "azure_func.HttpMmqbRun.generate_summary_artifacts",
            return_value=artifacts,
        ), mock.patch(
            "azure_func.HttpMmqbRun.generate_documents",
            return_value=(client_doc, internal_doc),
        ), mock.patch(
            "azure_func.HttpMmqbRun.render_client",
            side_effect=lambda doc: captured_client_docs.append(doc) or "<html>client</html>",
        ) as mock_render_client, mock.patch(
            "azure_func.HttpMmqbRun.render_internal",
            side_effect=lambda doc: _assert_copy(self, internal_doc, doc) or "<html>internal</html>",
        ) as mock_render_internal, mock.patch(
            "azure_func.HttpMmqbRun._call_procedure_no_results"
        ) as mock_save:
            response = main(request)

        self.assertEqual(fake_db.calls[0], ("proc_current", ["TEST_OP"]))
        self.assertEqual(response.status_code, 200)
        mock_render_client.assert_called_once()
        mock_render_internal.assert_called_once()
        mock_save.assert_called_once()
        saved_args = mock_save.call_args[0][2]
        self.assertEqual(saved_args[0:2], ["TEST_OP", "20240101_upload"])
        self.assertEqual(saved_args[2], json.dumps(internal_doc))
        client_payload = saved_args[3]
        self.assertNotIn("Client View", client_payload)
        self.assertIn("CLIENT Performance Report", client_payload)
        self.assertEqual(saved_args[4:], ["<html>internal</html>", "<html>client</html>"])
        self.assertEqual(len(captured_client_docs), 1)
        sanitized_doc = captured_client_docs[0]
        self.assertIsNot(sanitized_doc, client_doc)
        self.assertEqual(sanitized_doc.get("meta"), client_doc.get("meta"))
        self.assertNotIn("Client View", sanitized_doc.get("title", ""))


if __name__ == "__main__":
    unittest.main()
