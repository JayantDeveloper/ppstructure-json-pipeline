from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from normalized_document_parser import normalize_document_result
from structured_doc_utils import build_raw_document, collect_document_result


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())


class DocumentNormalizationTests(unittest.TestCase):
    def test_build_raw_document_matches_image2_golden_fixture(self):
        fixture = _load_fixture("image2_page_input.json")
        expected = _load_fixture("image2_expected_raw_document.json")

        raw_document = build_raw_document(
            fixture["page_artifacts"],
            source="image2.png",
            processing=fixture["processing"],
            page_number=2,
        )

        self.assertEqual(raw_document, expected)

    def test_normalized_document_matches_image2_golden_fixture(self):
        fixture = _load_fixture("image2_page_input.json")
        raw_document = build_raw_document(
            fixture["page_artifacts"],
            source="image2.png",
            processing=fixture["processing"],
            page_number=2,
        )
        expected = _load_fixture("image2_expected_normalized_document.json")

        document_result = {
            "source": "image2.png",
            "summary": {"page_count": 1},
            "raw_document": raw_document,
        }
        normalized = normalize_document_result(
            document_result,
            processing=fixture["processing"],
        )

        self.assertEqual(normalized, expected)

    def test_collect_document_result_keeps_legacy_tables_and_pages(self):
        fixture = _load_fixture("image2_page_input.json")
        page_json = fixture["page_artifacts"][0]["page_json"]
        fake_result = SimpleNamespace(json={"res": page_json}, markdown="")

        document_result = collect_document_result(
            [fake_result],
            source="image2.png",
            processing=fixture["processing"],
            page_number=2,
        )

        self.assertIn("tables", document_result)
        self.assertIn("pages", document_result)
        self.assertIn("raw_document", document_result)
        self.assertEqual(document_result["pages"][0]["page_index"], 0)
        self.assertEqual(document_result["tables"][0]["rows"][0], [
            "PDC Case No.",
            "IC Case No.",
            "Program Name",
            "Program Code",
            "Start Date",
            "End Date",
            "Recertification/Renewal Date",
            "Recertification/Renewal Status",
            "Assistance Unit",
        ])

    def test_participant_data_does_not_become_client_name(self):
        document_result = {
            "source": "image4.png",
            "raw_document": {
                "image_id": "image4",
                "metadata": {
                    "filename": "image4.png",
                    "page_number": 4,
                    "ocr_mode": "full",
                    "run_timestamp": "2026-04-02T14:38:11Z",
                },
                "blocks": [
                    {"type": "text", "text": "Participant Data"},
                    {"type": "text", "text": "External System"},
                    {
                        "type": "table",
                        "title": "Table",
                        "headers": ["Expense", "Expense All Recorded", "All Recorded"],
                        "rows": [["MAGI Spend down", "Living Expense", "Shelter Expense"]],
                    },
                ],
            },
        }

        normalized = normalize_document_result(document_result)

        self.assertNotIn("Client Name", normalized["key_facts"])
        self.assertEqual(normalized["document_context"], {})


if __name__ == "__main__":
    unittest.main()
