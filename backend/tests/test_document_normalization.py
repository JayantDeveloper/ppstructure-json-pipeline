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

    def test_markdown_heading_text_is_normalized_in_raw_and_normalized_output(self):
        raw_document = build_raw_document(
            [
                {
                    "page_index": 0,
                    "page_json": {},
                    "page_markdown": "# Program Eligibility for Selected Year",
                    "page_tables": [],
                }
            ],
            source="image3.png",
            processing={
                "ocr_mode": "full",
                "ocr_mode_label": "Full",
                "run_timestamp": "2026-04-05T17:07:25Z",
            },
            page_number=3,
        )

        self.assertEqual(raw_document["blocks"], [
            {"type": "text", "text": "Program Eligibility for Selected Year"}
        ])

        normalized = normalize_document_result(
            {
                "source": "image3.png",
                "raw_document": raw_document,
            }
        )
        self.assertEqual(normalized["ocr_text"], ["Program Eligibility for Selected Year"])

    def test_build_raw_document_preserves_perm_id_and_marital_from_eligibility_header(self):
        html = """
        <table border="1"><tbody>
          <tr><td>DC Eligibility Determination</td></tr>
          <tr>
            <td>Billy Summers</td>
            <td>1234 Spring Street</td>
            <td>Marital Married</td>
            <td>1/1/1234</td>
            <td>PERM ID: DCM2401F011</td>
          </tr>
          <tr><td>Eligibility Determination</td><td>Home</td><td>Eligibility</td></tr>
          <tr>
            <td>PDC Case No.</td><td>IC Case No.</td><td>Program Name</td><td>Program Code</td>
            <td>Start Date</td><td>End Date</td><td>Recertification/Renewal Date</td>
            <td>Recertification/Renewal Status</td><td>Assistance Unit</td>
          </tr>
          <tr>
            <td>12345678</td><td>12345678</td><td>SS</td><td>150V</td>
            <td>11/1/2021</td><td>2/29/2024</td><td>2/29/2024</td><td></td><td>Billy Summers (3x)</td>
          </tr>
        </tbody></table>
        """
        raw_document = build_raw_document(
            [
                {
                    "page_index": 0,
                    "page_json": {
                        "parsing_res_list": [
                            {
                                "block_label": "table",
                                "block_content": html,
                            }
                        ]
                    },
                    "page_markdown": "",
                    "page_tables": [],
                }
            ],
            source="image2.png",
            processing={
                "ocr_mode": "full",
                "ocr_mode_label": "Full",
                "run_timestamp": "2026-04-05T17:07:21Z",
            },
            page_number=2,
        )

        text_blocks = [block["text"] for block in raw_document["blocks"] if block.get("type") == "text"]
        self.assertIn("Marital Married", text_blocks)
        self.assertIn("PERM ID: DCM2401F011", text_blocks)

    def test_unknown_table_is_preserved_as_generic_normalized_table(self):
        document_result = {
            "source": "image4.png",
            "raw_document": {
                "image_id": "image4",
                "metadata": {
                    "filename": "image4.png",
                    "page_number": 4,
                    "ocr_mode": "full",
                    "run_timestamp": "2026-04-05T17:07:25Z",
                },
                "blocks": [
                    {
                        "type": "table",
                        "title": "- Coveroge Type Deteils -Entitiement",
                        "headers": ["Expense", "Expense All Recorded", "All Recorded"],
                        "rows": [
                            ["MAGI Spend down", "Living Expense", "Shelter Expense"],
                            ["- Court Order Expense", "Medical Expense", "Uneamed Rental Income Expenses", "Child Support Expense"],
                        ],
                    }
                ],
            },
        }

        normalized = normalize_document_result(document_result)

        self.assertEqual(len(normalized["tables"]), 1)
        self.assertEqual(normalized["tables"][0]["columns"], [
            "Expense",
            "Expense All Recorded",
            "All Recorded",
            "Extra Column 1",
        ])
        self.assertEqual(normalized["tables"][0]["rows"][1]["Extra Column 1"], "Child Support Expense")


if __name__ == "__main__":
    unittest.main()
