from __future__ import annotations

from io import BytesIO
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from normalized_document_parser import normalize_document_result
from ocr_service import _apply_identity_banner_fallback, _apply_synthetic_blocks, _build_identity_banner_blocks, _extract_text_lines_from_ocr_results, _prepare_input


class OcrServiceBannerFallbackTests(unittest.TestCase):
    def test_extract_text_lines_from_paddle_like_result_object(self):
        result = SimpleNamespace(
            json={
                "res": {
                    "rec_texts": [
                        "DC12401F011 Billy Summers Male 1/1/1234 1234 Spring Street Open",
                        "1 case(s)",
                    ]
                }
            }
        )

        lines = _extract_text_lines_from_ocr_results([result])
        self.assertEqual(lines, [
            "DC12401F011 Billy Summers Male 1/1/1234 1234 Spring Street Open",
            "1 case(s)",
        ])

    def test_build_identity_banner_blocks_from_flat_ocr_lines(self):
        blocks = _build_identity_banner_blocks(
            [
                "PERM ID Participant Gender DOB Address Action",
                "DC12401F011 Billy Summers Male 1/1/1234 1234 Spring Street Open",
                "1 case(s)",
            ]
        )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["type"], "table")
        self.assertEqual(blocks[0]["headers"], [
            "PERM ID",
            "Participant",
            "Gender",
            "DOB",
            "Address",
            "Action",
        ])
        self.assertEqual(blocks[0]["rows"], [[
            "DC12401F011",
            "Billy Summers",
            "Male",
            "1/1/1234",
            "1234 Spring Street",
            "Open",
        ]])
        self.assertEqual(blocks[1], {"type": "text", "text": "1 case(s)"})

    def test_build_identity_banner_blocks_from_split_banner_lines(self):
        blocks = _build_identity_banner_blocks(
            [
                "DC12401F011 Billy Summers",
                "Male 1/1/1234 1234 Spring Street Open",
                "1 case(s)",
            ]
        )

        self.assertEqual(blocks[0]["rows"], [[
            "DC12401F011",
            "Billy Summers",
            "Male",
            "1/1/1234",
            "1234 Spring Street",
            "Open",
        ]])
        self.assertEqual(blocks[1], {"type": "text", "text": "1 case(s)"})

    def test_build_identity_banner_blocks_normalizes_dom_perm_id_to_dcm(self):
        blocks = _build_identity_banner_blocks(
            [
                "DOM2401F011 Male 1/1/1234 1234 Spring Street Open",
            ]
        )

        self.assertEqual(blocks[0]["rows"][0][0], "DCM2401F011")

    def test_apply_synthetic_blocks_updates_legacy_and_normalized_output(self):
        document_result = {
            "source": "image1.png",
            "summary": {
                "engine": "PP-StructureV3",
                "page_count": 1,
                "table_count": 0,
                "field_count": 0,
                "text_characters": 0,
            },
            "raw_text": "",
            "markdown": "",
            "tables": [],
            "structured_fields": [],
            "pages": [
                {
                    "page_index": 0,
                    "markdown": "",
                    "tables": [],
                    "structured_fields": [],
                    "layout_blocks": 1,
                }
            ],
            "raw_document": {
                "image_id": "image1",
                "metadata": {
                    "filename": "image1.png",
                    "page_number": 1,
                    "ocr_mode": "Full",
                    "run_timestamp": "2026-04-05T18:00:00Z",
                },
                "blocks": [],
            },
        }

        blocks = _build_identity_banner_blocks(
            [
                "DC12401F011 Billy Summers Male 1/1/1234 1234 Spring Street Open",
                "1 case(s)",
            ]
        )
        _apply_synthetic_blocks(document_result, blocks)

        self.assertEqual(document_result["summary"]["table_count"], 1)
        self.assertIn("DC12401F011", document_result["markdown"])
        self.assertEqual(document_result["tables"][0]["rows"][0], [
            "PERM ID",
            "Participant",
            "Gender",
            "DOB",
            "Address",
            "Action",
        ])

        normalized = normalize_document_result(document_result)
        self.assertEqual(normalized["key_facts"]["PERM ID"], "DC12401F011")
        self.assertEqual(normalized["key_facts"]["Client Name"], "Billy Summers")
        self.assertEqual(normalized["key_facts"]["Gender"], "Male")
        self.assertEqual(normalized["key_facts"]["Address"], "1234 Spring Street")

    def test_prepare_input_preprocesses_short_banner_images(self):
        image_path = BACKEND_DIR.parent / "test_images" / "image1.png"
        prepared_bytes, prepared_suffix, processing = _prepare_input(
            image_path.read_bytes(),
            suffix=".png",
            profile_key="full",
        )

        prepared_image = Image.open(BytesIO(prepared_bytes))
        self.assertEqual(prepared_suffix, ".jpg")
        self.assertTrue(processing["banner_preprocessed"])
        self.assertEqual(processing["image_size"], [1433, 63])
        self.assertGreater(prepared_image.size[1], 63)
        self.assertGreater(prepared_image.size[0], 1433)

    def test_prepare_input_skips_non_tiny_banner_headings(self):
        image_path = BACKEND_DIR.parent / "test_images" / "image3.png"
        prepared_bytes, prepared_suffix, processing = _prepare_input(
            image_path.read_bytes(),
            suffix=".png",
            profile_key="full",
        )

        self.assertEqual(prepared_bytes, image_path.read_bytes())
        self.assertEqual(prepared_suffix, ".png")
        self.assertEqual(processing, {"downscaled": False, "image_size": [1167, 78]})

    def test_identity_banner_fallback_replaces_low_signal_table(self):
        source_image = Image.new("RGB", (1200, 60), "white")
        buffer = BytesIO()
        source_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        document_result = {
            "source": "image1.png",
            "summary": {
                "engine": "PP-StructureV3",
                "page_count": 1,
                "table_count": 1,
                "field_count": 0,
                "text_characters": 0,
            },
            "raw_text": "",
            "markdown": "",
            "tables": [],
            "structured_fields": [],
            "pages": [
                {
                    "page_index": 0,
                    "markdown": "",
                    "tables": [],
                    "structured_fields": [],
                    "layout_blocks": 1,
                }
            ],
            "raw_document": {
                "image_id": "image1",
                "metadata": {
                    "filename": "image1.png",
                    "page_number": 1,
                    "ocr_mode": "Full",
                    "run_timestamp": "2026-04-05T18:00:00Z",
                },
                "blocks": [
                    {
                        "type": "table",
                        "title": "Table",
                        "headers": ["cndo", "e", "201130", "w", "mn", "B"],
                        "rows": [["cre", "seaee", "800", "peee", "seireet", "GIEAEE"]],
                    }
                ],
            },
        }
        banner_blocks = _build_identity_banner_blocks(
            [
                "DC12401F011 Billy Summers Male 1/1/1234 1234 Spring Street Open",
                "1 case(s)",
            ]
        )

        with patch("ocr_service._extract_identity_banner_blocks", return_value=banner_blocks):
            _apply_identity_banner_fallback(
                document_result,
                image_bytes=image_bytes,
                suffix=".png",
                profile_key="full",
            )

        self.assertEqual(document_result["raw_document"]["blocks"], banner_blocks)
        self.assertIn("DC12401F011", document_result["markdown"])


if __name__ == "__main__":
    unittest.main()
