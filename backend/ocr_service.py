import io
import os
import tempfile

import numpy as np
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
from paddleocr import TableRecognitionPipelineV2

# ---------------------------------------------------------------------------
# Engine — loaded once at import time
# ---------------------------------------------------------------------------

table_engine = TableRecognitionPipelineV2(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    layout_detection_model_name="PP-DocLayout-L",
    wired_table_structure_recognition_model_name="SLANet_plus",
    wireless_table_structure_recognition_model_name="SLANet_plus",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    enable_mkldnn=True,
    cpu_threads=8,
)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_ocr(image_bytes: bytes) -> dict:
    # Open PIL image once for Tesseract masking; write raw bytes to temp file
    # for PaddleOCR to avoid a slow re-encode step.
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tables, table_bboxes = _extract_tables(image_bytes)
    raw_text = _extract_raw_text(img, table_bboxes)
    return {
        "text": raw_text,
        "table": [kv for t in tables for kv in t["cells"]],
    }


def _extract_tables(image_bytes: bytes):
    # Write raw bytes directly — no re-encoding, so PaddleOCR gets the
    # original file just as if it were reading it off disk.
    suffix = _sniff_suffix(image_bytes)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        result = table_engine.predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    tables = []
    table_bboxes = []

    for res in result:
        inner = res.json.get("res", {})
        boxes = inner.get("layout_det_res", {}).get("boxes", [])
        table_res_list = inner.get("table_res_list", [])

        table_boxes = [b for b in boxes if b.get("label") == "table"]

        for i, table_data in enumerate(table_res_list):
            bbox = table_boxes[i]["coordinate"] if i < len(table_boxes) else []
            if bbox:
                table_bboxes.append(bbox)
                tables.append({"bbox": bbox, "cells": _parse_table_cells(table_data)})

    return tables, table_bboxes


def _parse_table_cells(table_data: dict) -> list:
    html = table_data.get("pred_html", "")
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    pairs = []
    for row in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            pairs.append({"key": cells[0], "value": cells[1]})
        elif len(cells) == 1:
            pairs.append({"key": cells[0], "value": ""})
    return pairs


def _mask_table_regions(img: Image.Image, table_bboxes: list, padding: int = 5) -> Image.Image:
    img_np = np.array(img)
    for bbox in table_bboxes:
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(img_np.shape[1], int(x2) + padding)
        y2 = min(img_np.shape[0], int(y2) + padding)
        img_np[y1:y2, x1:x2] = 255
    return Image.fromarray(img_np)


def _extract_raw_text(img: Image.Image, table_bboxes: list) -> str:
    masked_img = _mask_table_regions(img, table_bboxes)
    word_data = pytesseract.image_to_data(
        masked_img,
        config=r"--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )
    words = [
        word_data["text"][i].strip()
        for i in range(len(word_data["text"]))
        if word_data["text"][i].strip() and int(word_data["conf"][i]) > 40
    ]
    return " ".join(words)


def _sniff_suffix(image_bytes: bytes) -> str:
    """Return a file extension matching the image format from its magic bytes."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if image_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tiff"
    if image_bytes[:2] == b"BM":
        return ".bmp"
    return ".png"  # fallback
