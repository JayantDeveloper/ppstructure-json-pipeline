from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from PIL import Image, ImageOps, UnidentifiedImageError

from normalized_document_parser import normalize_document_result
from structured_doc_utils import _dedupe_blocks, collect_document_result

_engines = {}

DEFAULT_OCR_MODE = (os.environ.get("OCR_DEMO_MODE") or "fast").strip().lower()

OCR_PROFILES = {
    "fast": {
        "display_name": "Fast",
        "max_image_side": 1800,
        "max_image_pixels": 3_500_000,
        "candidate_kwargs": lambda common_kwargs: [
            {
                **common_kwargs,
                "use_table_recognition": False,
                "layout_detection_model_name": "PP-DocLayout-S",
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "PP-OCRv5_mobile_rec",
            },
            {
                **common_kwargs,
                "use_table_recognition": False,
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "PP-OCRv5_mobile_rec",
            },
            {
                **common_kwargs,
                "use_table_recognition": False,
            },
        ],
    },
    "full": {
        "display_name": "Full",
        "max_image_side": 2600,
        "max_image_pixels": 7_000_000,
        "candidate_kwargs": lambda common_kwargs: [
            {
                **common_kwargs,
                "use_table_recognition": True,
                "text_detection_model_name": "PP-OCRv5_server_det",
                "text_recognition_model_name": "PP-OCRv5_server_rec",
            },
            {
                **common_kwargs,
                "use_table_recognition": True,
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "PP-OCRv5_mobile_rec",
            },
            {
                **common_kwargs,
                "use_table_recognition": True,
            },
        ],
    },
}


def run_ocr(
    file_bytes: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
    mode: str | None = None,
) -> dict:
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    profile_key = _normalize_mode(mode)
    profile = OCR_PROFILES[profile_key]

    suffix = _detect_suffix(file_bytes, filename, content_type)
    prepared_bytes, prepared_suffix, processing = _prepare_input(
        file_bytes,
        suffix=suffix,
        profile_key=profile_key,
    )
    source_name = filename or f"upload{suffix}"
    run_timestamp = _utc_timestamp()
    document_processing = {
        "ocr_mode": profile_key,
        "ocr_mode_label": profile["display_name"],
        "run_timestamp": run_timestamp,
        **processing,
    }

    results = _predict_results(prepared_bytes, suffix=prepared_suffix, profile_key=profile_key)

    if not results:
        raise ValueError("No document pages were produced by PP-StructureV3.")

    document_result = collect_document_result(
        results,
        source=source_name,
        processing=document_processing,
        page_number=1,
    )
    document_result["processing"] = document_processing
    _apply_targeted_fallbacks(
        document_result,
        image_bytes=prepared_bytes,
        suffix=prepared_suffix,
        profile_key=profile_key,
        source_name=source_name,
        processing=document_processing,
    )
    document_result["normalized_document"] = normalize_document_result(
        document_result,
        processing=document_processing,
    )
    return document_result


def _patch_paddle_analysis_config() -> None:
    import paddle

    config_cls = paddle.inference.Config
    if hasattr(config_cls, "set_optimization_level"):
        return

    def _set_optimization_level(self, level):
        setter = getattr(self, "set_tensorrt_optimization_level", None)
        if callable(setter):
            try:
                setter(level)
            except TypeError:
                pass

    setattr(config_cls, "set_optimization_level", _set_optimization_level)


def _get_engine(profile_key: str):
    if profile_key in _engines:
        return _engines[profile_key]

    _patch_paddle_analysis_config()
    from paddleocr import PPStructureV3

    device = os.environ.get("OCR_DEMO_DEVICE") or os.environ.get("OCRFULL_P4_DEVICE")
    common_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "use_formula_recognition": False,
        "use_chart_recognition": False,
        "format_block_content": True,
        "ocr_version": "PP-OCRv5",
        "lang": "en",
    }
    if device:
        common_kwargs["device"] = device

    candidate_kwargs = OCR_PROFILES[profile_key]["candidate_kwargs"](common_kwargs)

    last_error = None
    for kwargs in candidate_kwargs:
        try:
            _engines[profile_key] = PPStructureV3(**kwargs)
            return _engines[profile_key]
        except (TypeError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to initialize PP-StructureV3 profile '{profile_key}': {last_error}"
    )


def _predict_results(file_bytes: bytes, *, suffix: str, profile_key: str):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        return list(_get_engine(profile_key).predict(input=tmp_path))
    finally:
        os.unlink(tmp_path)


def _prepare_input(file_bytes: bytes, *, suffix: str, profile_key: str) -> tuple[bytes, str, dict]:
    if suffix == ".pdf":
        return file_bytes, suffix, {"downscaled": False}

    try:
        with Image.open(BytesIO(file_bytes)) as image:
            prepared_image = ImageOps.exif_transpose(image)
            original_size = prepared_image.size
            max_side, max_pixels = _get_image_limits(profile_key)
            resized_image, was_resized = _downscale_image(
                prepared_image,
                max_side=max_side,
                max_pixels=max_pixels,
            )

            if not was_resized:
                return file_bytes, suffix, {"downscaled": False, "image_size": list(original_size)}

            output_image, output_format, output_suffix = _normalize_output_image(resized_image)
            buffer = BytesIO()
            save_kwargs = {"format": output_format}
            if output_format == "JPEG":
                save_kwargs["quality"] = 92
                save_kwargs["optimize"] = True
            else:
                save_kwargs["optimize"] = True
            output_image.save(buffer, **save_kwargs)
            return (
                buffer.getvalue(),
                output_suffix,
                {
                    "downscaled": True,
                    "image_size": list(original_size),
                    "processed_image_size": list(output_image.size),
                },
            )
    except (UnidentifiedImageError, OSError):
        return file_bytes, suffix, {"downscaled": False}


def _apply_targeted_fallbacks(
    document_result: dict,
    *,
    image_bytes: bytes,
    suffix: str,
    profile_key: str,
    source_name: str,
    processing: dict,
) -> None:
    if suffix == ".pdf":
        return

    raw_document = document_result.get("raw_document") or {}
    blocks = raw_document.get("blocks") or []
    eligibility_table = _find_eligibility_table_block(blocks)
    if not eligibility_table:
        return

    with Image.open(BytesIO(image_bytes)) as image:
        prepared_image = ImageOps.exif_transpose(image).convert("RGB")

        better_table_block = None
        if len(eligibility_table.get("rows") or []) < 2:
            better_table_block = _extract_better_table_block(
                prepared_image,
                profile_key=profile_key,
                source_name=source_name,
                processing=processing,
            )

        supplemental_text_blocks: list[dict] = []
        if not _has_text_block(blocks, "perm id") or not _has_text_block(blocks, "marital"):
            supplemental_text_blocks = _extract_header_text_blocks(
                prepared_image,
                profile_key=profile_key,
                source_name=source_name,
                processing=processing,
            )

    if better_table_block:
        _replace_table_block(raw_document, better_table_block)

    if supplemental_text_blocks:
        _merge_text_blocks(raw_document, supplemental_text_blocks)


def _extract_better_table_block(
    image: Image.Image,
    *,
    profile_key: str,
    source_name: str,
    processing: dict,
) -> dict | None:
    candidate_boxes = [
        (0.0, 0.30, 1.0, 0.62),
        (0.0, 0.34, 1.0, 0.66),
    ]
    best_block = None
    best_score = -1

    for box in candidate_boxes:
        crop_bytes, crop_suffix = _crop_image_region(image, box)
        crop_results = _predict_results(crop_bytes, suffix=crop_suffix, profile_key=profile_key)
        crop_document = collect_document_result(
            crop_results,
            source=source_name,
            processing=processing,
            page_number=1,
        )
        table_block = _find_eligibility_table_block(crop_document.get("raw_document", {}).get("blocks") or [])
        if not table_block:
            continue
        row_count = len(table_block.get("rows") or [])
        text_count = sum(len([value for value in row if value]) for row in (table_block.get("rows") or []))
        score = row_count * 100 + text_count
        if score > best_score:
            best_score = score
            best_block = table_block

    return best_block


def _extract_header_text_blocks(
    image: Image.Image,
    *,
    profile_key: str,
    source_name: str,
    processing: dict,
) -> list[dict]:
    candidate_boxes = [
        (0.0, 0.0, 1.0, 0.32),
        (0.0, 0.0, 1.0, 0.38),
    ]
    collected: list[dict] = []
    seen: set[str] = set()

    for box in candidate_boxes:
        crop_bytes, crop_suffix = _crop_image_region(image, box)
        crop_results = _predict_results(crop_bytes, suffix=crop_suffix, profile_key=profile_key)
        crop_document = collect_document_result(
            crop_results,
            source=source_name,
            processing=processing,
            page_number=1,
        )
        for block in crop_document.get("raw_document", {}).get("blocks") or []:
            if block.get("type") != "text":
                continue
            text = str(block.get("text", "") or "").strip()
            if not text:
                continue
            signature = text.lower()
            if signature in seen:
                continue
            if not (
                "perm id" in signature
                or "marital" in signature
                or "born" in signature
                or _looks_like_header_name(signature)
                or _looks_like_header_address(text)
            ):
                continue
            seen.add(signature)
            collected.append({"type": "text", "text": text})

    return collected


def _crop_image_region(image: Image.Image, box: tuple[float, float, float, float]) -> tuple[bytes, str]:
    width, height = image.size
    left = max(0, min(width, int(width * box[0])))
    top = max(0, min(height, int(height * box[1])))
    right = max(left + 1, min(width, int(width * box[2])))
    bottom = max(top + 1, min(height, int(height * box[3])))

    cropped = image.crop((left, top, right, bottom))
    buffer = BytesIO()
    cropped.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), ".png"


def _find_eligibility_table_block(blocks: list[dict]) -> dict | None:
    for block in blocks:
        if block.get("type") != "table":
            continue
        headers = [str(value).strip() for value in block.get("headers", [])]
        if headers == [
            "PDC Case No.",
            "IC Case No.",
            "Program Name",
            "Program Code",
            "Start Date",
            "End Date",
            "Recertification/Renewal Date",
            "Recertification/Renewal Status",
            "Assistance Unit",
        ]:
            return block
    return None


def _replace_table_block(raw_document: dict, replacement: dict) -> None:
    blocks = raw_document.get("blocks") or []
    for index, block in enumerate(blocks):
        if block.get("type") != "table":
            continue
        headers = [str(value).strip() for value in block.get("headers", [])]
        replacement_headers = [str(value).strip() for value in replacement.get("headers", [])]
        if headers != replacement_headers:
            continue
        blocks[index] = replacement
        raw_document["blocks"] = _dedupe_blocks(blocks)
        return


def _merge_text_blocks(raw_document: dict, blocks: list[dict]) -> None:
    current_blocks = raw_document.get("blocks") or []
    current_blocks.extend(blocks)
    raw_document["blocks"] = _dedupe_blocks(current_blocks)


def _has_text_block(blocks: list[dict], needle: str) -> bool:
    normalized_needle = needle.strip().lower()
    for block in blocks:
        if block.get("type") != "text":
            continue
        text = str(block.get("text", "") or "").strip().lower()
        if normalized_needle in text:
            return True
    return False


def _looks_like_header_name(value: str) -> bool:
    if "participant" in value or "data" in value:
        return False
    if any(ch.isdigit() for ch in value):
        return False
    parts = value.split()
    return 1 < len(parts) <= 4 and all(part and part[0].isalpha() for part in parts)


def _looks_like_header_address(value: str) -> bool:
    return bool(
        re.search(
            r"\b(street|st\.?|avenue|ave\.?|road|rd\.?|drive|dr\.?|lane|ln\.?|court|ct\.?|boulevard|blvd\.?|way)\b",
            value,
            re.IGNORECASE,
        )
    )


def _downscale_image(image: Image.Image, *, max_side: int, max_pixels: int) -> tuple[Image.Image, bool]:
    width, height = image.size
    scale = 1.0

    longest_side = max(width, height)
    if max_side > 0 and longest_side > max_side:
        scale = min(scale, max_side / longest_side)

    total_pixels = width * height
    if max_pixels > 0 and total_pixels > max_pixels:
        scale = min(scale, math.sqrt(max_pixels / total_pixels))

    if scale >= 0.999:
        return image.copy(), False

    resized = image.copy()
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
    return resized, True


def _normalize_output_image(image: Image.Image) -> tuple[Image.Image, str, str]:
    if image.mode in {"RGBA", "LA"}:
        base = Image.new("RGB", image.size, "white")
        alpha = image.getchannel("A")
        base.paste(image.convert("RGBA"), mask=alpha)
        return base, "PNG", ".png"

    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
        return _normalize_output_image(image)

    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")

    if image.mode == "L":
        return image, "PNG", ".png"

    return image, "JPEG", ".jpg"


def _get_image_limits(profile_key: str) -> tuple[int, int]:
    override_side = _env_int("OCR_DEMO_MAX_IMAGE_SIDE")
    override_pixels = _env_int("OCR_DEMO_MAX_IMAGE_PIXELS")
    profile = OCR_PROFILES[profile_key]
    return (
        override_side or profile["max_image_side"],
        override_pixels or profile["max_image_pixels"],
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_mode(value: str | None) -> str:
    normalized = (value or DEFAULT_OCR_MODE).strip().lower()
    if normalized in {"light", "safe"}:
        return "fast"
    if normalized in {"standard", "accurate"}:
        return "full"
    return normalized if normalized in OCR_PROFILES else "fast"


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _detect_suffix(
    file_bytes: bytes,
    filename: str | None,
    content_type: str | None,
) -> str:
    if file_bytes[:4] == b"%PDF":
        return ".pdf"
    if file_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if file_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if file_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tiff"
    if file_bytes[:2] == b"BM":
        return ".bmp"

    guessed_suffix = Path(filename or "").suffix.lower()
    if guessed_suffix in {".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff", ".bmp"}:
        return guessed_suffix

    normalized_content_type = (content_type or "").lower()
    if normalized_content_type == "application/pdf":
        return ".pdf"
    if normalized_content_type == "image/png":
        return ".png"
    if normalized_content_type in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if normalized_content_type == "image/tiff":
        return ".tiff"
    if normalized_content_type == "image/bmp":
        return ".bmp"

    raise ValueError("Unsupported file type. Upload a PDF or common image format.")
