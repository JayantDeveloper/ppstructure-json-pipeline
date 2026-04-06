from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from normalized_document_parser import IDENTITY_COLUMNS, normalize_document_result
from structured_doc_utils import _dedupe_blocks, collect_document_result

_engines = {}
_text_ocr_engines = {}

ADDRESS_LINE_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9 .,'/-]+?(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Lane|Ln\.?|Court|Ct\.?|Boulevard|Blvd\.?|Way)\b",
    re.IGNORECASE,
)
CASE_COUNT_RE = re.compile(r"\b\d+\s+case\(s\)", re.IGNORECASE)
GENDER_RE = re.compile(r"\b(Male|Female)\b", re.IGNORECASE)
ACTION_RE = re.compile(r"\b(Open|Closed|Pending|Approved|Denied)\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
PERM_ID_RE = re.compile(r"(?:perm\s+id[:\s]*)?([A-Z]{2,}\d[A-Z0-9-]{5,})", re.IGNORECASE)

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
    _apply_identity_banner_fallback(
        document_result,
        image_bytes=prepared_bytes,
        suffix=prepared_suffix,
        profile_key=profile_key,
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


def _get_text_ocr_engine(profile_key: str):
    if profile_key in _text_ocr_engines:
        return _text_ocr_engines[profile_key]

    _patch_paddle_analysis_config()
    from paddleocr import PaddleOCR

    device = os.environ.get("OCR_DEMO_DEVICE") or os.environ.get("OCRFULL_P4_DEVICE")
    common_kwargs = {
        "ocr_version": "PP-OCRv5",
        "lang": "en",
    }
    if device:
        common_kwargs["device"] = device

    candidate_kwargs = [
        {
            **common_kwargs,
            "use_doc_orientation_classify": False,
            "use_textline_orientation": False,
            "text_detection_model_name": "PP-OCRv5_server_det",
            "text_recognition_model_name": "PP-OCRv5_server_rec",
        },
        {
            **common_kwargs,
            "use_doc_orientation_classify": False,
            "use_textline_orientation": False,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "PP-OCRv5_mobile_rec",
        },
        common_kwargs,
    ]
    if profile_key != "full":
        candidate_kwargs = [candidate_kwargs[1], candidate_kwargs[2]]

    last_error = None
    for kwargs in candidate_kwargs:
        try:
            _text_ocr_engines[profile_key] = PaddleOCR(**kwargs)
            return _text_ocr_engines[profile_key]
        except (TypeError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to initialize PaddleOCR profile '{profile_key}': {last_error}"
    )


def _predict_results(file_bytes: bytes, *, suffix: str, profile_key: str):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        return list(_get_engine(profile_key).predict(input=tmp_path))
    finally:
        os.unlink(tmp_path)


def _predict_text_results(file_bytes: bytes, *, suffix: str, profile_key: str):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        engine = _get_text_ocr_engine(profile_key)
        outputs = []
        if hasattr(engine, "ocr"):
            try:
                ocr_output = engine.ocr(tmp_path, cls=False) or []
                if ocr_output:
                    outputs.append(ocr_output)
            except TypeError:
                pass
        if hasattr(engine, "predict"):
            try:
                predict_output = list(engine.predict(input=tmp_path))
            except TypeError:
                try:
                    predict_output = list(engine.predict(tmp_path))
                except TypeError:
                    predict_output = []
            if predict_output:
                outputs.append(predict_output)
        return outputs
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
            working_image, was_banner_preprocessed = _prepare_banner_input(
                prepared_image,
                max_side=max_side,
            )
            resized_image, was_resized = _downscale_image(
                working_image,
                max_side=max_side,
                max_pixels=max_pixels,
            )

            if not was_banner_preprocessed and not was_resized:
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
                    "downscaled": was_resized,
                    "banner_preprocessed": was_banner_preprocessed,
                    "image_size": list(original_size),
                    "processed_image_size": list(output_image.size),
                },
            )
    except (UnidentifiedImageError, OSError):
        return file_bytes, suffix, {"downscaled": False}


def _prepare_banner_input(image: Image.Image, *, max_side: int) -> tuple[Image.Image, bool]:
    if not _needs_banner_preprocessing(image):
        return image, False

    width, height = image.size
    if width <= 0 or height <= 0:
        return image, False

    scale = min(max_side / width, max(1.0, 96 / height), 1.75)
    working = image.convert("RGB")
    if scale > 1.05:
        working = working.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )

    border_x = max(16, int(round(working.size[0] * 0.02)))
    border_y = max(40, int(round(working.size[1] * 0.5)))
    expanded = ImageOps.expand(
        working,
        border=(border_x, border_y, border_x, border_y),
        fill="white",
    )
    return expanded, expanded.size != image.size


def _needs_banner_preprocessing(image: Image.Image) -> bool:
    width, height = image.size
    if width <= 0 or height <= 0:
        return False
    return height <= 65 or (height / width) <= 0.05


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


def _apply_identity_banner_fallback(
    document_result: dict,
    *,
    image_bytes: bytes,
    suffix: str,
    profile_key: str,
) -> None:
    if suffix == ".pdf":
        return

    raw_document = document_result.get("raw_document") or {}
    blocks = raw_document.get("blocks") or []

    with Image.open(BytesIO(image_bytes)) as image:
        prepared_image = ImageOps.exif_transpose(image).convert("RGB")
        if not _is_short_banner_image(prepared_image):
            return

        current_score = _score_identity_signal(blocks)
        if current_score >= 80:
            return

        banner_blocks = _extract_identity_banner_blocks(prepared_image, profile_key=profile_key)
        if not banner_blocks:
            return
        banner_score = _score_identity_signal(banner_blocks)
        if banner_score <= current_score:
            return

    _apply_synthetic_blocks(document_result, banner_blocks)


def _is_short_banner_image(image: Image.Image) -> bool:
    width, height = image.size
    if width <= 0 or height <= 0:
        return False
    return height <= 120 or (height / width) <= 0.10


def _extract_identity_banner_blocks(image: Image.Image, *, profile_key: str) -> list[dict]:
    best_blocks: list[dict] = []
    best_score = -1

    for candidate in _iter_banner_ocr_images(image):
        lines: list[str] = []
        seen: set[str] = set()
        for crop in _iter_banner_ocr_crops(candidate):
            crop_bytes, crop_suffix = _encode_image(crop)
            text_results = _predict_text_results(crop_bytes, suffix=crop_suffix, profile_key=profile_key)
            for line in _extract_text_lines_from_ocr_results(text_results):
                signature = line.lower()
                if signature in seen:
                    continue
                seen.add(signature)
                lines.append(line)
        blocks = _build_identity_banner_blocks(lines)
        score = _score_banner_blocks(blocks)
        if score > best_score:
            best_score = score
            best_blocks = blocks

    return best_blocks


def _iter_banner_ocr_images(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    scale = 1
    if height < 160:
        scale = max(scale, int(math.ceil(160 / max(height, 1))))
    scale = min(max(scale, 4), 10)

    resized = image.resize((width * scale, height * scale), Image.Resampling.LANCZOS)
    padded = ImageOps.expand(resized, border=(32, 96, 32, 96), fill="white")
    grayscale = ImageOps.autocontrast(ImageOps.grayscale(padded))
    sharpened = grayscale.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=2))
    thresholded = sharpened.point(lambda value: 255 if value > 168 else 0)
    return [
        resized,
        padded,
        grayscale.convert("RGB"),
        sharpened.convert("RGB"),
        thresholded.convert("RGB"),
    ]


def _iter_banner_ocr_crops(image: Image.Image) -> list[Image.Image]:
    crops = [image]
    width, height = image.size
    if width < 600:
        return crops

    boxes = [
        (0.0, 0.0, 0.42, 1.0),
        (0.18, 0.0, 0.62, 1.0),
        (0.38, 0.0, 0.82, 1.0),
        (0.58, 0.0, 1.0, 1.0),
    ]
    for left, top, right, bottom in boxes:
        crop = image.crop((
            max(0, int(width * left)),
            max(0, int(height * top)),
            min(width, int(width * right)),
            min(height, int(height * bottom)),
        ))
        if crop.size[0] > 10 and crop.size[1] > 10:
            crops.append(crop)
    return crops


def _encode_image(image: Image.Image) -> tuple[bytes, str]:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), ".png"


def _extract_text_lines_from_ocr_results(results) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        text = _normalize_ocr_line(value)
        if not text:
            return
        signature = text.lower()
        if signature in seen:
            return
        seen.add(signature)
        lines.append(text)

    def walk(node) -> None:
        if isinstance(node, str):
            add(node)
            return
        if hasattr(node, "json"):
            json_value = getattr(node, "json", None)
            if isinstance(json_value, dict):
                walk(json_value)
                return
        if hasattr(node, "res"):
            res_value = getattr(node, "res", None)
            if isinstance(res_value, dict):
                walk(res_value)
                return
        if hasattr(node, "rec_texts"):
            rec_texts = getattr(node, "rec_texts", None)
            if isinstance(rec_texts, list):
                for value in rec_texts:
                    if isinstance(value, str):
                        add(value)
        if hasattr(node, "__dict__") and getattr(node, "__dict__", None):
            walk(vars(node))
            return
        if isinstance(node, dict):
            for key in ("rec_texts", "texts"):
                values = node.get(key)
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str):
                            add(value)
            text_value = node.get("text")
            if isinstance(text_value, str):
                add(text_value)
            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    walk(value)
            return
        if isinstance(node, (list, tuple)):
            if (
                len(node) == 2
                and isinstance(node[1], (list, tuple))
                and node[1]
                and isinstance(node[1][0], str)
            ):
                add(node[1][0])
                return
            for value in node:
                walk(value)

    walk(results)
    return lines


def _build_identity_banner_blocks(lines: list[str]) -> list[dict]:
    cleaned_lines = [_normalize_ocr_line(line) for line in lines if _normalize_ocr_line(line)]
    if not cleaned_lines:
        return []

    case_count = next((line for line in cleaned_lines if CASE_COUNT_RE.fullmatch(line)), "")
    content_lines = [line for line in cleaned_lines if not _looks_like_identity_header_line(line)]
    field_values = _extract_identity_fields_from_labeled_lines(content_lines)
    inferred_values = _infer_identity_fields_from_text(" ".join(content_lines))
    for header in IDENTITY_COLUMNS:
        if not field_values.get(header) and inferred_values.get(header):
            field_values[header] = inferred_values[header]
    if field_values.get("PERM ID"):
        field_values["PERM ID"] = _normalize_perm_id(field_values["PERM ID"])

    if not field_values.get("PERM ID"):
        return []

    row = [field_values.get(header, "") for header in IDENTITY_COLUMNS]
    blocks: list[dict] = [
        {
            "type": "table",
            "title": None,
            "headers": IDENTITY_COLUMNS,
            "rows": [row],
        }
    ]
    if case_count:
        blocks.append({"type": "text", "text": case_count})
    return blocks


def _extract_identity_fields_from_labeled_lines(lines: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    label_map = {
        "perm id": "PERM ID",
        "participant": "Participant",
        "gender": "Gender",
        "dob": "DOB",
        "address": "Address",
        "action": "Action",
    }
    for line in lines:
        if _looks_like_identity_header_line(line):
            continue
        match = re.match(
            r"^(PERM ID|Participant|Gender|DOB|Address|Action)\s*:?\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if not match:
            continue
        key = label_map[match.group(1).strip().lower()]
        values[key] = match.group(2).strip()
    return values


def _infer_identity_fields_from_text(text: str) -> dict[str, str]:
    plain = _normalize_ocr_line(text)
    if not plain:
        return {}

    plain = re.sub(r"\bPERM\s+Review\s+System\b", " ", plain, flags=re.IGNORECASE)
    plain = CASE_COUNT_RE.sub(" ", plain)
    plain = re.sub(
        r"\b(PERM\s+ID|Participant|Gender|DOB|Address|Action)\b",
        " ",
        plain,
        flags=re.IGNORECASE,
    )
    plain = _normalize_ocr_line(plain)
    if not plain:
        return {}

    values = {header: "" for header in IDENTITY_COLUMNS}

    perm_id_match = PERM_ID_RE.search(plain)
    if perm_id_match:
        values["PERM ID"] = _normalize_perm_id(perm_id_match.group(1))

    gender_match = GENDER_RE.search(plain)
    if gender_match:
        values["Gender"] = gender_match.group(1).title()

    dob_match = DATE_RE.search(plain)
    if dob_match:
        values["DOB"] = dob_match.group(0)

    action_match = ACTION_RE.search(plain)
    if action_match:
        values["Action"] = action_match.group(1).title()

    plain_address_match = ADDRESS_LINE_RE.search(plain)
    address_source = plain
    if dob_match:
        address_start = dob_match.end()
        address_end = action_match.start() if action_match else len(plain)
        address_source = plain[address_start:address_end]
    address_match = ADDRESS_LINE_RE.search(address_source) or plain_address_match
    if address_match:
        values["Address"] = _normalize_ocr_line(address_match.group(0))

    if perm_id_match:
        candidate_end_positions = [
            match.start()
            for match in (gender_match, dob_match, plain_address_match, action_match)
            if match
        ]
        candidate_end = min(candidate_end_positions) if candidate_end_positions else len(plain)
        name_candidate = plain[perm_id_match.end():candidate_end]
        name_candidate = re.sub(r"[:|]+", " ", name_candidate)
        name_candidate = _normalize_ocr_line(name_candidate)
        if _looks_like_person_name(name_candidate):
            values["Participant"] = name_candidate

    return values


def _score_banner_blocks(blocks: list[dict]) -> int:
    if not blocks:
        return -1
    score = 0
    for block in blocks:
        if block.get("type") == "table":
            rows = block.get("rows") or []
            if rows:
                score += sum(1 for value in rows[0] if str(value).strip()) * 10
        elif block.get("type") == "text":
            score += 1
    return score


def _score_identity_signal(blocks: list[dict]) -> int:
    if not blocks:
        return 0

    score = 0
    collected: list[str] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = _normalize_ocr_line(block.get("text", ""))
            if text:
                collected.append(text)
            continue

        if block_type != "table":
            continue

        headers = [_normalize_ocr_line(value) for value in block.get("headers", [])]
        rows = block.get("rows") or []
        if headers == IDENTITY_COLUMNS:
            score += 80
        elif any(header.lower() == "perm id" for header in headers):
            score += 20

        collected.extend(text for text in headers if text)
        for row in rows[:2]:
            collected.extend(_normalize_ocr_line(value) for value in row if _normalize_ocr_line(value))

    combined = " ".join(collected)
    if PERM_ID_RE.search(combined):
        score += 40
    if GENDER_RE.search(combined):
        score += 10
    if DATE_RE.search(combined):
        score += 15
    if ADDRESS_LINE_RE.search(combined):
        score += 15
    if ACTION_RE.search(combined):
        score += 10
    if _looks_like_person_name(_extract_name_candidate(combined)):
        score += 10

    return score


def _normalize_perm_id(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9-]", "", str(value or "").upper())
    if normalized.startswith("DOM"):
        return "DCM" + normalized[3:]
    return normalized


def _extract_name_candidate(value: str) -> str:
    if not value:
        return ""

    perm_id_match = PERM_ID_RE.search(value)
    if not perm_id_match:
        return ""

    tail = value[perm_id_match.end():]
    stop_positions = [
        match.start()
        for match in (
            GENDER_RE.search(tail),
            DATE_RE.search(tail),
            ADDRESS_LINE_RE.search(tail),
            ACTION_RE.search(tail),
        )
        if match
    ]
    name_segment = tail[: min(stop_positions)] if stop_positions else tail
    return _normalize_ocr_line(name_segment)


def _looks_like_person_name(value: str) -> bool:
    if not value or any(char.isdigit() for char in value):
        return False
    parts = value.split()
    return 1 < len(parts) <= 4 and all(part and part[0].isalpha() for part in parts)


def _looks_like_identity_header_line(value: str) -> bool:
    lowered = value.lower()
    header_hits = sum(
        1
        for label in ("perm id", "participant", "gender", "dob", "address", "action")
        if label in lowered
    )
    return header_hits >= 3


def _normalize_ocr_line(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" |:\t\r\n")


def _apply_synthetic_blocks(document_result: dict, blocks: list[dict]) -> None:
    raw_document = document_result.setdefault("raw_document", {})
    raw_document["blocks"] = _dedupe_blocks(blocks)

    markdown = _blocks_to_markdown(raw_document["blocks"])
    table_entries = [_table_block_to_legacy_entry(block) for block in raw_document["blocks"] if block.get("type") == "table"]

    document_result["raw_text"] = markdown
    document_result["markdown"] = markdown
    document_result["tables"] = table_entries

    pages = document_result.get("pages") or []
    if pages:
        pages[0]["markdown"] = markdown
        pages[0]["tables"] = table_entries
        pages[0]["structured_fields"] = pages[0].get("structured_fields") or []
        pages[0]["layout_blocks"] = max(len(raw_document["blocks"]), pages[0].get("layout_blocks", 0))

    summary = document_result.get("summary") or {}
    summary["table_count"] = len(table_entries)
    summary["text_characters"] = len(markdown)
    document_result["summary"] = summary


def _blocks_to_markdown(blocks: list[dict]) -> str:
    lines: list[str] = []
    for block in blocks:
        if block.get("type") == "table":
            headers = [str(value).strip() for value in block.get("headers", [])]
            rows = block.get("rows") or []
            if headers:
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                padded = [str(value).strip() for value in row]
                padded.extend([""] * max(0, len(headers) - len(padded)))
                lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
        elif block.get("type") == "text":
            text = str(block.get("text", "") or "").strip()
            if text:
                lines.append(text)
    return "\n".join(lines).strip()


def _table_block_to_legacy_entry(block: dict) -> dict:
    headers = [str(value).strip() for value in block.get("headers", [])]
    rows = [[str(value).strip() for value in row] for row in (block.get("rows") or [])]
    return {
        "page_index": 0,
        "bbox": [0, 0, 0, 0],
        "html": "",
        "rows": [headers, *rows] if headers else rows,
        "cells": [],
    }


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
        (0.0, 0.0, 1.0, 0.24),
        (0.0, 0.0, 1.0, 0.28),
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

        fallback_text = _extract_targeted_header_texts(crop_document.get("raw_text", ""))
        for text in fallback_text:
            signature = text.lower()
            if signature in seen:
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


def _extract_targeted_header_texts(value: str) -> list[str]:
    plain = re.sub(r"<[^>]+>", " ", str(value or ""))
    plain = re.sub(r"\s+", " ", plain).strip()
    if not plain:
        return []

    texts: list[str] = []

    perm_id_match = re.search(r"perm\s+id\s*:?\s*([A-Z0-9-]{6,})", plain, re.IGNORECASE)
    if perm_id_match:
        texts.append(f"PERM ID: {_normalize_perm_id(perm_id_match.group(1))}")

    marital_match = re.search(r"marital\s*:?\s*([A-Za-z][A-Za-z /-]{1,40})", plain, re.IGNORECASE)
    if marital_match:
        texts.append(f"Marital {marital_match.group(1).strip()}")

    return texts


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
