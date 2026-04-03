from __future__ import annotations

from pathlib import Path
import re

from field_parser import parse_structured_fields
from table_parser import ELIGIBILITY_COLUMNS, parse_raw_rows, parse_table, split_table_regions

ADDRESS_RE = re.compile(
    r"\b(street|st\.?|avenue|ave\.?|road|rd\.?|drive|dr\.?|lane|ln\.?|court|ct\.?|boulevard|blvd\.?|way)\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
PHONE_RE = re.compile(r"\b\d{7,12}\b")
MENU_TERMS = {
    "home",
    "eligibility",
    "evidence",
    "cases and applications",
    "cases and appllications",
    "issues and proceedings",
    "financial transactions",
    "client contact",
    "administration",
    "compliance",
    "add picture",
}


def collect_document_result(
    results,
    *,
    source: str,
    processing: dict | None = None,
    page_number: int = 1,
) -> dict:
    all_tables: list[dict] = []
    all_fields: list[dict] = []
    markdown_pages: list[str] = []
    page_results: list[dict] = []
    page_artifacts: list[dict] = []

    for page_index, result in enumerate(results):
        page_json = _result_json(result)
        page_markdown = _result_markdown(result) or _fallback_markdown(page_json)
        page_tables = _extract_tables(page_json, page_index=page_index)
        page_fields = (
            parse_structured_fields(page_markdown, page_index=page_index)
            if page_markdown
            else []
        )

        markdown_pages.append(page_markdown)
        all_tables.extend(page_tables)
        all_fields.extend(page_fields)
        page_results.append(
            {
                "page_index": page_index,
                "markdown": page_markdown,
                "tables": page_tables,
                "structured_fields": page_fields,
                "layout_blocks": len(page_json.get("parsing_res_list", [])),
            }
        )
        page_artifacts.append(
            {
                "page_index": page_index,
                "page_json": page_json,
                "page_markdown": page_markdown,
                "page_tables": page_tables,
            }
        )

    markdown = "\n\n".join(part for part in markdown_pages if part)
    summary = {
        "engine": "PP-StructureV3",
        "page_count": len(page_results),
        "table_count": len(all_tables),
        "field_count": len(all_fields),
        "text_characters": len(markdown),
    }

    return {
        "source": source,
        "summary": summary,
        "raw_text": markdown,
        "markdown": markdown,
        "tables": all_tables,
        "structured_fields": all_fields,
        "pages": page_results,
        "raw_document": build_raw_document(
            page_artifacts,
            source=source,
            processing=processing,
            page_number=page_number,
        ),
    }


def build_raw_document(
    page_artifacts: list[dict],
    *,
    source: str,
    processing: dict | None = None,
    page_number: int = 1,
) -> dict:
    blocks: list[dict] = []

    for artifact in page_artifacts:
        blocks.extend(_extract_raw_blocks_from_page(artifact))

    blocks = _dedupe_blocks(blocks)
    if _contains_eligibility_table(blocks):
        blocks = _reorder_eligibility_blocks(blocks)

    return {
        "image_id": Path(source or "document").stem,
        "metadata": {
            "filename": source,
            "page_number": page_number,
            "ocr_mode": (processing or {}).get("ocr_mode_label")
            or (processing or {}).get("ocr_mode"),
            "run_timestamp": (processing or {}).get("run_timestamp"),
        },
        "blocks": blocks,
    }


def _extract_raw_blocks_from_page(artifact: dict) -> list[dict]:
    page_json = artifact.get("page_json") or {}
    blocks: list[dict] = []

    parsing_blocks = page_json.get("parsing_res_list") or []
    for block in parsing_blocks:
        if not isinstance(block, dict):
            continue

        label = str(block.get("block_label", "")).lower()
        content = str(block.get("block_content", "") or "").strip()
        if not content:
            continue

        if "table" in label and "<table" in content.lower():
            blocks.extend(_blocks_from_table_html(content))
            continue

        blocks.extend(_blocks_from_text_content(content))

    if blocks:
        return blocks

    for table in artifact.get("page_tables") or []:
        html = str(table.get("html", "") or "")
        if "<table" in html.lower():
            blocks.extend(_blocks_from_table_html(html))

    markdown = artifact.get("page_markdown") or ""
    if not blocks and markdown:
        blocks.extend(_blocks_from_text_content(markdown))

    return blocks


def _blocks_from_table_html(html: str) -> list[dict]:
    raw_rows = parse_raw_rows(html)
    eligibility_region = split_table_regions(raw_rows, ELIGIBILITY_COLUMNS)
    if eligibility_region:
        return _build_eligibility_blocks(eligibility_region)

    parsed_table = parse_table(html)
    rows = parsed_table.get("rows") or []
    if not rows:
        return []

    headers = rows[0]
    title = _infer_generic_table_title(raw_rows) or "Table"
    return [
        {
            "type": "table",
            "title": title,
            "headers": headers,
            "rows": rows[1:] if len(rows) > 1 else [],
        }
    ]


def _build_eligibility_blocks(region: dict) -> list[dict]:
    leading_rows = region.get("leading_rows") or []
    header = region.get("header") or ELIGIBILITY_COLUMNS
    data_rows = region.get("data_rows") or []

    blocks: list[dict] = []
    system_name = _extract_system_name(leading_rows)
    if system_name:
        blocks.append({"type": "text", "text": system_name})

    profile_row = _extract_profile_row(leading_rows)
    if profile_row:
        blocks.extend(_profile_row_to_blocks(profile_row))

    screen_title = _extract_screen_title(leading_rows)
    if screen_title:
        blocks.append({"type": "text", "text": screen_title})

    if _contains_text(leading_rows, "Home"):
        blocks.append({"type": "text", "text": "Home"})

    if _contains_text(leading_rows, "Eligibility"):
        blocks.append({"type": "text", "text": "Eligibility"})

    normalized_rows = _normalize_eligibility_rows(data_rows)
    if normalized_rows:
        blocks.append(
            {
                "type": "table",
                "title": "Eligibility",
                "headers": header,
                "rows": normalized_rows,
            }
        )

    noise_values = _extract_noise_values(leading_rows)
    if noise_values:
        blocks.append({"type": "noise", "text": " ".join(noise_values)})

    return blocks


def _profile_row_to_blocks(row: list[str]) -> list[dict]:
    blocks: list[dict] = []
    seen: set[str] = set()
    compact = [cell.strip() for cell in row if cell and cell.strip()]

    name = next((cell for cell in compact if _looks_like_name(cell)), "")
    if name:
        seen.add(name.lower())
        blocks.append({"type": "text", "text": name})

    address = next((cell for cell in compact if ADDRESS_RE.search(cell)), "")
    if address:
        seen.add(address.lower())
        blocks.append({"type": "text", "text": address})

    marital = next((cell for cell in compact if "marital" in cell.lower()), "")
    if marital:
        seen.add(marital.lower())
        blocks.append({"type": "text", "text": marital})

    born_value = next((cell for cell in compact if DATE_RE.search(cell)), "")
    if born_value:
        text = f"Born {_clean_date(born_value)}"
        seen.add(text.lower())
        blocks.append({"type": "text", "text": text})

    phone = next((cell for cell in compact if PHONE_RE.search(cell)), "")
    if phone:
        seen.add(phone.lower())
        blocks.append({"type": "text", "text": phone})

    status_blob = next((cell for cell in compact if "not recorded" in cell.lower()), "")
    if status_blob:
        parts = [part.strip(" +") for part in status_blob.split("+") if part.strip(" +")]
        for part in parts:
            signature = part.lower()
            if signature in seen:
                continue
            seen.add(signature)
            blocks.append({"type": "text", "text": part})

    for cell in compact:
        normalized = cell.lower()
        if normalized in seen:
            continue
        if cell.isdigit() and len(cell) < 7:
            continue
        if normalized in {"male", "female"}:
            continue
        if _looks_like_name(cell) or ADDRESS_RE.search(cell) or DATE_RE.search(cell):
            continue
        if "not recorded" in normalized:
            continue
        if PHONE_RE.search(cell):
            continue
        if "marital" in normalized:
            continue
        if "+" in cell:
            continue
        seen.add(normalized)
        blocks.append({"type": "text", "text": cell})

    return blocks


def _extract_system_name(rows: list[list[str]]) -> str:
    for row in rows:
        compact = [cell for cell in row if cell]
        if len(compact) != 1:
            continue
        value = compact[0]
        if value.lower().startswith("dc "):
            return value
    return ""


def _extract_screen_title(rows: list[list[str]]) -> str:
    for row in rows:
        compact = [cell for cell in row if cell]
        if not compact:
            continue
        if "eligibility determination" in compact[0].lower() and not compact[0].lower().startswith("dc "):
            return compact[0]
    return ""


def _extract_profile_row(rows: list[list[str]]) -> list[str]:
    for row in rows:
        compact = [cell for cell in row if cell]
        if len(compact) < 5:
            continue
        if any(DATE_RE.search(cell) for cell in compact) and any(ADDRESS_RE.search(cell) for cell in compact):
            return row
    return []


def _contains_text(rows: list[list[str]], value: str) -> bool:
    needle = value.strip().lower()
    for row in rows:
        for cell in row:
            if cell and cell.strip().lower() == needle:
                return True
    return False


def _extract_noise_values(rows: list[list[str]]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        compact = [cell for cell in row if cell]
        if not compact:
            continue
        menu_hits = [cell for cell in compact if cell.strip().lower() in MENU_TERMS]
        if len(menu_hits) >= 3:
            for cell in compact:
                lowered = cell.strip().lower()
                if cell.strip().isdigit() or lowered not in MENU_TERMS:
                    continue
                cleaned = _fix_noise_spelling(cell)
                signature = cleaned.lower()
                if signature in seen:
                    continue
                seen.add(signature)
                values.append(cleaned)
            continue

        for cell in compact:
            lowered = cell.strip().lower()
            if lowered not in MENU_TERMS:
                continue
            cleaned = _fix_noise_spelling(cell)
            signature = cleaned.lower()
            if signature in seen:
                continue
            seen.add(signature)
            values.append(cleaned)
    return values


def _normalize_eligibility_rows(rows: list[list[str]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    carry: list[str] = []

    for row in rows:
        compact = [cell for cell in row if cell]
        if not compact:
            continue
        tokens = carry + compact
        carry = []

        normalized_row, carry = _consume_eligibility_tokens(tokens)
        if normalized_row:
            normalized.append(normalized_row)

    return normalized


def _consume_eligibility_tokens(tokens: list[str]) -> tuple[list[str], list[str]]:
    if len(tokens) < 7 or not _looks_like_case_number(tokens[0]) or not _looks_like_case_number(tokens[1]):
        return [], []

    index = 0
    pdc_case = tokens[index]
    ic_case = tokens[index + 1]
    index += 2

    program_name = tokens[index]
    index += 1

    program_code = ""
    if index < len(tokens) and _looks_like_program_code(tokens[index]):
        program_code = tokens[index]
        index += 1

    dates: list[str] = []
    while index < len(tokens) and len(dates) < 3 and DATE_RE.search(tokens[index]):
        dates.append(_clean_date(tokens[index]))
        index += 1

    while len(dates) < 3:
        dates.append("")

    remaining = tokens[index:]
    status = ""
    assistance_unit = ""
    carry: list[str] = []

    if remaining:
        if _looks_like_assistance_unit(remaining[0]):
            assistance_unit = remaining[0]
            carry = remaining[1:]
        elif len(remaining) >= 2:
            status = remaining[0]
            assistance_unit = remaining[1]
            carry = remaining[2:]
        else:
            assistance_unit = remaining[0]

    return [
        pdc_case,
        ic_case,
        _normalize_program_name(program_name),
        program_code,
        dates[0],
        dates[1],
        dates[2],
        status,
        assistance_unit,
    ], carry


def _blocks_from_text_content(content: str) -> list[dict]:
    plain = re.sub(r"<[^>]+>", " ", content)
    lines = [re.sub(r"\s+", " ", line).strip() for line in plain.splitlines()]
    blocks: list[dict] = []

    for line in lines:
        if not line or line == "---":
            continue
        lowered = line.lower()
        if lowered.startswith("imgs/") or "width=" in lowered or "alt=" in lowered:
            continue
        if _looks_like_noise_text(line):
            blocks.append({"type": "noise", "text": _fix_noise_spelling(line)})
        else:
            blocks.append({"type": "text", "text": line})

    return blocks


def _looks_like_noise_text(value: str) -> bool:
    lowered = value.lower()
    if sum(1 for term in MENU_TERMS if term in lowered) >= 3:
        return True
    return lowered.strip() in MENU_TERMS


def _contains_eligibility_table(blocks: list[dict]) -> bool:
    return any(
        block.get("type") == "table"
        and [cell.strip() for cell in block.get("headers", [])] == ELIGIBILITY_COLUMNS
        for block in blocks
    )


def _reorder_eligibility_blocks(blocks: list[dict]) -> list[dict]:
    def _priority(block: dict) -> int:
        btype = block.get("type")
        if btype == "noise":
            return 14
        if btype == "table":
            return 12
        text = str(block.get("text", "") or "").strip()
        if not text:
            return 13
        lowered = text.lower()
        if lowered.startswith("dc "):
            return 0
        if "perm id" in lowered:
            return 8
        if text == "Eligibility Determination":
            return 9
        if text == "Home":
            return 10
        if text == "Eligibility":
            return 11
        if "recorded" in lowered or "+" in text:
            return 6
        if _looks_like_name(text):
            return 1
        if ADDRESS_RE.search(text):
            return 2
        if "marital" in lowered:
            return 3
        if "born" in lowered:
            return 4
        if PHONE_RE.search(text):
            return 5
        return 7

    return _dedupe_blocks(sorted(blocks, key=_priority))


def _dedupe_blocks(blocks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []

    for block in blocks:
        signature = _block_signature(block)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        unique.append(block)

    return unique


def _block_signature(block: dict) -> str:
    block_type = block.get("type")
    if block_type == "table":
        return f"table::{block.get('title')}::{block.get('headers')}::{block.get('rows')}"
    text = str(block.get("text", "") or "").strip().lower()
    return f"{block_type}::{text}" if text else ""


def _infer_generic_table_title(rows: list[list[str]]) -> str:
    for row in rows[:2]:
        compact = [cell for cell in row if cell]
        if len(compact) == 1 and any(ch.isalpha() for ch in compact[0]):
            return compact[0]
    return ""


def _looks_like_name(value: str) -> bool:
    if not value or any(ch.isdigit() for ch in value):
        return False
    lowered = value.lower()
    if lowered.startswith("dc "):
        return False
    if any(term in lowered for term in {"recorded", "eligibility", "determination", "participant", "external system", "marital"}):
        return False
    parts = value.split()
    return 1 < len(parts) <= 4 and all(part and part[0].isalpha() for part in parts)


def _looks_like_case_number(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return len(digits) >= 6


def _looks_like_program_code(value: str) -> bool:
    lowered = value.strip().lower()
    if DATE_RE.search(value):
        return False
    return 0 < len(lowered) <= 8 and any(ch.isdigit() for ch in lowered)


def _looks_like_assistance_unit(value: str) -> bool:
    lowered = value.lower()
    if "recorded" in lowered:
        return False
    return "(" in value or "," in value or _looks_like_name(value)


def _normalize_program_name(value: str) -> str:
    return "SSI" if value.strip().lower() == "ss" else value


def _clean_date(value: str) -> str:
    match = DATE_RE.search(value.replace("Bom", "Born"))
    return match.group(0) if match else value.replace("Bom", "").replace("Born", "").strip()


def _fix_noise_spelling(value: str) -> str:
    return (
        value
        .replace("Eligibillity", "Eligibility")
        .replace("Appllications", "Applications")
        .strip()
    )


def _result_json(result) -> dict:
    json_data = getattr(result, "json", None)
    if isinstance(json_data, dict):
        return json_data.get("res", json_data)
    if isinstance(result, dict):
        return result.get("res", result)
    return {}


def _result_markdown(result) -> str:
    markdown_data = getattr(result, "markdown", None)
    if isinstance(markdown_data, dict):
        return str(markdown_data.get("markdown_texts", "") or "").strip()
    if isinstance(markdown_data, str):
        return markdown_data.strip()
    return ""


def _fallback_markdown(page_json: dict) -> str:
    blocks = page_json.get("parsing_res_list") or []
    lines = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        label = str(block.get("block_label", "")).lower()
        if label in {
            "image",
            "chart",
            "formula",
            "display_formula",
            "inline_formula",
            "header_image",
            "footer_image",
        }:
            continue
        content = str(block.get("block_content", "") or "").strip()
        if not content:
            continue
        if "table" in label and "<table" in content.lower():
            continue
        lines.append(content)
    return "\n\n".join(lines).strip()


def _extract_tables(page_json: dict, *, page_index: int) -> list[dict]:
    tables: list[dict] = []
    seen_signatures: set[tuple] = set()

    for block in page_json.get("parsing_res_list") or []:
        if not isinstance(block, dict):
            continue
        label = str(block.get("block_label", "")).lower()
        if "table" not in label:
            continue
        html = str(block.get("block_content", "") or "")
        if "<table" not in html.lower():
            continue
        parsed_table = parse_table(html)
        signature = _table_signature(parsed_table["rows"])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        tables.append(
            {
                "page_index": page_index,
                "bbox": _normalize_bbox(block.get("block_bbox")),
                "html": html,
                "rows": parsed_table["rows"],
                "cells": parsed_table["cells"],
            }
        )

    for table in page_json.get("table_res_list") or []:
        if not isinstance(table, dict):
            continue
        html = _extract_table_html(table)
        if not html:
            continue
        parsed_table = parse_table(html)
        signature = _table_signature(parsed_table["rows"])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        candidate = {
            "page_index": page_index,
            "bbox": _normalize_bbox(
                table.get("bbox")
                or table.get("table_bbox")
                or table.get("table_region_bbox")
                or table.get("cell_box_list")
            ),
            "html": html,
            "rows": parsed_table["rows"],
            "cells": parsed_table["cells"],
        }
        if candidate not in tables:
            tables.append(candidate)

    return tables


def _extract_table_html(table: dict) -> str:
    for key in ("html", "table_html", "pred_html", "pred"):
        value = table.get(key)
        if isinstance(value, str) and "<table" in value.lower():
            return value
    return ""


def _normalize_bbox(bbox) -> list[int]:
    if not bbox:
        return [0, 0, 0, 0]
    if (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
    ):
        return [int(value) for value in bbox]
    if (
        isinstance(bbox, list)
        and len(bbox) >= 4
        and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in bbox[:4])
    ):
        xs = [float(point[0]) for point in bbox[:4]]
        ys = [float(point[1]) for point in bbox[:4]]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    return [0, 0, 0, 0]


def _table_signature(rows: list[list[str]]) -> tuple:
    return tuple(
        tuple(cell.strip().lower() for cell in row)
        for row in rows
        if any(cell.strip() for cell in row)
    )
