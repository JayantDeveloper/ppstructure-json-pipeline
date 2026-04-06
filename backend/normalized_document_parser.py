from __future__ import annotations

from pathlib import Path
import re

from table_parser import ELIGIBILITY_COLUMNS

IDENTITY_COLUMNS = [
    "PERM ID",
    "Participant",
    "Gender",
    "DOB",
    "Address",
    "Action",
]
BENEFIT_COLUMNS = ["Benefits", "From", "To"]
DECISION_COLUMNS = ["Coverage Period", "Decision"]

ADDRESS_RE = re.compile(
    r"\b(street|st\.?|avenue|ave\.?|road|rd\.?|drive|dr\.?|lane|ln\.?|court|ct\.?|boulevard|blvd\.?|way)\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
PERM_ID_RE = re.compile(r"perm\s+id\s*:?\s*([A-Z0-9-]{6,})", re.IGNORECASE)
MENU_VALUES = [
    "Home",
    "Eligibility",
    "Evidence",
    "Cases and Applications",
    "Issues and Proceedings",
    "Financial Transactions",
    "Client Contact",
    "Administration",
    "Compliance",
    "Add Picture...",
]


def normalize_document_result(document_result: dict, *, processing: dict | None = None) -> dict:
    raw_document = document_result.get("raw_document") or _fallback_raw_document(
        document_result,
        processing=processing,
    )
    blocks = raw_document.get("blocks") or []

    metadata = {
        "filename": raw_document.get("metadata", {}).get("filename")
        or document_result.get("source"),
        "page_number": raw_document.get("metadata", {}).get("page_number", 1),
        "ocr_mode": raw_document.get("metadata", {}).get("ocr_mode")
        or (processing or {}).get("ocr_mode_label")
        or (processing or {}).get("ocr_mode"),
        "run_timestamp": raw_document.get("metadata", {}).get("run_timestamp")
        or (processing or {}).get("run_timestamp"),
    }

    document_context = _extract_document_context(blocks)
    key_facts = _extract_key_facts(blocks)
    tables, found_structured = _extract_tables_and_kind(blocks)
    filtered_noise = _collect_filtered_noise(blocks, document_context)
    ocr_text = [] if found_structured else _collect_fallback_ocr_text(
        blocks,
        document_context=document_context,
        key_facts=key_facts,
        filtered_noise=filtered_noise,
    )

    return {
        "image_id": raw_document.get("image_id") or Path(document_result.get("source") or "document").stem,
        "metadata": metadata,
        "document_context": document_context,
        "key_facts": key_facts,
        "tables": tables,
        "ocr_text": _dedupe_preserve_order(ocr_text),
        "filtered_noise": _dedupe_preserve_order(filtered_noise),
    }


def _extract_document_context(blocks: list[dict]) -> dict[str, str]:
    context: dict[str, str] = {}
    for block in blocks:
        if block.get("type") != "text":
            continue
        text = _normalize_text_value(block.get("text", ""))
        lowered = text.lower()
        if lowered.startswith("dc ") and "system_name" not in context:
            context["system_name"] = text
        elif lowered == "eligibility determination" and "screen_title" not in context:
            context["screen_title"] = text
        elif lowered == "eligibility" and "section" not in context:
            context["section"] = text
    return context


def _extract_key_facts(blocks: list[dict]) -> dict[str, str]:
    key_facts: dict[str, str] = {}

    for block in blocks:
        if block.get("type") == "table":
            _merge_key_facts(key_facts, _extract_key_facts_from_table(block))

    for block in blocks:
        if block.get("type") != "text":
            continue
        text = _normalize_text_value(block.get("text", ""))
        lowered = text.lower()
        if not text or lowered in {"home", "eligibility", "eligibility determination"}:
            continue

        perm_id_match = PERM_ID_RE.search(text)
        if perm_id_match and "PERM ID" not in key_facts:
            key_facts["PERM ID"] = perm_id_match.group(1)
            continue

        marital_match = re.search(r"marital\s*:?\s*(.+)", text, re.IGNORECASE)
        if marital_match and "Marital" not in key_facts:
            key_facts["Marital"] = marital_match.group(1).strip()
            continue

        if "Born" not in key_facts and DATE_RE.search(text) and "born" in lowered:
            key_facts["Born"] = _clean_date(text)
            continue

        if "Address" not in key_facts and ADDRESS_RE.search(text):
            key_facts["Address"] = text
            continue

        if "Client Name" not in key_facts and _looks_like_name(text) and text not in MENU_VALUES:
            key_facts["Client Name"] = text

    return key_facts


def _extract_tables_and_kind(blocks: list[dict]) -> tuple[list[dict], bool]:
    normalized_tables: list[dict] = []
    found_structured = False
    generic_table_index = 1

    for block in blocks:
        if block.get("type") != "table":
            continue

        headers = [str(value).strip() for value in block.get("headers", [])]
        rows = block.get("rows", []) or []

        if headers == ELIGIBILITY_COLUMNS:
            normalized_tables.append(
                {
                    "table_id": "eligibility_programs",
                    "title": block.get("title") or "Eligibility",
                    "columns": ELIGIBILITY_COLUMNS,
                    "rows": [
                        _row_to_object(ELIGIBILITY_COLUMNS, normalized)
                        for row in rows
                        if (normalized := _normalize_eligibility_row(row))
                    ],
                }
            )
            found_structured = True
            continue

        if headers == BENEFIT_COLUMNS:
            normalized_tables.append(
                {
                    "table_id": "benefit_periods",
                    "title": block.get("title") or "Benefits",
                    "columns": BENEFIT_COLUMNS,
                    "rows": [_row_to_object(BENEFIT_COLUMNS, _pad_row(row, len(BENEFIT_COLUMNS))) for row in rows],
                }
            )
            found_structured = True
            continue

        if headers == IDENTITY_COLUMNS:
            normalized_tables.append(
                {
                    "table_id": "participant_identity",
                    "title": block.get("title") or "Participant Data",
                    "columns": IDENTITY_COLUMNS,
                    "rows": [_row_to_object(IDENTITY_COLUMNS, _pad_row(row, len(IDENTITY_COLUMNS))) for row in rows],
                }
            )
            found_structured = True
            continue

        if headers == DECISION_COLUMNS:
            normalized_tables.append(
                {
                    "table_id": "coverage_decision",
                    "title": block.get("title") or "Coverage Decision",
                    "columns": DECISION_COLUMNS,
                    "rows": [_row_to_object(DECISION_COLUMNS, _pad_row(row, len(DECISION_COLUMNS))) for row in rows],
                }
            )
            found_structured = True
            continue

        generic_table = _normalize_generic_table(block, generic_table_index)
        if generic_table:
            normalized_tables.append(generic_table)
            found_structured = True
            generic_table_index += 1

    normalized_tables = [
        table
        for table in normalized_tables
        if table.get("rows") or table.get("columns")
    ]
    return normalized_tables, found_structured


def _extract_key_facts_from_table(block: dict) -> dict[str, str]:
    headers = [str(value).strip() for value in block.get("headers", [])]
    rows = block.get("rows", []) or []
    if headers == IDENTITY_COLUMNS and rows:
        first_row = _pad_row(rows[0], len(IDENTITY_COLUMNS))
        mapped = _row_to_object(IDENTITY_COLUMNS, first_row)
        return {
            "PERM ID": mapped.get("PERM ID", ""),
            "Client Name": mapped.get("Participant", ""),
            "Gender": mapped.get("Gender", ""),
            "Born": _clean_date(mapped.get("DOB", "")),
            "Address": mapped.get("Address", ""),
        }

    if headers == DECISION_COLUMNS and rows:
        first_row = _row_to_object(DECISION_COLUMNS, _pad_row(rows[0], len(DECISION_COLUMNS)))
        return {
            "Coverage Period": first_row.get("Coverage Period", ""),
            "Decision": first_row.get("Decision", ""),
        }

    return {}


def _normalize_eligibility_row(row: list[str]) -> list[str]:
    values = [str(value).strip() for value in row if str(value).strip()]
    if len(values) < 6:
        return []

    pdc_case = values[0] if len(values) >= 1 else ""
    ic_case = values[1] if len(values) >= 2 else ""
    program_name = _normalize_program_name(values[2] if len(values) >= 3 else "")

    index = 3
    program_code = ""
    if index < len(values) and _looks_like_program_code(values[index]):
        program_code = values[index]
        index += 1

    dates: list[str] = []
    while index < len(values) and len(dates) < 3 and DATE_RE.search(values[index]):
        dates.append(_clean_date(values[index]))
        index += 1
    while len(dates) < 3:
        dates.append("")

    remaining = values[index:]
    status = ""
    assistance = ""

    if remaining:
        if len(remaining) == 1:
            assistance = remaining[0]
        elif _looks_like_assistance_unit(remaining[0]) and _looks_like_case_number(remaining[1]):
            assistance = remaining[0]
        elif _looks_like_assistance_unit(remaining[0]) and not _looks_like_status_value(remaining[0]):
            assistance = remaining[0]
        else:
            status = remaining[0]
            assistance = remaining[1] if len(remaining) > 1 else ""

    return [
        pdc_case,
        ic_case,
        program_name,
        program_code,
        dates[0],
        dates[1],
        dates[2],
        status,
        assistance,
    ]


def _collect_filtered_noise(blocks: list[dict], document_context: dict[str, str]) -> list[str]:
    noise: list[str] = []
    for block in blocks:
        if block.get("type") == "noise":
            noise.extend(_expand_noise_text(str(block.get("text", "") or "")))
            continue

        if block.get("type") != "text":
            continue

        text = _normalize_text_value(block.get("text", ""))
        if text in {"Home", "Eligibility"}:
            noise.append(text)
            continue
        if text == document_context.get("section"):
            noise.append(text)

    return noise


def _collect_fallback_ocr_text(
    blocks: list[dict],
    *,
    document_context: dict[str, str],
    key_facts: dict[str, str],
    filtered_noise: list[str],
) -> list[str]:
    context_values = {value for value in document_context.values() if value}
    fact_values = {value for value in key_facts.values() if value}
    noise_values = {value for value in filtered_noise if value}
    lines: list[str] = []

    for block in blocks:
        if block.get("type") != "text":
            continue
        text = _normalize_text_value(block.get("text", ""))
        if not text:
            continue
        if text in context_values or text in fact_values or text in noise_values:
            continue
        lines.append(text)

    return lines


def _expand_noise_text(value: str) -> list[str]:
    lowered = value.lower()
    expanded = []
    for item in MENU_VALUES:
        candidates = {item.lower()}
        if item == "Cases and Applications":
            candidates.add("cases and appllications")
        if item == "Add Picture...":
            candidates.update({"add picture", "add picture..."})
        if any(candidate in lowered for candidate in candidates):
            expanded.append(item)
    return expanded or [value]


def _fallback_raw_document(document_result: dict, *, processing: dict | None = None) -> dict:
    return {
        "image_id": Path(document_result.get("source") or "document").stem,
        "metadata": {
            "filename": document_result.get("source"),
            "page_number": 1,
            "ocr_mode": (processing or {}).get("ocr_mode_label")
            or (processing or {}).get("ocr_mode"),
            "run_timestamp": (processing or {}).get("run_timestamp"),
        },
        "blocks": [],
    }


def _row_to_object(columns: list[str], row: list[str]) -> dict[str, str]:
    return {
        column: row[index] if index < len(row) else ""
        for index, column in enumerate(columns)
    }


def _normalize_generic_table(block: dict, generic_table_index: int) -> dict | None:
    raw_headers = [str(value).strip() for value in block.get("headers", []) if str(value).strip()]
    raw_rows = [
        [str(value).strip() for value in row if str(value).strip()]
        for row in (block.get("rows") or [])
    ]
    raw_rows = [row for row in raw_rows if row]
    if not raw_headers and not raw_rows:
        return None

    max_width = max([len(raw_headers), *[len(row) for row in raw_rows]], default=0)
    if max_width == 0:
        return None

    columns = list(raw_headers)
    while len(columns) < max_width:
        columns.append(f"Extra Column {len(columns) - len(raw_headers) + 1}")

    title = _clean_table_title(block.get("title")) or f"Table {generic_table_index}"
    table_id = _slugify(title) or f"generic_table_{generic_table_index}"
    return {
        "table_id": table_id,
        "title": title,
        "columns": columns,
        "rows": [_row_to_object(columns, _pad_row(row, len(columns))) for row in raw_rows],
    }


def _pad_row(row: list[str], length: int) -> list[str]:
    values = [str(value).strip() for value in row[:length]]
    if len(values) < length:
        values.extend([""] * (length - len(values)))
    return values


def _merge_key_facts(target: dict[str, str], incoming: dict[str, str]) -> None:
    for key, value in incoming.items():
        if key not in target and value:
            target[key] = value


def _looks_like_name(value: str) -> bool:
    if not value or any(ch.isdigit() for ch in value):
        return False
    lowered = value.lower()
    if lowered.startswith("dc "):
        return False
    if lowered in {"participant data", "external system", "eligibility override"}:
        return False
    if any(term in lowered for term in {"recorded", "eligibility", "determination", "participant", "external system", "marital"}):
        return False
    parts = value.split()
    return 1 < len(parts) <= 4 and all(part and part[0].isalpha() for part in parts)


def _looks_like_program_code(value: str) -> bool:
    lowered = value.strip().lower()
    if DATE_RE.search(value):
        return False
    return 0 < len(lowered) <= 8 and any(ch.isdigit() for ch in lowered)


def _looks_like_case_number(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return len(digits) >= 6


def _looks_like_assistance_unit(value: str) -> bool:
    lowered = value.lower()
    if not value or "recorded" in lowered:
        return False
    return "(" in value or "," in value or _looks_like_name(value)


def _looks_like_status_value(value: str) -> bool:
    lowered = value.lower()
    return lowered in {"active", "pending", "closed", "approved", "denied"}


def _normalize_program_name(value: str) -> str:
    return "SSI" if value.strip().lower() == "ss" else value


def _clean_date(value: str) -> str:
    match = DATE_RE.search(value.replace("Bom", "Born"))
    return match.group(0) if match else value.replace("Bom", "").replace("Born", "").strip()


def _clean_table_title(value: str | None) -> str:
    title = _normalize_text_value(value or "")
    title = re.sub(r"\s*-\s*", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _normalize_text_value(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^#{1,6}\s+", "", text)
    return text.strip()


def _slugify(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    return lowered.strip("_")


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        signature = cleaned.lower()
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(cleaned)
    return deduped
