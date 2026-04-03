from __future__ import annotations

import re


_COLON_PAIR_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9/&()#.,' -]{1,60}?)\s*:\s*(.+?)\s*$"
)


def parse_structured_fields(raw_text: str, *, page_index: int | None = None) -> list[dict]:
    lines = [_normalize_line(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

    fields: list[dict] = []
    used_indices: set[int] = set()

    for idx, line in enumerate(lines):
        match = _COLON_PAIR_RE.match(line)
        if not match:
            continue
        field = _clean_field(match.group(1))
        value = _clean_value(match.group(2))
        if not field or not value:
            continue
        used_indices.add(idx)
        fields.append(_build_field(field, value, page_index, method="colon"))

    for idx in range(len(lines) - 1):
        if idx in used_indices or idx + 1 in used_indices:
            continue

        field_line = lines[idx]
        value_line = lines[idx + 1]
        if not _looks_like_field_label(field_line):
            continue
        if not _looks_like_value(value_line):
            continue

        field = _clean_field(field_line)
        value = _clean_value(value_line)
        if not field or not value:
            continue

        used_indices.add(idx)
        used_indices.add(idx + 1)
        fields.append(_build_field(field, value, page_index, method="stacked"))

    return _dedupe_fields(fields)


def _build_field(field: str, value: str, page_index: int | None, *, method: str) -> dict:
    item = {
        "field": field,
        "value": value,
        "method": method,
    }
    if page_index is not None:
        item["page_index"] = page_index
    return item


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _clean_field(field: str) -> str:
    return _normalize_line(field).strip("-: ")


def _clean_value(value: str) -> str:
    return _normalize_line(value).strip("-: ")


def _looks_like_field_label(line: str) -> bool:
    if ":" in line or len(line) > 48:
        return False
    tokens = line.split()
    if not 1 <= len(tokens) <= 6:
        return False
    if sum(ch.isdigit() for ch in line) > max(2, len(line) // 4):
        return False
    if not any(ch.isalpha() for ch in line):
        return False
    if line.lower() in {"yes", "no", "male", "female"}:
        return False
    return True


def _looks_like_value(line: str) -> bool:
    if len(line) < 2:
        return False
    if _looks_like_field_label(line) and len(line.split()) <= 3 and not any(
        ch.isdigit() for ch in line
    ):
        return False
    return True


def _dedupe_fields(fields: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    unique: list[dict] = []
    for item in fields:
        key = (item.get("page_index"), item["field"].lower(), item["value"].lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
