from __future__ import annotations

from html.parser import HTMLParser
import re

ELIGIBILITY_COLUMNS = [
    "PDC Case No.",
    "IC Case No.",
    "Program Name",
    "Program Code",
    "Start Date",
    "End Date",
    "Recertification/Renewal Date",
    "Recertification/Renewal Status",
    "Assistance Unit",
]


class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: str | None = None
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._current_row = []
        elif tag in {"td", "th"}:
            self._current_cell = ""
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in {"td", "th"}:
            if self._current_row is not None and self._current_cell is not None:
                self._current_row.append(self._current_cell.strip())
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr":
            if self._current_row is not None:
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data):
        if self._in_cell and self._current_cell is not None:
            self._current_cell += data


def parse_table(html: str) -> dict:
    raw_rows = parse_raw_rows(html)
    rows = _refine_rows(raw_rows)

    return {
        "rows": rows,
        "cells": _extract_field_value_cells(rows),
    }


def parse_raw_rows(html: str) -> list[list[str]]:
    parser = _TableHTMLParser()
    parser.feed(html)
    return _normalize_rows(parser.rows)


def find_header_index(rows: list[list[str]], columns: list[str]) -> int | None:
    targets = [_normalize_key(column) for column in columns]

    for index, row in enumerate(rows):
        normalized = [_normalize_key(cell) for cell in row if cell]
        if len(normalized) < min(3, len(targets)):
            continue
        matches = sum(1 for target in targets if target in normalized)
        if matches >= max(2, len(targets) - 2):
            return index

    return None


def split_table_regions(rows: list[list[str]], columns: list[str]) -> dict | None:
    normalized_rows = _normalize_rows(rows)
    header_index = find_header_index(normalized_rows, columns)
    if header_index is None:
        return None

    leading_rows = normalized_rows[:header_index]
    header = _pad_row(normalized_rows[header_index], len(columns))
    data_rows: list[list[str]] = []
    empty_rows = 0

    for row in normalized_rows[header_index + 1 :]:
        compact = [cell for cell in row if cell]
        if not compact:
            empty_rows += 1
            if data_rows and empty_rows >= 2:
                break
            continue
        empty_rows = 0
        if len(compact) < 2:
            if data_rows:
                break
            continue
        data_rows.append(row)

    return {
        "leading_rows": leading_rows,
        "header": header,
        "data_rows": data_rows,
    }


def _normalize_rows(rows: list[list[str]]) -> list[list[str]]:
    normalized_rows: list[list[str]] = []
    for row in rows:
        normalized = [_normalize_cell(cell) for cell in row]
        while normalized and not normalized[-1]:
            normalized.pop()
        if any(normalized):
            normalized_rows.append(normalized)
    return normalized_rows


def _refine_rows(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return []

    eligibility_region = split_table_regions(rows, ELIGIBILITY_COLUMNS)
    if eligibility_region and eligibility_region["data_rows"]:
        return [eligibility_region["header"], *eligibility_region["data_rows"]]

    compact_rows = [[cell for cell in row if cell] for row in rows if any(cell for cell in row)]
    if len(compact_rows) <= 2:
        return compact_rows

    header_block = _extract_header_data_block(compact_rows)
    if header_block:
        return header_block

    dense_rows = [row for row in compact_rows if len(row) >= 3]
    if len(dense_rows) >= 2:
        return dense_rows

    pair_rows = [row for row in compact_rows if len(row) >= 2]
    if len(pair_rows) >= 2:
        return pair_rows

    return compact_rows


def _extract_header_data_block(rows: list[list[str]]) -> list[list[str]]:
    best_start: int | None = None
    best_score = 0.0

    for index in range(len(rows) - 1):
        header = rows[index]
        data = rows[index + 1]
        if len(header) < 3 or len(data) < 3:
            continue
        if abs(len(header) - len(data)) > 1:
            continue

        header_alpha = _alphabetic_ratio(header)
        header_numeric = _numeric_ratio(header)
        data_numeric = _numeric_ratio(data)
        if header_alpha < 0.7 or data_numeric < 0.2 or data_numeric <= header_numeric:
            continue

        score = min(len(header), len(data)) + (data_numeric * 4) + ((data_numeric - header_numeric) * 4)
        if score > best_score:
            best_score = score
            best_start = index

    if best_start is None:
        return []

    anchor = rows[best_start]
    kept = [anchor]
    follow_index = best_start + 1
    while follow_index < len(rows):
        row = rows[follow_index]
        if len(row) < 3 or abs(len(row) - len(anchor)) > 1:
            break
        kept.append(row)
        follow_index += 1

    return kept if len(kept) >= 2 else []


def _extract_field_value_cells(rows: list[list[str]]) -> list[dict]:
    if not rows:
        return []

    meaningful_rows = []
    for row in rows:
        compact = [cell for cell in row if cell]
        if compact:
            meaningful_rows.append(compact)

    if not meaningful_rows:
        return []

    # Only flatten narrow, key/value-style tables. Wide tabular data should stay as rows.
    if any(len(row) > 2 for row in meaningful_rows):
        return []

    cells: list[dict] = []
    has_header = _is_header_row(meaningful_rows[0])

    if has_header:
        headers = meaningful_rows[0]
        for row in meaningful_rows[1:]:
            row_values = row + [""] * max(0, len(headers) - len(row))
            for index, header in enumerate(headers):
                value = row_values[index] if index < len(row_values) else ""
                if header or value:
                    cells.append({"field": header, "value": value})
        return cells

    for row in meaningful_rows:
        if len(row) == 1:
            continue
        field = row[0]
        value = row[1]
        if field or value:
            cells.append({"field": field, "value": value})
    return cells


def _normalize_cell(cell: str) -> str:
    return " ".join(cell.split()).strip()


def _pad_row(row: list[str], length: int) -> list[str]:
    values = [cell.strip() for cell in row[:length]]
    if len(values) < length:
        values.extend([""] * (length - len(values)))
    return values


def _alphabetic_ratio(row: list[str]) -> float:
    if not row:
        return 0.0
    alphabetic_cells = sum(1 for cell in row if any(ch.isalpha() for ch in cell))
    return alphabetic_cells / len(row)


def _numeric_ratio(row: list[str]) -> float:
    if not row:
        return 0.0
    numeric_cells = sum(1 for cell in row if any(ch.isdigit() for ch in cell))
    return numeric_cells / len(row)


def _is_header_row(row: list[str]) -> bool:
    if len(row) != 2:
        return False
    lowered = [cell.lower() for cell in row]
    return lowered in (
        ["field", "value"],
        ["name", "value"],
        ["label", "value"],
        ["key", "value"],
    )


def _normalize_key(value: str) -> str:
    normalized = value.lower().replace("benefit", "benefits")
    normalized = normalized.replace("eligibillity", "eligibility")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return normalized.strip()
