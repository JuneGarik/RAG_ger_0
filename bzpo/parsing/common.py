"""Общее для извлекателей pptx и docx: таблица в md, очистка текста, YAML-строка."""

from __future__ import annotations

import re


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Сетка ячеек -> md-таблица; короткие строки дополняются пустыми ячейками."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def cell_text(text: str) -> str:
    """Текст ячейки таблицы: одна строка, без символа-разделителя md."""
    return clean(text).replace("|", "/").replace("\n", " ")


def clean(text: str) -> str:
    """Неразрывные пробелы, табуляции и вертикальная табуляция PowerPoint -> пробел."""
    text = text.replace("\u000b", " ").replace(" ", " ").replace("\t", " ")
    return re.sub(r"[ ]+", " ", text).strip()


def yaml_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')
