"""Определение формата файла по содержимому, а не по расширению.

pptx и docx — zip-архивы Office: различаются внутренней структурой
(`ppt/presentation.xml` или `word/document.xml`). md различается разметкой:
размеченная презентация (метки `<!-- slide: N -->`) или обычный документ.
Расширение используется только для сообщения, когда файл не распознан.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Format = Literal["pptx", "docx", "md_slides", "md_document", "pdf", "legacy_office", "unknown"]

# Как обрабатывается каждый формат — показывается пользователю.
METHODS: dict[Format, str] = {
    "pptx": "презентация: извлечение из XML (текст, таблицы, данные диаграмм) + проверка",
    "docx": "документ Word: заголовки, абзацы, таблицы из XML; рисунки — на разбор (П2)",
    "md_slides": "готовый md презентации: проверка разметки, индексируется как есть",
    "md_document": "готовый md документа: индексируется по разделам",
    "pdf": "pdf: пока не поддерживается (этап по SPEC — текст и OCR)",
    "legacy_office": "старый формат Office (ppt/doc): пересохраните в pptx/docx",
    "unknown": "формат не распознан",
}
SUPPORTED: tuple[Format, ...] = ("pptx", "docx", "md_slides", "md_document")


@dataclass(frozen=True, slots=True)
class Detection:
    format: Format
    reason: str

    @property
    def method(self) -> str:
        return METHODS[self.format]

    @property
    def supported(self) -> bool:
        return self.format in SUPPORTED


def detect(path: Path) -> Detection:
    with path.open("rb") as handle:
        head = handle.read(8)
    if head.startswith(b"PK"):
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile:
            return Detection("unknown", "повреждённый zip-архив")
        if "ppt/presentation.xml" in names:
            return Detection("pptx", "zip Office с ppt/presentation.xml")
        if "word/document.xml" in names:
            return Detection("docx", "zip Office с word/document.xml")
        return Detection("unknown", "zip-архив, но не pptx и не docx")
    if head.startswith(b"%PDF"):
        return Detection("pdf", "сигнатура %PDF")
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return Detection("legacy_office", "двоичный формат Office 97–2003")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Detection("unknown", f"двоичный файл {path.suffix or 'без расширения'}")
    from bzpo.ingest.slides import is_slide_markdown

    if is_slide_markdown(text):
        return Detection("md_slides", "текст с метками <!-- slide: N -->")
    return Detection("md_document", "текст без меток слайдов")
