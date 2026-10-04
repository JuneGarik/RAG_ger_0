"""Единая точка входа конвертации: файл любого поддерживаемого формата -> md.

Пользователь не выбирает способ разбора: формат определяется по содержимому
(`detect`), дальше — свой извлекатель и, где есть исходный XML, проверка
результата против него.

    pptx         -> parsing.pptx  + проверка md ↔ XML
    docx         -> parsing.docx  (рисунки — на разбор, П2)
    md (слайды)  -> проверка разметки, копия как есть
    md (документ)-> копия как есть
    pdf, ppt/doc -> отказ с объяснением
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bzpo.parsing.check import DeckCheck, check
from bzpo.parsing.detect import Detection, detect

SOURCE_SUFFIXES = (".pptx", ".docx", ".md", ".pdf", ".ppt", ".doc")


@dataclass(slots=True)
class Conversion:
    source: Path
    detection: Detection
    markdown: str | None
    needs_vlm: int = 0
    check: DeckCheck | None = None
    error: str | None = None
    notes: str = ""

    @property
    def ok(self) -> bool:
        return self.markdown is not None and self.error is None and (
            self.check is None or self.check.passed
        )


def convert(path: Path) -> Conversion:
    detection = detect(path)
    if not detection.supported:
        return Conversion(path, detection, None, error=detection.method)

    if detection.format == "pptx":
        from bzpo.parsing import pptx

        deck = pptx.extract(path)
        markdown = pptx.to_markdown(deck)
        return Conversion(path, detection, markdown,
                          needs_vlm=sum(s.needs_vlm for s in deck.slides),
                          check=check(path, markdown),
                          notes=f"{len(deck.slides)} слайдов")

    if detection.format == "docx":
        from bzpo.parsing import docx

        document = docx.extract(path)
        screenshots = sum(f.slide_screenshot for f in document.figures)
        return Conversion(path, detection, docx.to_markdown(document),
                          needs_vlm=document.needs_vlm,
                          notes=f"{document.headings} заголовков, {len(document.figures)} "
                                f"рисунков, из них похожих на скриншоты слайдов: {screenshots}")

    markdown = path.read_text(encoding="utf-8")
    if detection.format == "md_slides":
        from bzpo.ingest.slides import SlideMarkdownError, chunk_slides

        try:
            deck = chunk_slides(markdown, lambda text: 0, hard_limit=1)
        except SlideMarkdownError as exc:
            return Conversion(path, detection, None, error=f"разметка слайдов: {exc}")
        return Conversion(path, detection, markdown, notes=f"{len(deck.chunks)} слайдов")
    return Conversion(path, detection, markdown, notes="документ без разметки слайдов")


def sources(directory: Path) -> list[Path]:
    """Исходные файлы папки (без временных файлов Office «~$…»)."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES and not p.name.startswith("~$")
    )
