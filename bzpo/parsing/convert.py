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

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    vlm_issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.markdown is not None and self.error is None and (
            self.check is None or self.check.passed
        )


def convert(
    path: Path,
    *,
    vlm: bool = False,
    cache_dir: Path = Path("data/cache"),
    progress: Callable[[int, int, int], None] = lambda done, total, slide: None,
    only_slides: set[int] | None = None,
) -> Conversion:
    """Файл -> md. `vlm=True` — с разбором изображений моделью (pptx, ~25 с на слайд).

    `progress(сделано, всего, номер слайда)` вызывается после каждого слайда;
    `only_slides` — разобрать только эти слайды (для пробы и отладки).
    """
    detection = detect(path)
    if not detection.supported:
        return Conversion(path, detection, None, error=detection.method)

    if detection.format == "pptx":
        from bzpo.parsing import pptx

        deck = pptx.extract(path)
        notes = f"{len(deck.slides)} слайдов"
        issues: list[str] = []
        if vlm:
            done, issues = _vision_pass(deck, path, cache_dir, progress, only_slides)
            notes += f"; разобрано моделью: {done}"
        markdown = pptx.to_markdown(deck)
        return Conversion(path, detection, markdown,
                          needs_vlm=sum(s.needs_vlm and not s.vlm_done for s in deck.slides),
                          check=check(path, markdown), notes=notes, vlm_issues=issues)

    if detection.format == "docx":
        from bzpo.parsing import docx

        document = docx.extract(path)
        issues = []
        if vlm:
            issues = _ocr_pass(document, cache_dir, progress)
        screenshots = sum(f.slide_screenshot for f in document.figures)
        read = sum(1 for f in document.figures if f.text)
        return Conversion(path, detection, docx.to_markdown(document),
                          needs_vlm=document.needs_vlm,
                          notes=f"{document.headings} заголовков, {len(document.figures)} "
                                f"рисунков, из них похожих на скриншоты слайдов: {screenshots}; "
                                f"текст распознан на {read}",
                          vlm_issues=issues)

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


def _vision_pass(
    deck: Any, path: Path, cache_dir: Path,
    progress: Callable[[int, int, int], None], only_slides: set[int] | None,
) -> tuple[int, list[str]]:
    """Рендер + модель по слайдам, которым нужен разбор; вернуть число и замечания."""
    from bzpo.config import load_config
    from bzpo.parsing import vlm
    from bzpo.parsing.render import render_pages
    from bzpo.providers.ollama import OllamaClient, OllamaError

    pages = render_pages(path, cache_dir / "render")
    if len(pages) != len(deck.slides):
        # Скрытые слайды LibreOffice может не выводить — сопоставление съедет.
        return 0, [f"страниц в рендере {len(pages)}, слайдов {len(deck.slides)}: "
                   "разбор изображений пропущен"]
    targets = [s for s in deck.slides if s.needs_vlm
               and (only_slides is None or s.number in only_slides)]
    settings = load_config()
    issues: list[str] = []
    with OllamaClient(settings.ollama, role="vision") as client:
        for done, slide in enumerate(targets, start=1):
            try:
                result = vlm.analyze(slide, pages[slide.number - 1], client,
                                     cache_dir / "vlm", client.model)
            except OllamaError as exc:
                # Сбой на одном слайде не останавливает документ: слайд остаётся
                # «нужен разбор», остальные разбираются.
                issues.append(f"слайд {slide.number}: модель не ответила ({exc})")
                progress(done, len(targets), slide.number)
                continue
            vlm.apply(slide, result)
            slide.vlm_done = True
            issues += [f"слайд {slide.number}: {issue}" for issue in result.issues]
            progress(done, len(targets), slide.number)
    return len(targets), issues


def _ocr_pass(document: Any, cache_dir: Path,
              progress: Callable[[int, int, int], None]) -> list[str]:
    """Распознать текст на скриншотах слайдов и крупных рисунках документа Word."""
    from bzpo.config import load_config
    from bzpo.parsing import ocr
    from bzpo.providers.ollama import OllamaClient, OllamaError

    targets = [f for f in document.figures if f.worth_reading]
    issues: list[str] = []
    with OllamaClient(load_config().ollama, role="ocr") as client:
        for done, figure in enumerate(targets, start=1):
            try:
                figure.text = ocr.transcribe(figure.image, client, cache_dir / "ocr",
                                             client.model) or None
            except OllamaError as exc:
                issues.append(f"рисунок {figure.number}: модель не ответила ({exc})")
                progress(done, len(targets), figure.number)
                continue
            if figure.text is None:
                issues.append(f"рисунок {figure.number}: текста не найдено")
            progress(done, len(targets), figure.number)
    return issues
