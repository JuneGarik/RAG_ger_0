"""Детерминированное извлечение pptx -> md по контракту (docs/PARSING_PLAN.md, этап П1).

Всё, что лежит в файле, берётся кодом и точно: заголовки, тексты с уровнями
списков, таблицы, данные диаграмм из XML, ссылки, непустые заметки. Модель
здесь не участвует. Чего код не видит — смысла картинок, схем и тепловых карт —
то помечается для прохода мультимодальной модели (этап П2):

- картинка на слайде -> строка «[Изображение …]» и `needs_vlm` у слайда;
- ячейка таблицы без текста, но с заливкой -> «■ #RRGGBB»: значение тепловой
  карты передано цветом, сопоставление с легендой — задача П2.

Порядок чтения — сверху вниз, слева направо по координатам фигур; группы
обходятся рекурсивно. Объекты think-cell (OLE «think-cell data») пропускаются:
их диаграммы лежат рядом обычными диаграммами и подписями.

Формат вывода совпадает с тем, что читает `ingest/slides.py`; поля
«Тип слайда» определяются эвристикой, «Ключевой посыл» и «Визуал» не
заполняются — это интерпретация, её пишет П2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

CONVERTER_VERSION = "pptx-xml-1"
UNNAMED_CHART_NOTE = ("*Подписи рядов и категорий заданы на слайде отдельными надписями — "
                      "сопоставление по расположению (этап П2).*")

_SKIP_SHAPE = re.compile(r"think-cell", re.I)
_DIVIDER_NUMBER = re.compile(r"^\d{1,2}$")
# Надпись раздела над заголовком (колонтитул think-cell) короче этого — длиннее уже текст.
_SECTION_LABEL_MAX = 120
_LABEL_MIN_REPEATS = 3


@dataclass(slots=True)
class SlideData:
    number: int
    hidden: bool
    title: str
    section: str | None
    subsection: str | None = None
    blocks: list[str] = field(default_factory=list)  # готовые фрагменты md по порядку
    tables: int = 0
    charts: int = 0
    images: list[str] = field(default_factory=list)
    notes: str | None = None
    words: int = 0

    @property
    def slide_type(self) -> str:
        if self.section and self.section.endswith(self.title) and self.words < 40:
            return "разделитель раздела"
        kinds = []
        if self.charts:
            kinds.append("диаграмма")
        if self.tables:
            kinds.append("таблица")
        if self.images and self.words < 40:
            kinds.append("иллюстрация")
        return " + ".join(kinds) or "текст"

    @property
    def needs_vlm(self) -> bool:
        """Есть то, что код прочитать не может: картинки, значения цветом, подписи
        диаграммы отдельными надписями."""
        return bool(self.images) or any("■ #" in b or UNNAMED_CHART_NOTE in b
                                        for b in self.blocks)


@dataclass(slots=True)
class DeckData:
    source: Path
    title: str
    slides: list[SlideData]


def extract(path: Path) -> DeckData:
    from pptx import Presentation

    presentation = Presentation(str(path))
    # Проход 1: заголовок, фигуры и надпись над заголовком на каждом слайде.
    raw = []
    for number, slide in enumerate(presentation.slides, start=1):
        title_shape = slide.shapes.title
        title = _text_frame(title_shape.text_frame) if title_shape is not None else ""
        title = title.replace("\n", " ")
        shapes = [s for s in _flatten(slide.shapes) if s is not title_shape]
        if not title:
            title = next((_text_frame(s.text_frame).replace("\n", " ") for s in _ordered(shapes)
                          if _has_text(s)), f"Слайд {number}")
        texts = [s for s in shapes if _has_text(s)]
        labels = []
        if title_shape is not None and title_shape.top is not None:
            labels = [s for s in texts if s.top is not None and s.top < title_shape.top
                      and len(_text_frame(s.text_frame)) <= _SECTION_LABEL_MAX]
        label = _text_frame(min(labels, key=lambda s: s.top).text_frame) if labels else None
        divider = next((_text_frame(s.text_frame) for s in texts
                        if _DIVIDER_NUMBER.match(_text_frame(s.text_frame))), None)
        raw.append((number, slide, title, shapes, labels, label,
                    divider if divider and len(texts) <= 3 else None))

    # Колонтитул раздела повторяется по слайдам; разовая надпись над заголовком
    # (гриф, цитата, подпись) разделом не считается и остаётся в тексте.
    counts: dict[str, int] = {}
    for *_, label, _divider in raw:
        if label:
            counts[label] = counts.get(label, 0) + 1

    # Проход 2: раздел (со слайда-разделителя), подраздел (колонтитул), содержимое.
    slides: list[SlideData] = []
    section: str | None = None
    for number, slide, title, shapes, labels, label, divider in raw:
        if divider:
            section = f"{divider}. {title}"
        subsection = None
        if label and counts[label] >= _LABEL_MIN_REPEATS:
            shapes = [s for s in shapes if s not in labels]
            if section is None or label.lower() not in section.lower():
                subsection = label
        data = SlideData(number, slide._element.get("show") == "0", title, section, subsection)
        seen: set[str] = {title}
        for shape in _ordered(shapes):
            _render_shape(shape, data, seen)
        if slide.has_notes_slide:
            data.notes = _clean(slide.notes_slide.notes_text_frame.text) or None
        data.words = sum(len(b.split()) for b in data.blocks)
        slides.append(data)

    core_title = (presentation.core_properties.title or "").strip()
    generic = {"", "презентация powerpoint", "powerpoint presentation"}
    title = core_title if core_title.lower() not in generic else (
        slides[0].title if slides else path.stem)
    return DeckData(path, title, slides)


def to_markdown(deck: DeckData, *, source_name: str | None = None) -> str:
    notes = any(s.notes for s in deck.slides)
    front = [
        "---",
        f'doc_title: "{_yaml(deck.title)}"',
        'doc_type: "Презентация"',
        f'source_file: "{_yaml(source_name or deck.source.name)}"',
        "source_format: pptx",
        f"converter_version: {CONVERTER_VERSION}",
        f"converted_at: {date.today().isoformat()}",
        f"slides_total: {len(deck.slides)}",
        f'speaker_notes: "{"есть" if notes else "в исходном файле отсутствуют"}"',
        f"needs_vlm: {sum(s.needs_vlm for s in deck.slides)}",
        "---",
        "",
        f"# {deck.title}",
        "",
    ]
    body: list[str] = []
    for s in deck.slides:
        body += [f"<!-- slide: {s.number} -->", f"## [Слайд {s.number}] {s.title}"]
        body.append(f"**Раздел:** {s.section or '—'}")
        if s.subsection:
            body.append(f"**Подраздел:** {s.subsection}")
        body.append(f"**Тип слайда:** {s.slide_type}" + (" (скрытый)" if s.hidden else ""))
        body.append("**Источник значений:** xml" + ("; нужен разбор изображения" if s.needs_vlm
                                                    else ""))
        body.append("")
        body += [block for b in s.blocks for block in (b, "")]
        if s.notes:
            body += [f"> **Заметки докладчика:** {s.notes}", ""]
        body += ["---", ""]
    return "\n".join(front + body)


# --------------------------------------------------------------- фигуры


def _flatten(shapes: Any) -> list[Any]:
    """Все фигуры слайда; группы разворачиваются рекурсивно."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    result = []
    for shape in shapes:
        if _SKIP_SHAPE.search(shape.name or ""):
            continue
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            result += _flatten(shape.shapes)
        else:
            result.append(shape)
    return result


def _ordered(shapes: list[Any]) -> list[Any]:
    # Полоса в ~0,5 см по вертикали считается одной строкой: подписи, стоящие
    # рядом, читаются слева направо, а не по миллиметрам высоты.
    band = 180_000
    return sorted(shapes, key=lambda s: ((s.top or 0) // band, s.left or 0))


def _render_shape(shape: Any, data: SlideData, seen: set[str]) -> None:
    if getattr(shape, "has_table", False) and shape.has_table:
        data.tables += 1
        data.blocks.append(_table(shape.table))
        return
    if getattr(shape, "has_chart", False) and shape.has_chart:
        data.charts += 1
        data.blocks.append(_chart(shape.chart))
        return
    if hasattr(shape, "image"):
        alt = (shape._element.xpath(".//p:cNvPr/@descr") or [""])[0].strip()
        label = alt or shape.name or "без подписи"
        data.images.append(label)
        data.blocks.append(f"*[Изображение: {_clean(label.replace(chr(10), ' '))} — "
                           "содержимое не распознано]*")
        return
    if _has_text(shape):
        text = _text_frame(shape.text_frame)
        if text and text not in seen:
            seen.add(text)
            data.blocks.append(text)


def _has_text(shape: Any) -> bool:
    return bool(getattr(shape, "has_text_frame", False) and shape.has_text_frame
                and _text_frame(shape.text_frame))


def _text_frame(frame: Any) -> str:
    """Текст абзацев с уровнями списков.

    Читается из XML абзаца, а не через `paragraph.runs`: think-cell пишет
    подписи диаграмм полями (`a:fld`), которых в runs нет, — иначе годы и
    значения подписей молча пропадают.
    """
    from pptx.oxml.ns import qn

    run_tags, br_tag, t_tag = (qn("a:r"), qn("a:fld")), qn("a:br"), qn("a:t")
    lines = []
    for paragraph in frame.paragraphs:
        links = {id(r._r): r.hyperlink.address for r in paragraph.runs
                 if r.hyperlink is not None and r.hyperlink.address}
        parts = []
        for child in paragraph._p:
            if child.tag in run_tags:
                piece = "".join(t.text or "" for t in child.iter(t_tag))
                address = links.get(id(child))
                parts.append(f"[{piece}]({address})" if address and piece.strip() else piece)
            elif child.tag == br_tag:
                parts.append(" ")
        text = _clean("".join(parts))
        if not text:
            continue
        level = paragraph.level
        lines.append(("  " * (level - 1) + "- " + text) if level > 0 else text)
    return "\n".join(lines)


def _table(table: Any) -> str:
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if cell.is_spanned:
                cells.append("")
                continue
            text = _clean(cell.text).replace("|", "/").replace("\n", " ")
            if not text:
                color = _fill_color(cell)
                text = f"■ #{color}" if color else ""
            cells.append(text)
        rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _fill_color(cell: Any) -> str | None:
    """Сплошная заливка ячейки (RGB); цвет темы без RGB — None."""
    try:
        fill = cell.fill
        if fill.type == 1:  # MSO_FILL.SOLID
            return str(fill.fore_color.rgb)
    except (AttributeError, TypeError, ValueError):
        return None
    return None


def _chart(chart: Any) -> str:
    title = ""
    if chart.has_title and chart.chart_title.has_text_frame:
        title = _clean(chart.chart_title.text_frame.text)
    categories: list[str] = []
    columns: list[tuple[str, list[Any]]] = []
    for plot in chart.plots:
        if not categories:
            categories = [_clean(str(c)) for c in plot.categories]
        for series in plot.series:
            name = _clean(series.name or "") or f"ряд {len(columns) + 1}"
            columns.append((name, list(series.values)))
    if not columns:
        return ""
    unnamed = not categories or all(n.startswith("ряд ") for n, _ in columns)
    size = max([len(categories)] + [len(v) for _, v in columns])
    categories += [""] * (size - len(categories))
    head = ["Категория"] + [name for name, _ in columns]
    lines = [f"**Диаграмма{': ' + title if title else ''}** (значения из данных диаграммы)",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    if unnamed:
        # think-cell: подписи рядов и категорий — отдельные надписи на слайде.
        lines.insert(1, UNNAMED_CHART_NOTE)
    for i in range(size):
        row = [categories[i] or str(i + 1)]
        row += [_number(values[i]) if i < len(values) else "" for _, values in columns]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text.replace(".", ",")
    return str(value)


def _clean(text: str) -> str:
    text = text.replace("\u000b", " ").replace(" ", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _yaml(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')
