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

from bzpo.parsing import charts, heatmap
from bzpo.parsing.common import cell_text, clean, rows_to_markdown, yaml_string

CONVERTER_VERSION = "pptx-xml-1"
HEATMAP_NOTE = ("*Значения ячеек восстановлены по цвету заливки и легенде слайда "
                "(кодом, по XML).*")
LEGEND_CHART_NOTE = ("*Подписи рядов и категорий сопоставлены с легендой слайда по цвету "
                     "(кодом, по XML).*")
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
    # Заполняет проход модели (parsing/vlm.py); интерпретация, не цитируется.
    key_message: str | None = None
    visual: str | None = None
    vlm_done: bool = False

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
        title_element = title_shape._element if title_shape is not None else None
        shapes = [s for s in _flatten(slide.shapes) if s.raw._element is not title_element]
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
            # Убирается только сам колонтитул; другие надписи над заголовком
            # (гриф, дата, цитата) остаются в тексте слайда.
            shapes = [s for s in shapes
                      if not (s in labels and _text_frame(s.text_frame) == label)]
            if section is None or label.lower() not in section.lower():
                subsection = label
        data = SlideData(number, slide._element.get("show") == "0", title, section, subsection)
        seen: set[str] = {title}
        legend = heatmap.legend(
            slide.shapes, lambda s: _text_frame(s.text_frame) if _has_text(s) else ""
        )
        for shape in _ordered(shapes):
            _render_shape(shape, data, seen, legend, shapes)
        if slide.has_notes_slide:
            data.notes = clean(slide.notes_slide.notes_text_frame.text) or None
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
        f'doc_title: "{yaml_string(deck.title)}"',
        'doc_type: "Презентация"',
        f'source_file: "{yaml_string(source_name or deck.source.name)}"',
        "source_format: pptx",
        f"converter_version: {CONVERTER_VERSION}",
        f"converted_at: {date.today().isoformat()}",
        f"slides_total: {len(deck.slides)}",
        f'speaker_notes: "{"есть" if notes else "в исходном файле отсутствуют"}"',
        f"needs_vlm: {sum(s.needs_vlm and not s.vlm_done for s in deck.slides)}",
        f"vlm_slides: {sum(s.vlm_done for s in deck.slides)}",
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
        if s.key_message:
            body.append(f"**Ключевой посыл:** {s.key_message}")
        if s.visual:
            body.append(f"**Визуал:** {s.visual}")
        if s.vlm_done:
            provenance = "xml; изображение разобрано моделью"
        elif s.needs_vlm:
            provenance = "xml; нужен разбор изображения"
        else:
            provenance = "xml"
        body.append(f"**Источник значений:** {provenance}")
        body.append("")
        body += [block for b in s.blocks for block in (b, "")]
        if s.notes:
            body += [f"> **Заметки докладчика:** {s.notes}", ""]
        body += ["---", ""]
    return "\n".join(front + body)


# --------------------------------------------------------------- фигуры


class _Placed:
    """Фигура с координатами на слайде.

    python-pptx отдаёт координаты фигур внутри группы в системе группы
    (`a:chOff`/`a:chExt`), а не слайда: три легенды на слайде 41 ДЕВ «стоят» в
    одной точке. Здесь координаты пересчитаны в систему слайда с учётом
    вложенных групп; остальное делегируется исходной фигуре.
    """

    __slots__ = ("raw", "left", "top", "width", "height")

    def __init__(self, raw: Any, left: int, top: int, width: int, height: int) -> None:
        self.raw, self.left, self.top, self.width, self.height = raw, left, top, width, height

    def __getattr__(self, name: str) -> Any:
        return getattr(self.raw, name)


_Transform = tuple[float, float, float, float, float, float]  # off x/y, масштаб x/y, chOff x/y
_IDENTITY: _Transform = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)


def _flatten(shapes: Any, transform: _Transform = _IDENTITY) -> list[Any]:
    """Все фигуры слайда в координатах слайда; группы разворачиваются рекурсивно."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    ox, oy, sx, sy, cx, cy = transform

    def place(x: int | None, y: int | None, w: int | None, h: int | None) -> tuple[int, ...]:
        return (round(ox + ((x or 0) - cx) * sx), round(oy + ((y or 0) - cy) * sy),
                round((w or 0) * sx), round((h or 0) * sy))

    result: list[Any] = []
    for shape in shapes:
        if _SKIP_SHAPE.search(shape.name or ""):
            continue
        left, top, width, height = place(shape.left, shape.top, shape.width, shape.height)
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            child = _group_child_space(shape)
            if child is None:
                result += _flatten(shape.shapes, transform)
                continue
            ch_x, ch_y, ch_w, ch_h = child
            inner = (left, top, width / ch_w if ch_w else 1.0, height / ch_h if ch_h else 1.0,
                     ch_x, ch_y)
            result += _flatten(shape.shapes, inner)
        else:
            result.append(_Placed(shape, left, top, width, height))
    return result


def _group_child_space(group: Any) -> tuple[int, int, int, int] | None:
    """chOff и chExt группы — система координат её детей."""
    off = group._element.xpath("./p:grpSpPr/a:xfrm/a:chOff")
    ext = group._element.xpath("./p:grpSpPr/a:xfrm/a:chExt")
    if not off or not ext:
        return None
    return (int(off[0].get("x", 0)), int(off[0].get("y", 0)),
            int(ext[0].get("cx", 0)), int(ext[0].get("cy", 0)))


def _ordered(shapes: list[Any]) -> list[Any]:
    # Полоса в ~0,5 см по вертикали считается одной строкой: подписи, стоящие
    # рядом, читаются слева направо, а не по миллиметрам высоты.
    band = 180_000
    return sorted(shapes, key=lambda s: ((s.top or 0) // band, s.left or 0))


def _render_shape(
    shape: Any, data: SlideData, seen: set[str], legend: list[heatmap.LegendItem],
    shapes: list[Any],
) -> None:
    if getattr(shape, "has_table", False) and shape.has_table:
        data.tables += 1
        data.blocks.append(_table(shape.table, legend))
        return
    if getattr(shape, "has_chart", False) and shape.has_chart:
        data.charts += 1
        names = charts.labels(shape, shapes,
                              lambda s: _text_frame(s.text_frame) if _has_text(s) else "")
        data.blocks.append(_chart(shape.chart, names))
        return
    if _is_picture(shape):
        alt = (shape._element.xpath(".//p:cNvPr/@descr") or [""])[0].strip()
        label = alt or shape.name or "без подписи"
        data.images.append(label)
        data.blocks.append(f"*[Изображение: {clean(label.replace(chr(10), ' '))} — "
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
        text = clean("".join(parts))
        if not text:
            continue
        level = paragraph.level
        lines.append(("  " * (level - 1) + "- " + text) if level > 0 else text)
    return "\n".join(lines)


def _table(table: Any, legend: list[heatmap.LegendItem]) -> str:
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if cell.is_spanned:
                cells.append("")
                continue
            text = cell_text(cell.text)
            if not text:
                color = heatmap.shape_fill(cell)
                text = "" if color is None else (
                    f"■ {color}" if color == heatmap.THEME else f"■ #{color}")
            cells.append(text)
        rows.append(cells)
    mapped = heatmap.map_cells(rows, legend) if legend else None
    if mapped is not None:
        return HEATMAP_NOTE + "\n" + rows_to_markdown(mapped)
    # Цвет темы без легенды ничего не значит — обычная пустая ячейка.
    rows = [[c if c != f"■ {heatmap.THEME}" else "" for c in row] for row in rows]
    return rows_to_markdown(rows)


def _is_picture(shape: Any) -> bool:
    """Картинка, в том числе связанная (не встроенная) и в заполнителе-рисунке.

    `shape.image` у связанной картинки бросает ValueError, а не AttributeError,
    поэтому проверка идёт по типу фигуры и по наличию blip, без чтения image.
    """
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        return True
    return bool(shape._element.xpath("./p:blipFill"))


def _chart(chart: Any, names: charts.ChartLabels | None = None) -> str:
    title = ""
    if chart.has_title and chart.chart_title.has_text_frame:
        title = clean(chart.chart_title.text_frame.text)
    categories: list[str] = []
    columns: list[tuple[str, list[Any]]] = []
    for plot in chart.plots:
        if not categories:
            # Пустая категория приходит как None — это не подпись «None».
            categories = [clean(str(c)) if c not in (None, "") else "" for c in plot.categories]
        for series in plot.series:
            columns.append((clean(series.name or ""), list(series.values)))
    if not columns:
        return ""
    from_legend = False
    # think-cell: подписи — отдельные надписи на слайде, сопоставленные по цвету (charts.py).
    if names is not None and names.series and len(names.series) == len(columns) \
            and not any(name for name, _ in columns):
        columns = [(cell_text(n), v) for n, (_, v) in zip(names.series, columns, strict=True)]
        from_legend = True
    size = max([len(categories)] + [len(v) for _, v in columns])
    if names is not None and names.categories and len(names.categories) == size \
            and not any(categories):
        categories = [cell_text(c) for c in names.categories]
        from_legend = True
    columns = [(name or f"ряд {i + 1}", v) for i, (name, v) in enumerate(columns)]
    unnamed = not any(categories) or any(n.startswith("ряд ") for n, _ in columns)
    categories += [""] * (size - len(categories))
    head = ["Категория"] + [name for name, _ in columns]
    lines = [f"**Диаграмма{': ' + title if title else ''}** (значения из данных диаграммы)",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    notes = []
    if from_legend:
        notes.append(LEGEND_CHART_NOTE)
    if unnamed:
        notes.append(UNNAMED_CHART_NOTE)
    lines[1:1] = notes
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

