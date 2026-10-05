"""Шкалы-оценки из закрашенных сегментов поверх таблицы — кодом, по XML.

Оценка «6 из 8» в отчётах часто нарисована не текстом, а рядом одинаковых
фигур в ячейке таблицы: закрашенные сегменты — балл, светло-серые — пустые
(слайд 6 «ОЦО Ритейл»: 8 критериев × 2 решения × 8 сегментов). Фигуры лежат
поверх таблицы, ячейки при этом пустые, а модель такие шкалы не считает.

Сегменты привязываются к ячейке по геометрии таблицы (центр фигуры внутри
ячейки). Класс сегмента — его заливка целиком (цвет темы с модификаторами или
RGB). Пустым считается класс фона: `bg1`/`lt1`/`bg2`/`lt2` или почти белый
RGB. Если закрашенных классов два (на слайде — бирюзовый и голубой), раскладка
показывается, но не трактуется: что значит второй цвет, на слайде не сказано.
"""

from __future__ import annotations

import re
from typing import Any

_SEGMENT_MAX = 400_000  # сегмент шкалы — не больше ~1 см
_MIN_SEGMENTS = 3
_BACKGROUND_SCHEMES = {"bg1", "bg2", "lt1", "lt2"}


def cell_scales(table_shape: Any, shapes: list[Any]) -> dict[tuple[int, int], str]:
    """{(строка, колонка): «6 из 8 (из них 3 — светлым цветом)»} для ячеек со шкалой."""
    table = table_shape.table
    xs = [table_shape.left]
    for column in table.columns:
        xs.append(xs[-1] + column.width)
    ys = [table_shape.top]
    for row in table.rows:
        ys.append(ys[-1] + row.height)

    cells: dict[tuple[int, int], list[Any]] = {}
    for shape in shapes:
        if shape is table_shape or not _segment(shape):
            continue
        cx = shape.left + shape.width / 2
        cy = shape.top + shape.height / 2
        row = next((i for i in range(len(ys) - 1) if ys[i] <= cy < ys[i + 1]), None)
        col = next((j for j in range(len(xs) - 1) if xs[j] <= cx < xs[j + 1]), None)
        if row is not None and col is not None:
            cells.setdefault((row, col), []).append(shape)

    result = {}
    for key, segments in cells.items():
        sizes = {(s.width, s.height) for s in segments}
        if len(segments) < _MIN_SEGMENTS or len(sizes) != 1:
            continue
        kinds = [_fill_key(s) for s in sorted(segments, key=lambda s: s.left)]
        filled = [k for k in kinds if not _is_background(k)]
        if len(filled) == len(kinds):
            continue  # нет пустых сегментов — не шкала, а, например, ряд иконок
        text = f"{len(filled)} из {len(kinds)}"
        classes = list(dict.fromkeys(filled))  # в порядке появления слева направо
        if len(classes) > 1:
            extra = sum(1 for k in filled if k != classes[0])
            text += f" (из них {extra} — светлым цветом)"
        result[key] = text
    return result


def _segment(shape: Any) -> bool:
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame \
            and shape.text_frame.text.strip():
        return False
    if getattr(shape, "has_table", False) or getattr(shape, "has_chart", False):
        return False
    width, height = shape.width or 0, shape.height or 0
    return 0 < width <= _SEGMENT_MAX and 0 < height <= _SEGMENT_MAX and _fill_key(shape) != ""


def _fill_key(shape: Any) -> str:
    """Заливка фигуры как строка: цвет темы с модификаторами или RGB; «» — без заливки."""
    fills = shape._element.xpath("./p:spPr/a:solidFill")
    if not fills:
        return ""
    xml = re.sub(r'\s*xmlns(:\w+)?="[^"]+"', "", fills[0].xml)
    return re.sub(r"\s+", " ", xml)


def _is_background(key: str) -> bool:
    scheme = re.search(r'schemeClr val="(\w+)"', key)
    if scheme:
        return scheme.group(1) in _BACKGROUND_SCHEMES
    rgb = re.search(r'srgbClr val="([0-9A-Fa-f]{6})"', key)
    if not rgb:
        return False
    r, g, b = (int(rgb.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    return min(r, g, b) >= 225 and max(r, g, b) - min(r, g, b) <= 20
