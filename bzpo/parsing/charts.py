"""Подписи диаграмм think-cell по XML — кодом, без модели.

think-cell хранит в данных диаграммы только числа: названия рядов и
категорий — отдельные надписи на слайде. Мультимодальная модель путает,
какая подпись легенды к каким данным относится (слайд 16 ДЕВ: «Чистый долг /
EBITDA» приписан ряду EBITDA), а XML даёт всё нужное точно.

Легенда — маленькая фигура-маркер с цветом ряда (или сектора) и надпись справа
в той же строке. Поэтому:

- ряды: цвет ряда в `c:ser/c:spPr` -> маркер того же цвета на слайде -> подпись;
  ряд цветом темы (обычно линия) получает подпись из той же строки легенды,
  не занятую другими рядами;
- круговая диаграмма (один ряд, свой цвет у каждого сектора в `c:dPt`) —
  так же, но по секторам: это её категории;
- категории остальных диаграмм — надписи нижней полосы или левой колонки
  области диаграммы, если их ровно столько, сколько значений.

Что не подтвердилось — None (остаётся «ряд N» / номер категории). Подписи
ставятся частично: подписанные ряды без категорий лучше, чем ничего.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from bzpo.parsing.heatmap import THEME, shape_fill

_MARKER_MAX = 400_000  # маркер легенды — не больше ~1 см
_ROW = 120_000  # подпись и маркер в одной строке легенды — по вертикали ближе ~0,3 см
_LABEL_GAP = 1_500_000  # подпись не дальше ~4 см вправо от маркера
_BAND = 60_000  # подписи категорий — в одной полосе


@dataclass(frozen=True, slots=True)
class ChartLabels:
    categories: list[str] | None
    series: list[str] | None


def labels(chart_shape: Any, shapes: list[Any], text_of: Any) -> ChartLabels:
    texts = [s for s in shapes if s is not chart_shape and _placed(s) and text_of(s)]
    markers = [s for s in shapes if s is not chart_shape and _placed(s) and not text_of(s)
               and (s.width or 0) <= _MARKER_MAX and (s.height or 0) <= _MARKER_MAX]
    plots = list(chart_shape.chart.plots)
    all_series = [series for plot in plots for series in plot.series]
    count = max((len(list(s.values)) for s in all_series), default=0)

    # Одна палитра на несколько диаграмм слайда — у каждого цвета несколько
    # маркеров; легенда диаграммы — ближайшая к ней.
    markers.sort(key=lambda m: _distance(m, chart_shape))
    if len(all_series) == 1 and len(_point_colors(all_series[0])) == count > 1:
        # Круговая и подобные: категории — сектора со своими цветами.
        categories = _by_color(_point_colors(all_series[0]), markers, texts, text_of)
        return ChartLabels(categories, None)

    series = _by_color([_series_color(s) for s in all_series], markers, texts, text_of)
    inside = [t for t in texts if _inside(t, chart_shape)]
    return ChartLabels(_categories(inside, count, text_of), series)


def _distance(shape: Any, box: Any) -> float:
    """Расстояние от фигуры до прямоугольника диаграммы (0 — внутри)."""
    dx = max(box.left - shape.left, 0, shape.left - (box.left + box.width))
    dy = max(box.top - shape.top, 0, shape.top - (box.top + box.height))
    return (dx * dx + dy * dy) ** 0.5


def _placed(shape: Any) -> bool:
    return shape.left is not None and shape.top is not None


def _inside(shape: Any, box: Any) -> bool:
    return (box.left <= shape.left <= box.left + box.width
            and box.top <= shape.top <= box.top + box.height)


def _series_color(series: Any) -> str:
    xml = series._element.xml
    xml = re.sub(r"<c:dPt>.*?</c:dPt>", "", xml, flags=re.S)
    sp = re.search(r"<c:spPr>(.*?)</c:spPr>", xml, re.S)
    colors = re.findall(r'srgbClr val="([0-9A-Fa-f]{6})"', sp.group(1)) if sp else []
    return colors[0].upper() if colors else THEME


def _point_colors(series: Any) -> list[str]:
    points = re.findall(r"<c:dPt>.*?</c:dPt>", series._element.xml, re.S)
    colors = []
    for point in points:
        found = re.findall(r'srgbClr val="([0-9A-Fa-f]{6})"', point)
        colors.append(found[0].upper() if found else THEME)
    return colors


def _label_for(marker: Any, texts: list[Any]) -> Any | None:
    """Ближайшая надпись справа от маркера в той же строке."""
    center = marker.top + (marker.height or 0) / 2
    row = [t for t in texts if 0 < t.left - marker.left <= _LABEL_GAP
           and abs(t.top + (t.height or 0) / 2 - center) <= _ROW]
    return min(row, key=lambda t: t.left - marker.left) if row else None


def _by_color(colors: list[str], markers: list[Any], texts: list[Any],
              text_of: Any) -> list[str] | None:
    """Подписи по цветам; один цвет темы — по остатку строк той же легенды."""
    if not colors:
        return None
    column = _column_legend(colors, markers, texts, text_of)
    if column is not None:
        return column
    nearest: dict[str, Any] = {}  # маркеры уже отсортированы по близости к диаграмме
    for marker in markers:
        fill = shape_fill(marker)
        if fill and fill != THEME and _label_for(marker, texts) is not None:
            nearest.setdefault(fill.upper(), marker)
    names: list[Any | None] = []
    for color in colors:
        marker = nearest.get(color) if color != THEME else None
        names.append(_label_for(marker, texts) if marker is not None else None)
    chosen = [n for n in names if n is not None]
    if len({id(n) for n in chosen}) != len(chosen):
        return None  # одна подпись досталась двум рядам — сопоставление ненадёжно
    missing = [i for i, n in enumerate(names) if n is None]
    if missing:
        if len(missing) > 1 or colors[missing[0]] != THEME or not chosen:
            return None
        # Остаток строк легенды: надписи в строках уже найденных подписей.
        rest = [t for t in texts if all(t is not c for c in chosen)
                and any(abs(t.top - c.top) <= _ROW for c in chosen)
                and min(abs(t.left - c.left) for c in chosen) <= 3 * _LABEL_GAP]
        if len(rest) != 1:
            return None
        names[missing[0]] = rest[0]
    return [text_of(n) for n in names]


def _column_legend(colors: list[str], markers: list[Any], texts: list[Any],
                   text_of: Any) -> list[str] | None:
    """Легенда одной надписью: абзац на пункт, слева столбик маркеров.

    Абзацы сопоставляются с рядами (секторами) по порядку, но только если
    цвета маркеров сверху вниз идут в том же порядке, что цвета в данных
    (допускается один маркер без явного цвета). Ближайшая к диаграмме легенда
    проверяется первой: маркеры уже отсортированы по расстоянию.
    """
    if len(colors) < 2:
        return None
    candidates = []
    for text in texts:
        paragraphs = [p.lstrip("- ").strip() for p in text_of(text).split("\n") if p.strip()]
        if len(paragraphs) != len(colors):
            continue
        beside = [m for m in markers if 0 < text.left - m.left <= _LABEL_GAP
                  and text.top - _ROW <= m.top <= text.top + (text.height or 0) + _ROW]
        if not beside or len(beside) < len(colors) - 1:
            continue
        sequence = [fill.upper() for m in sorted(beside, key=lambda m: m.top)
                    if (fill := shape_fill(m)) and fill != THEME]
        if len(sequence) < len(colors) - 1:
            continue
        distance = min(markers.index(m) for m in beside)
        if _in_order(sequence, colors):
            candidates.append((distance, paragraphs))
        elif _in_order(sequence, colors[::-1]):
            # Накопительные столбцы: легенда сверху вниз, ряды в данных — снизу вверх.
            candidates.append((distance, paragraphs[::-1]))
    if not candidates:
        return None
    return min(candidates, key=lambda c: c[0])[1]


def _in_order(sequence: list[str], colors: list[str]) -> bool:
    """sequence — подпоследовательность colors с сохранением порядка."""
    position = 0
    for color in sequence:
        try:
            position = colors.index(color, position) + 1
        except ValueError:
            return False
    return True


def _categories(texts: list[Any], count: int, text_of: Any) -> list[str] | None:
    if count == 0 or not texts:
        return None
    lowest = max(t.top for t in texts)
    bottom = sorted((t for t in texts if lowest - t.top <= _BAND), key=lambda t: t.left)
    if len(bottom) == count:
        return [text_of(t) for t in bottom]
    leftmost = min(t.left for t in texts)
    column = sorted((t for t in texts if t.left - leftmost <= _BAND), key=lambda t: t.top)
    if len(column) == count:
        return [text_of(t) for t in column]
    return None
