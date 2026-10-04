"""Подача широких md-таблиц генератору (этап 4, сборка контекста).

Плотная 9B-модель путает значения в таблице на 12+ колонок: чтобы прочитать
ячейку, ей нужно отсчитать N-й разделитель «|» в строке и сопоставить его с
N-м заголовком, и она сбивается на соседнюю колонку. Линеаризация убирает
отсчёт: каждая ячейка идёт рядом со своим заголовком колонки

    | Критерий | Покупка ЗУ | Проектирование |       - Видение и стратегия: Покупка ЗУ — ≤1,7;
    |---|---|---|                         ->         Проектирование — ≤1,7
    | Видение и стратегия | ≤1,7 | ≤1,7 |

Применяется только к тексту, который уходит в промпт; в базе, поиске и
реранкинге таблица остаётся как есть. Узкие таблицы не трогаются: там модель
не ошибается, а линеаризация удлиняет промпт (заголовки повторяются в каждой
строке).
"""

from __future__ import annotations

import re

_ROW = re.compile(r"^\|.*\|[ \t]*$")
_SEPARATOR = re.compile(r"^\|(?:[ \t]*:?-{3,}:?[ \t]*\|)+[ \t]*$")


def linearize_wide_tables(text: str, min_columns: int = 6) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if (
            i + 1 < len(lines)
            and _ROW.match(lines[i])
            and _SEPARATOR.match(lines[i + 1])
            and len(_cells(lines[i])) >= min_columns
        ):
            header = _cells(lines[i])
            i += 2
            while i < len(lines) and _ROW.match(lines[i]):
                out.append(_linearize_row(header, _cells(lines[i])))
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _linearize_row(header: list[str], row: list[str]) -> str:
    pairs = [
        f"{name} — {value}"
        for name, value in zip(header[1:], row[1:], strict=False)
        if value
    ]
    return f"- {row[0]}: " + "; ".join(pairs)
