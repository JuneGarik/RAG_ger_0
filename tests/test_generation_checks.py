"""Линеаризация широких таблиц и проверка чисел в ответе против фрагментов."""

from __future__ import annotations

from bzpo.generation.tables import linearize_wide_tables
from bzpo.generation.validate import check_numbers

WIDE = """Тепловая карта
| Критерий | Ф1 | Ф2 | Ф3 | Ф4 | Ф5 |
|---|---|---|---|---|---|
| **Видение** | ≤1,7 | 1,8–1,9 | н/д | 2,0–2,1 | ≥2,4 |
| Данные | 2,2–2,3 | | ≤1,7 | ≤1,7 | 2,0–2,1 |
**Источник:** анализ"""


def test_wide_table_cells_next_to_column_names() -> None:
    lines = linearize_wide_tables(WIDE).split("\n")
    assert lines[0] == "Тепловая карта"
    assert lines[1] == "- **Видение**: Ф1 — ≤1,7; Ф2 — 1,8–1,9; Ф3 — н/д; Ф4 — 2,0–2,1; Ф5 — ≥2,4"
    # Пустая ячейка пропускается, а не сдвигает соседей.
    assert lines[2] == "- Данные: Ф1 — 2,2–2,3; Ф3 — ≤1,7; Ф4 — ≤1,7; Ф5 — 2,0–2,1"
    assert lines[3] == "**Источник:** анализ"


def test_narrow_table_untouched() -> None:
    narrow = "| Год | Выручка |\n|---|---|\n| 2024 | 128 650 |"
    assert linearize_wide_tables(narrow) == narrow


FRAGMENTS = {
    1: "Сводный балл 2,1 из 5. Выручка 2024 — 128 650 млн руб., падение 12,3%.",
    2: "Чистый долг / EBITDA — 2,73 на 31.12.2024.",
}


def test_supported_numbers_pass() -> None:
    answer = "Сводный балл — 2,1 из 5 [1]. Выручка — 128 650 млн руб. [1]. Долг — 2,73 [2]."
    assert check_numbers(answer, FRAGMENTS) == []


def test_invented_number_caught() -> None:
    [issue] = check_numbers("Адаптация сократилась до 95% [1].", FRAGMENTS)
    assert issue.number == "95"
    assert issue.cited == (1,)


def test_number_from_wrong_fragment_caught() -> None:
    [issue] = check_numbers("Долговая нагрузка — 2,73 [1].", FRAGMENTS)
    assert issue.number == "2.73"
    assert "[1]" in issue.reason


def test_number_without_marker_caught() -> None:
    [issue] = check_numbers("Выручка составила 128 650 млн руб.", FRAGMENTS)
    assert issue.cited == ()


def test_formats_normalized_and_small_counts_ignored() -> None:
    # 128650 без пробела, 2.10 вместо 2,1; «3 направления» — счётное до 10.
    answer = "Выручка 128650 млн руб. и балл 2.10 [1]. Выделено 3 направления [1]."
    assert check_numbers(answer, FRAGMENTS) == []


HEATMAP = {1: "- Видение: Ф1 — ≤1,7; Ф2 — 2,2–2,3; Ф3 — ≥2,4"}


def test_ranges_and_bounds_checked_whole() -> None:
    assert check_numbers("Ф2 — 2,2–2,3, Ф3 — ≥2,4, Ф1 — ≤1,7 [1].", HEATMAP) == []
    # Диапазон склеен из чисел, которые во фрагменте есть по отдельности.
    [issue] = check_numbers("Ф3 оценивается в 2,3–2,4 [1].", HEATMAP)
    assert issue.number == "2.3–2.4"


def test_multiple_markers_union() -> None:
    assert check_numbers("Балл 2,1 при долге 2,73 [1][2].", FRAGMENTS) == []
