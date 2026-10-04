"""Чанкер презентаций в md: один слайд — один чанк, предел — крупнейший слайд."""

from __future__ import annotations

import pytest

from bzpo.ingest.slides import SlideMarkdownError, chunk_slides, is_slide_markdown


def words(text: str) -> int:
    return len(text.split())


FRONTMATTER = """---
doc_title: "Диагностика цифровой зрелости"
doc_type: "Аналитический отчет (презентация)"
customer: "ООО «Заказчик»"
report_date: "2025-07-18"
slides_total: 3
---

# Диагностика цифровой зрелости
"""

TABLE_ROWS = "\n".join(f"| {year} | {year * 10} |" for year in range(2000, 2060))

SLIDES = f"""
<!-- slide: 1 -->
## [Слайд 1] Титул
**Раздел:** Титульный лист
**Тип слайда:** титул

Отчет для заказчика.

---

<!-- slide: 2 -->
## [Слайд 2] Выручка выросла на 12%
**Раздел:** 01. Финансы
**Подраздел:** 1.1. Выручка
**Тип слайда:** диаграмма
**Ключевой посыл:** рост обеспечен продажами
**Визуал:** столбцы по годам, бирюзовый — выручка

| Год | Выручка |
|---|---|
{TABLE_ROWS}

**Источник:** материалы клиента

---

<!-- slide: 3 -->
## [Слайд 3] Контакты
**Раздел:** Контакты
**Тип слайда:** контакты

Почта.
"""


def deck(text: str = FRONTMATTER + SLIDES, **kwargs: object):
    return chunk_slides(text, words, hard_limit=kwargs.pop("hard_limit", 8192), **kwargs)


def test_one_chunk_per_slide_even_above_chunk_max() -> None:
    result = deck()
    assert [chunk.slide for chunk in result.chunks] == [1, 2, 3]
    big = result.chunks[1]
    # Таблица на 60 строк — больше любого chunk_max в словах, но не режется.
    assert big.token_count > 200
    assert big.is_table
    assert "| 2059 | 20590 |" in big.text_raw


def test_deck_limit_is_largest_slide() -> None:
    result = deck()
    assert result.max_tokens == max(chunk.token_count for chunk in result.chunks)
    assert result.max_tokens == result.chunks[1].token_count


def test_derived_fields_searchable_but_not_citable() -> None:
    chunk = deck().chunks[1]
    for marker in ("Ключевой посыл", "Визуал", "Тип слайда"):
        assert marker not in chunk.text_raw
        assert marker in chunk.text_enriched
    assert "**Источник:** материалы клиента" in chunk.text_raw
    assert chunk.text_raw.startswith("Выручка выросла на 12%")


def test_section_path_and_doc_header() -> None:
    chunk = deck().chunks[1]
    assert chunk.section == "01. Финансы"
    assert chunk.subsection == "1.1. Выручка"
    assert chunk.slide_type == "диаграмма"
    header = chunk.text_enriched.split("\n")[0]
    assert "ООО «Заказчик»" in header
    assert "2025" in header
    assert "слайд 2" in header
    assert "01. Финансы / 1.1. Выручка" in header
    assert "Раздел:" not in chunk.text_raw


def test_customer_label_from_registry_overrides_frontmatter() -> None:
    chunk = deck(customer_label="крупный девелопер").chunks[0]
    assert "крупный девелопер" in chunk.text_enriched
    assert "ООО «Заказчик»" not in chunk.text_enriched


def test_slide_above_embedder_limit_fails_instead_of_splitting() -> None:
    with pytest.raises(SlideMarkdownError, match="слайд 2"):
        deck(hard_limit=100)


def test_gap_in_slide_numbers_fails() -> None:
    broken = (FRONTMATTER + SLIDES).replace("slide: 2 -->", "slide: 4 -->").replace(
        "[Слайд 2]", "[Слайд 4]"
    )
    with pytest.raises(SlideMarkdownError, match="пропущены"):
        deck(broken)


def test_title_number_mismatch_fails() -> None:
    with pytest.raises(SlideMarkdownError, match="Слайд 9"):
        deck((FRONTMATTER + SLIDES).replace("[Слайд 3]", "[Слайд 9]"))


def test_slides_total_mismatch_fails() -> None:
    with pytest.raises(SlideMarkdownError, match="slides_total"):
        deck((FRONTMATTER + SLIDES).replace("slides_total: 3", "slides_total: 147"))


def test_detects_slide_markdown() -> None:
    assert is_slide_markdown(FRONTMATTER + SLIDES)
    assert not is_slide_markdown("# Отчет\n\nОбычный документ без слайдов.")
