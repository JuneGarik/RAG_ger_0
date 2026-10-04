"""Чанкинг презентаций, размеченных в Markdown: один слайд — один чанк.

Вход — md, подготовленный из pptx (пока вручную, позже — шаг конвейера):
YAML-frontmatter с метаданными документа, затем блоки слайдов

    <!-- slide: N -->
    ## [Слайд N] Заголовок-вывод
    **Раздел:** ...
    **Подраздел:** ...            (необязательно)
    **Тип слайда:** ...
    **Ключевой посыл:** ...       (необязательно)
    **Визуал:** ...               (необязательно)
    содержание: текст, таблицы, «Источник:»
    ---

ПРАВИЛО: слайд не режется и не склеивается с соседями. Слайд — единица
смысла и единица ссылки («слайд 14» в «Источниках»), а разрезанная таблица
теряет шапку. Поэтому предел чанка для презентации — не `chunk_max`, а размер
самого крупного слайда (`SlideDeck.max_tokens`). Единственный жёсткий предел —
вход эмбеддера (`chunk_hard_limit_embed`): слайд длиннее нельзя векторизовать
целиком, это ошибка разметки, а не повод резать.

Тип слайда, ключевой посыл и визуал — интерпретация, которую добавил
конвертер, а не текст отчёта. Они идут в `text_enriched` (по нему ищем), но не
в `text_raw` (его цитирует генератор и читает реранкер).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import yaml

SLIDE_MARKER = re.compile(r"^<!-- slide: (\d+) -->[ \t]*$", re.M)
_TITLE = re.compile(r"^## \[Слайд (\d+)\][ \t]*(.*)$", re.M)
_SECTION_FIELDS = {"Раздел": "section", "Подраздел": "subsection"}
# Интерпретация конвертера: в поиск — да, в цитируемый текст — нет.
_DERIVED_FIELDS = ("Тип слайда", "Ключевой посыл", "Визуал")
_FIELD_LINE = re.compile(r"^\*\*([^*\n]+?):\*\*[ \t]*(.*)$")
_TABLE_ROW = re.compile(r"^\|.*\|[ \t]*$", re.M)


class SlideMarkdownError(ValueError):
    """md презентации не соответствует разметке: такой документ не индексируется."""


@dataclass(frozen=True, slots=True)
class SlideChunk:
    slide: int
    title: str
    section: str | None
    subsection: str | None
    slide_type: str | None
    text_raw: str
    text_enriched: str
    token_count: int
    is_table: bool


@dataclass(frozen=True, slots=True)
class SlideDeck:
    meta: dict[str, Any]
    chunks: list[SlideChunk]

    @property
    def max_tokens(self) -> int:
        """Предел чанка для этой презентации: размер самого крупного слайда."""
        return max((chunk.token_count for chunk in self.chunks), default=0)


def is_slide_markdown(text: str) -> bool:
    return SLIDE_MARKER.search(text) is not None


def chunk_slides(
    text: str,
    count_tokens: Callable[[str], int],
    *,
    hard_limit: int,
    customer_label: str | None = None,
) -> SlideDeck:
    """Разобрать md презентации в чанки по слайдам.

    `count_tokens` — токенизатор эмбеддера: предел чанка и `token_count`
    считаются в тех же токенах, в которых его ограничивает модель.

    `customer_label` — имя заказчика для шапки с учётом публичности (из
    реестра; для anon — `anon_label`). Без него берётся `customer` из
    frontmatter — только для разбора вне конвейера индексации.
    """
    meta, body = _split_frontmatter(text)
    if customer_label is not None:
        meta = {**meta, "customer": customer_label}
    markers = list(SLIDE_MARKER.finditer(body))
    if not markers:
        raise SlideMarkdownError("нет ни одной метки <!-- slide: N -->")

    numbers = [int(marker.group(1)) for marker in markers]
    expected = list(range(1, len(numbers) + 1))
    if numbers != expected:
        # Номер слайда — адрес ссылки в «Источниках»: пропуск или дубль дают
        # ссылку не на тот слайд, и это не видно до ручной проверки.
        raise SlideMarkdownError(f"слайды не идут подряд с 1: {_describe_gap(numbers)}")
    declared = meta.get("slides_total")
    if declared is not None and declared != len(numbers):
        raise SlideMarkdownError(
            f"slides_total во frontmatter = {declared}, а слайдов в разметке {len(numbers)}"
        )

    doc_prefix = _doc_prefix(meta)
    chunks: list[SlideChunk] = []
    for position, marker in enumerate(markers):
        end = markers[position + 1].start() if position + 1 < len(markers) else len(body)
        chunk = _parse_slide(numbers[position], body[marker.end() : end], doc_prefix)
        tokens = count_tokens(chunk.text_enriched)
        if tokens > hard_limit:
            raise SlideMarkdownError(
                f"слайд {chunk.slide}: {tokens} токенов больше предела эмбеддера "
                f"{hard_limit}; слайд не режется — проверьте разметку"
            )
        chunks.append(replace(chunk, token_count=tokens))
    return SlideDeck(meta=meta, chunks=chunks)


# --------------------------------------------------------------- детали


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        raise SlideMarkdownError("frontmatter открыт строкой '---', но не закрыт")
    meta = yaml.safe_load(text[4:end]) or {}
    if not isinstance(meta, dict):
        raise SlideMarkdownError("frontmatter должен быть словарём YAML")
    return meta, text[end + len("\n---\n") :]


def _parse_slide(number: int, block: str, doc_prefix: str) -> SlideChunk:
    block = re.sub(r"\n---[ \t]*\s*$", "", block.strip("\n")).strip()
    title_match = _TITLE.search(block)
    if title_match is None:
        raise SlideMarkdownError(f"слайд {number}: нет заголовка '## [Слайд {number}] …'")
    if int(title_match.group(1)) != number:
        raise SlideMarkdownError(
            f"слайд {number}: в заголовке стоит [Слайд {title_match.group(1)}]"
        )
    title = title_match.group(2).strip()

    fields: dict[str, str] = {}
    derived: list[str] = []
    raw_lines: list[str] = []
    for line in block[title_match.end() :].split("\n"):
        field = _FIELD_LINE.match(line)
        if field and field.group(1) in _SECTION_FIELDS:
            fields[_SECTION_FIELDS[field.group(1)]] = field.group(2).strip()
        elif field and field.group(1) in _DERIVED_FIELDS:
            fields[field.group(1)] = field.group(2).strip()
            derived.append(line)
        else:
            raw_lines.append(line)

    content = re.sub(r"\n{3,}", "\n\n", "\n".join(raw_lines)).strip()
    text_raw = f"{title}\n\n{content}" if content else title
    path = " / ".join(v for v in (fields.get("section"), fields.get("subsection")) if v)
    header = f"{doc_prefix}, слайд {number}" + (f". Раздел: {path}" if path else "")
    text_enriched = "\n".join([header, *derived, "", text_raw])
    return SlideChunk(
        slide=number,
        title=title,
        section=fields.get("section"),
        subsection=fields.get("subsection"),
        slide_type=fields.get("Тип слайда"),
        text_raw=text_raw,
        text_enriched=text_enriched,
        token_count=0,
        is_table=len(_TABLE_ROW.findall(content)) >= 2,
    )


def _doc_prefix(meta: Mapping[str, Any]) -> str:
    """Детерминированная шапка вместо LLM-обогащения: всё нужное уже в разметке."""
    parts = [str(meta.get("doc_type") or "Презентация")]
    if meta.get("doc_title"):
        parts.append(f"«{meta['doc_title']}»")
    if meta.get("customer"):
        parts.append(f"заказчик {meta['customer']}")
    if meta.get("report_date"):
        parts.append(str(meta["report_date"])[:4])
    return ", ".join(parts)


def _describe_gap(numbers: list[int]) -> str:
    counts = Counter(numbers)
    duplicates = sorted(n for n, count in counts.items() if count > 1)
    missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
    return f"пропущены {missing[:10]}, повторяются {duplicates[:10]}"
