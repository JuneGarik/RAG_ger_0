"""Чанкинг md-документа (Word и прочий текст без разметки слайдов), SPEC 4.2.

Документ режется по структуре: граница заголовка — всегда граница чанка;
внутри раздела блоки (абзацы, списки) набираются до `chunk_target` токенов.
Таблица и рисунок неделимы, как слайд в презентации: таблица без шапки и
рисунок без подписи теряют смысл. Хвост короче `chunk_min` приклеивается к
предыдущему чанку того же раздела.

Путь раздела (заголовки уровней 2+) идёт в `section`/`subsection` и в шапку
`text_enriched`; заголовок первого уровня — название документа.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bzpo.ingest.slides import SlideChunk, SlideDeck, SlideMarkdownError, _split_frontmatter

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FIGURE = re.compile(r"^<!-- figure: (\d+); kind: (\w+) -->")
_TABLE_ROW = re.compile(r"^\|.*\|\s*$")


@dataclass(slots=True)
class _Block:
    text: str
    atomic: bool = False
    table: bool = False
    figure: str | None = None


@dataclass(slots=True)
class _Pending:
    path: tuple[str, ...]
    blocks: list[_Block] = field(default_factory=list)
    tokens: int = 0


def chunk_document(
    text: str,
    count_tokens: Callable[[str], int],
    *,
    target: int,
    minimum: int,
    hard_limit: int,
    customer_label: str | None = None,
) -> SlideDeck:
    meta, body = _split_frontmatter(text)
    if customer_label is not None:
        meta = {**meta, "customer": customer_label}
    title = str(meta.get("doc_title") or "")
    pending: list[_Pending] = []
    path: list[str] = []

    for block in _blocks(body):
        heading = _HEADING.match(block.text) if "\n" not in block.text else None
        if heading:
            level = len(heading.group(1))
            if level == 1:
                title = title or heading.group(2)
                continue
            del path[level - 2:]
            path += [""] * (level - 2 - len(path)) + [heading.group(2)]
            pending.append(_Pending(tuple(p for p in path if p)))
            continue
        tokens = count_tokens(block.text)
        if tokens > hard_limit:
            raise SlideMarkdownError(
                f"блок «{block.text[:60]}…»: {tokens} токенов больше предела эмбеддера {hard_limit}"
            )
        current = pending[-1] if pending else None
        if current is None or current.path != tuple(p for p in path if p):
            current = _Pending(tuple(p for p in path if p))
            pending.append(current)
        overflow = current.blocks and current.tokens + tokens > target
        if overflow or (block.atomic and current.blocks):
            current = _Pending(current.path)
            pending.append(current)
        current.blocks.append(block)
        current.tokens += tokens
        if block.atomic:
            pending.append(_Pending(current.path))

    pending = _merge_small([p for p in pending if p.blocks], minimum)
    prefix = _doc_prefix(meta, title)
    chunks = []
    for number, part in enumerate(pending, start=1):
        heading = part.path[-1] if part.path else title
        content = "\n\n".join(b.text for b in part.blocks)
        text_raw = f"{heading}\n\n{content}" if heading else content
        location = " / ".join(part.path)
        figures = [b.figure for b in part.blocks if b.figure]
        header = prefix + (f". Раздел: {location}" if location else "")
        if figures:
            header += f". Рисунок: {', '.join(figures)}"
        enriched = f"{header}\n\n{text_raw}"
        chunks.append(SlideChunk(
            slide=number,
            title=heading,
            section=part.path[0] if part.path else None,
            subsection=" / ".join(part.path[1:]) or None,
            slide_type="рисунок" if figures and len(part.blocks) == 1 else "раздел документа",
            text_raw=text_raw,
            text_enriched=enriched,
            token_count=count_tokens(enriched),
            is_table=any(b.table for b in part.blocks),
            page=None,
        ))
    if not chunks:
        raise SlideMarkdownError("в документе нет текста")
    return SlideDeck(meta=meta, chunks=chunks, kind="document")


def _blocks(body: str) -> list[_Block]:
    blocks = []
    for raw in re.split(r"\n\s*\n", body):
        text = raw.strip()
        if not text or text == "---":
            continue
        figure = _FIGURE.match(text)
        if figure:
            kind = "скриншот слайда" if figure.group(2) == "slide_screenshot" else "изображение"
            blocks.append(_Block(text, atomic=True, figure=f"№ {figure.group(1)} ({kind})"))
            continue
        lines = text.split("\n")
        heading_lines = [i for i, line in enumerate(lines) if _HEADING.match(line)]
        if heading_lines and len(lines) > 1:
            # Заголовок, приклеенный к тексту без пустой строки, — отдельный блок.
            start = 0
            for i in heading_lines:
                if i > start:
                    blocks.append(_Block("\n".join(lines[start:i])))
                blocks.append(_Block(lines[i]))
                start = i + 1
            if start < len(lines):
                blocks.append(_Block("\n".join(lines[start:])))
            continue
        table = sum(bool(_TABLE_ROW.match(line)) for line in lines) >= 2
        blocks.append(_Block(text, atomic=table, table=table))
    return blocks


def _merge_small(parts: list[_Pending], minimum: int) -> list[_Pending]:
    merged: list[_Pending] = []
    for part in parts:
        atomic = any(b.atomic for b in part.blocks)
        previous = merged[-1] if merged else None
        if (previous is not None and part.tokens < minimum and not atomic
                and previous.path == part.path
                and not any(b.atomic for b in previous.blocks)):
            previous.blocks += part.blocks
            previous.tokens += part.tokens
        else:
            merged.append(part)
    return merged


def _doc_prefix(meta: dict[str, Any], title: str) -> str:
    parts = [str(meta.get("doc_type") or "Документ")]
    if title:
        parts.append(f"«{title}»")
    if meta.get("customer"):
        parts.append(f"заказчик {meta['customer']}")
    if meta.get("report_date"):
        parts.append(str(meta["report_date"])[:4])
    return ", ".join(parts)
