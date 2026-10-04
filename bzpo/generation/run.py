"""Запуск генерации подхода: поиск по граням -> контекст -> Qwen -> проверка.

Общая точка для CLI (`python -m bzpo.generation`) и интерфейса (bzpo/ui):
оба вызывают `generate_approach` и получают одинаковый результат.

Режимы:
  quick    — 2 фрагмента на грань и краткий формат (таблицы до 3 строк,
             разделы по 1–2 пункта), ответ до 2200 токенов: ~2 мин;
  thorough — 4 фрагмента на грань, полный формат, до gen_max_tokens: ~3 мин.
Лимит быстрого режима — страховка, а не способ сократить: ответ, обрезанный
посередине раздела, бесполезен, поэтому краткость задаётся промптом.
Модель в обоих режимах одна (роль query): GigaChat Lightning на этой задаче
выдумывал факты, и быстрый режим экономит на объёме, а не на точности.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from bzpo.config import Settings
from bzpo.generation.approach import (
    SYSTEM_PROMPT,
    ApproachRequest,
    Section,
    assemble_context,
    build_prompt,
    facet_queries,
    postprocess,
    render,
)
from bzpo.providers.ollama import OllamaClient
from bzpo.search.query import search
from bzpo.storage import repo

Mode = Literal["quick", "thorough"]


@dataclass(frozen=True, slots=True)
class ModeParams:
    per_facet: int
    num_predict: int | None  # None — gen_max_tokens из конфигурации
    brief: bool


MODES: dict[Mode, ModeParams] = {
    "quick": ModeParams(per_facet=2, num_predict=2200, brief=True),
    "thorough": ModeParams(per_facet=4, num_predict=None, brief=False),
}


@dataclass(slots=True)
class ApproachRun:
    request: ApproachRequest
    mode: Mode
    section: Section | None
    markdown: str
    not_found: list[str]
    search_seconds: float
    generation_seconds: float
    completion_tokens: int | None
    truncated: bool


def generate_approach(
    conn: Any,
    settings: Settings,
    embedder: Any,
    reranker: Any,
    request: ApproachRequest,
    *,
    mode: Mode = "thorough",
    scope: repo.SearchScope | None = None,
    progress: Callable[[str], None] = lambda message: None,
) -> ApproachRun:
    params = MODES[mode]
    started = time.perf_counter()
    hits: dict[str, list[Any]] = {}
    not_found: list[str] = []
    for facet, query in facet_queries(request).items():
        progress(f"поиск: грань «{facet}»")
        result = search(conn, settings, embedder, reranker, query, top_k=params.per_facet,
                        scope=scope)
        hits[facet] = result.hits
        if result.not_found:
            not_found.append(facet)
    search_seconds = time.perf_counter() - started
    if not any(hits.values()):
        return ApproachRun(request, mode, None, "Ничего не найдено ни по одной грани подхода: "
                           "генерация не выполнялась.", not_found, search_seconds, 0.0, None, False)

    margin = settings.generation.context_tokenizer_margin
    fragments = assemble_context(
        hits, lambda text: int(embedder.count_tokens(text) * margin),
        settings.generation.context_budget_tokens,
    )
    progress(f"генерация: {len(fragments)} фрагментов в контексте")
    with OllamaClient(settings.ollama, role="query") as client:
        generation = client.generate(
            build_prompt(request, fragments, brief=params.brief),
            system=SYSTEM_PROMPT.read_text(encoding="utf-8"),
            num_predict=params.num_predict,
        )
    section = postprocess(generation.text, fragments, request)
    markdown = render(section)
    if not_found:
        markdown += "\n\n### Не найдено в корпусе\n" + "\n".join(f"- грань {f}" for f in not_found)
    return ApproachRun(
        request, mode, section, markdown, not_found, search_seconds,
        time.perf_counter() - started - search_seconds,
        generation.completion_tokens, generation.truncated_by_length,
    )


def save_draft(run: ApproachRun, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now():%Y%m%d_%H%M%S}_approach.md"
    r = run.request
    header = (f"<!-- цель: {r.goal} | область: {r.area or '—'} | ситуация: {r.situation or '—'} "
              f"| режим: {run.mode} -->")
    path.write_text(f"{header}\n\n{run.markdown}\n", encoding="utf-8")
    return path
