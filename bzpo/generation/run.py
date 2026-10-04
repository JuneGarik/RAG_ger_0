"""Запуск генерации подхода: поиск по граням -> контекст -> Qwen -> проверка.

Общая точка для CLI (`python -m bzpo.generation`) и интерфейса (bzpo/ui):
оба вызывают `generate_approach` и получают одинаковый результат.

Режимы:
  quick    — 2 фрагмента на грань и краткий формат (таблицы до 3 строк,
             разделы по 1–2 пункта), ответ до 3000 токенов: ~1,5–2 мин;
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
from bzpo.search.query import SearchResult, search
from bzpo.storage import repo

Mode = Literal["quick", "thorough"]


@dataclass(frozen=True, slots=True)
class ModeParams:
    per_facet: int
    candidates: int  # сколько кандидатов на грань читает реранкер
    num_predict: int | None  # None — gen_max_tokens из конфигурации
    brief: bool


# Кандидаты реранкера на грань (замер на отчёте ДЕВ, 6 граней): 50 — 20 с,
# 30 — 12 с и 83% тех же слайдов, 20 — 10 с и 62%. Выигрыш мал рядом с
# генерацией (~2 мин), поэтому «тщательно» не экономит на поиске.
MODES: dict[Mode, ModeParams] = {
    "quick": ModeParams(per_facet=2, candidates=30, num_predict=3000, brief=True),
    "thorough": ModeParams(per_facet=4, candidates=50, num_predict=None, brief=False),
}


@dataclass(slots=True)
class ApproachRun:
    request: ApproachRequest
    mode: Mode
    facets: dict[str, SearchResult]
    section: Section | None
    markdown: str
    not_found: list[str]
    search_seconds: float
    generation_seconds: float
    completion_tokens: int | None
    truncated: bool


def search_facets(
    conn: Any,
    settings: Settings,
    embedder: Any,
    reranker: Any,
    request: ApproachRequest,
    *,
    mode: Mode = "thorough",
    scope: repo.SearchScope | None = None,
    progress: Callable[[str, SearchResult], None] = lambda facet, result: None,
) -> dict[str, SearchResult]:
    """Поиск по всем граням подхода; `progress` вызывается после каждой грани."""
    params = MODES[mode]
    results: dict[str, SearchResult] = {}
    for facet, query in facet_queries(request).items():
        result = search(conn, settings, embedder, reranker, query, top_k=params.per_facet,
                        scope=scope, candidates=params.candidates)
        results[facet] = result
        progress(facet, result)
    return results


def generate_approach(
    conn: Any,
    settings: Settings,
    embedder: Any,
    reranker: Any,
    request: ApproachRequest,
    *,
    mode: Mode = "thorough",
    scope: repo.SearchScope | None = None,
    facets: dict[str, SearchResult] | None = None,
    progress: Callable[[str], None] = lambda message: None,
    on_token: Callable[[str], None] | None = None,
) -> ApproachRun:
    """Поиск (если `facets` не переданы) -> контекст -> генерация -> проверка.

    `on_token` включает потоковую генерацию: каждый кусок текста передаётся
    сразу, итог и проверка — после завершения.
    """
    params = MODES[mode]
    started = time.perf_counter()
    if facets is None:
        facets = search_facets(
            conn, settings, embedder, reranker, request, mode=mode, scope=scope,
            progress=lambda facet, result: progress(f"поиск: грань «{facet}»"),
        )
    not_found = [facet for facet, result in facets.items() if result.not_found]
    search_seconds = time.perf_counter() - started
    hits = {facet: result.hits for facet, result in facets.items()}
    if not any(hits.values()):
        return ApproachRun(request, mode, facets, None,
                           "Ничего не найдено ни по одной грани подхода: генерация не выполнялась.",
                           not_found, search_seconds, 0.0, None, False)

    margin = settings.generation.context_tokenizer_margin
    fragments = assemble_context(
        hits, lambda text: int(embedder.count_tokens(text) * margin),
        settings.generation.context_budget_tokens,
    )
    progress(f"генерация: {len(fragments)} фрагментов в контексте")
    prompt = build_prompt(request, fragments, brief=params.brief)
    system = SYSTEM_PROMPT.read_text(encoding="utf-8")
    with OllamaClient(settings.ollama, role="query") as client:
        if on_token is None:
            generation = client.generate(prompt, system=system, num_predict=params.num_predict)
        else:
            generation = None
            for event in client.stream(prompt, system=system, num_predict=params.num_predict):
                if isinstance(event, str):
                    on_token(event)
                else:
                    generation = event
            assert generation is not None  # stream() всегда завершается Generation
    section = postprocess(generation.text, fragments, request)
    markdown = render(section)
    if generation.truncated_by_length:
        markdown += ("\n\n> ⚠ Ответ модели обрезан по лимиту длины: последний раздел может "
                     "быть неполным.")
    if not_found:
        markdown += "\n\n### Не найдено в корпусе\n" + "\n".join(f"- грань {f}" for f in not_found)
    return ApproachRun(
        request, mode, facets, section, markdown, not_found, search_seconds,
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
