"""Описание подхода: подзапросы граней, сборка контекста, постобработка, источники."""

from __future__ import annotations

from datetime import date

from bzpo.generation.approach import (
    FACETS,
    ApproachRequest,
    Fragment,
    assemble_context,
    build_prompt,
    facet_queries,
    postprocess,
    render,
)
from bzpo.search.query import SearchHit
from bzpo.storage.repo import ChunkHit


def hit(cid: str, text: str, publicity: str = "open", page: int = 1) -> SearchHit:
    chunk = ChunkHit(cid, text, text, [1.0], page, "Раздел", "Отчёт ДЕВ", date(2025, 7, 18),
                     "Недвижимость", "won", publicity, "ГК ДЕВ", "крупный девелопер",
                     "Диагностика")
    return SearchHit(cid, 1.0, 1.0, (), {}, chunk)


def words(text: str) -> int:
    return len(text.split())


def test_one_query_per_facet() -> None:
    queries = facet_queries(ApproachRequest("Диагностика зрелости", area="девелопмент"))
    assert set(queries) == set(FACETS)
    assert all(q.startswith("Диагностика зрелости (девелопмент): ") for q in queries.values())


def test_context_round_robin_dedup_and_labels() -> None:
    hits = {
        "stages": [hit("a", "этапы проекта"), hit("b", "ещё этапы")],
        "risks": [hit("a", "этапы проекта"), hit("c", "риски", publicity="anon")],
    }
    fragments = assemble_context(hits, words, budget=1000)
    # По кругу: a (stages), a уже был -> c (risks), затем b.
    assert [f.hit.chunk_id for f in fragments] == ["a", "c", "b"]
    assert [f.number for f in fragments] == [1, 2, 3]
    assert fragments[0].text.startswith("[1] Проект «Диагностика»; заказчик: ГК ДЕВ; Отчёт ДЕВ")
    assert "заказчик: крупный девелопер" in fragments[1].text  # anon -> обезличенное имя
    assert fragments[0].body == "этапы проекта"


def test_context_budget_drops_tail_not_facet() -> None:
    hits = {"stages": [hit("a", "x " * 30), hit("b", "x " * 30)], "risks": [hit("c", "y " * 30)]}
    fragments = assemble_context(hits, words, budget=90)
    assert {f.facet for f in fragments} == {"stages", "risks"}
    assert len(fragments) == 2


def fragments() -> list[Fragment]:
    return [
        Fragment(1, "methodology", hit("a", "Проведено 47 интервью и опрос", page=36),
                 "[1] метка\nПроведено 47 интервью и опрос"),
        Fragment(2, "risks", hit("b", "Отсутствие стратегии ЦТ", page=19),
                 "[2] метка\nОтсутствие стратегии ЦТ"),
    ]


def test_prompt_contains_inputs_and_fragments() -> None:
    prompt = build_prompt(ApproachRequest("Цель", situation="нет стратегии"), fragments())
    assert "Цель нового проекта: Цель" in prompt
    assert "Ситуация нового заказчика: нет стратегии" in prompt
    assert "[2] метка" in prompt


def test_postprocess_flags_and_sources() -> None:
    answer = "\n".join([
        "## Подход: тест",
        "### 1. Опора: подходы прошлых проектов",
        "В проекте для девелопера проведено 47 интервью [1].",
        "В проекте проведено 52 интервью с менеджерами [1].",  # числа нет во фрагменте
        "Барьер проекта — отсутствие стратегии и руководство сомневалось в пользе [2].",
        "### 3. Этапы и действия",
        "Этап нового проекта: интервью руководства, которое сомневается в окупаемости [1].",
        "Заказчик сомневается в окупаемости цифровизации и ведёт учёт в Excel.",
        "[вывод] Для нового заказчика начать с интервью ключевых руководителей.",
        "(вывод) Пометка в круглых скобках тоже засчитывается моделью.",
        "Эта строка утверждает что-то важное без какого-либо источника.",
        "Ссылка на несуществующий фрагмент номер девять здесь [9].",
        "### 7. Пробелы и вопросы для уточнения",
        "- Какие процессы ведутся в Excel и сколько сотрудников их выполняют",
    ])
    request = ApproachRequest(
        "Стратегия ЦТ", situation="руководство сомневается в окупаемости цифровизации, учёт в Excel"
    )
    section = postprocess(answer, fragments(), request)
    issues = "\n".join(section.unverified)
    assert "[9]" not in section.text and "ссылка [9]" in issues
    assert "52" in issues
    assert "перенос ситуации" in issues and "сомне" in issues
    # Вне раздела «Опора» упоминание ситуации нового заказчика — не перенос.
    assert "Этап нового проекта" not in issues
    # Пересказ входных данных — не утверждение без источника.
    assert "ведёт учёт в Excel" not in issues
    assert "без какого-либо источника" in issues
    assert "Пометка в круглых" not in issues and "ключевых руководителей" not in issues
    assert "сколько сотрудников" not in issues  # раздел «Пробелы» источника не требует
    assert section.cited == [1, 2]

    output = render(section)
    assert "### Источники" in output
    assert "Отчёт ДЕВ — проект «Диагностика», заказчик: ГК ДЕВ: [1] слайд 36; [2] слайд 19" in output
    assert "### Не подтверждено" in output


def test_brief_mode_adds_instruction() -> None:
    request = ApproachRequest("Цель")
    assert "КРАТКИЙ РЕЖИМ" in build_prompt(request, fragments(), brief=True)
    assert "КРАТКИЙ РЕЖИМ" not in build_prompt(request, fragments())
