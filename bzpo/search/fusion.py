"""Объединение списков кандидатов взвешенным RRF (SPEC 5.2).

На каждую грань подхода строятся четыре списка по `PER_LIST_K` кандидатов:
векторный и FTS по подзапросу, векторный и FTS по гипотезе HyDE. Они сливаются
формулой

    score(d) = Σ_i  weight_i / (RRF_K + rank_i(d))          rank — с единицы

Веса нужны именно здесь: без них утверждение «гипотеза только добавляет
кандидатов» неверно — документ, найденный исходным запросом, вытесняется из
среза `CANDIDATES` документами гипотезы (recs.md п.50). Вес 0.7 у списков
гипотезы оставляет ей возможность добавить, но не переставить.

Модуль чистый: ни базы, ни сети. Провенанс (из каких списков и с какими
позициями пришёл кандидат) сохраняется в результате — на нём держится метрика
вклада HyDE без отдельной разметки (SPEC 5.2, recs.md п.76).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

#: Идентификатор кандидата: chunk_id или section_id в виде строки.
CandidateId = str

QUERY_LIST_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class RankedList:
    """Один список кандидатов в порядке убывания релевантности.

    `items` — идентификаторы; ранг определяется позицией. Повтор идентификатора
    внутри одного списка учитывается по лучшей (первой) позиции.
    """

    name: str
    items: Sequence[CandidateId]
    weight: float = QUERY_LIST_WEIGHT


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """Кандидат после слияния: оценка и провенанс (список -> ранг с единицы)."""

    candidate_id: CandidateId
    score: float
    provenance: Mapping[str, int] = field(default_factory=dict)

    @property
    def source_lists(self) -> tuple[str, ...]:
        return tuple(self.provenance)


def standard_lists(
    *,
    vector_query: Sequence[CandidateId],
    fts_query: Sequence[CandidateId],
    vector_hypothesis: Sequence[CandidateId] | None = None,
    fts_hypothesis: Sequence[CandidateId] | None = None,
    weight_hyde: float = 0.7,
) -> list[RankedList]:
    """Собрать четыре списка схемы SPEC 5.2 с правильными весами.

    Списки гипотезы необязательны: при выключенном HyDE (`USE_HYDE=false`)
    схема вырождается в два списка, и это штатный режим, а не ошибка.
    """
    lists = [
        RankedList("vector:query", vector_query, QUERY_LIST_WEIGHT),
        RankedList("fts:query", fts_query, QUERY_LIST_WEIGHT),
    ]
    if vector_hypothesis is not None:
        lists.append(RankedList("vector:hypothesis", vector_hypothesis, weight_hyde))
    if fts_hypothesis is not None:
        lists.append(RankedList("fts:hypothesis", fts_hypothesis, weight_hyde))
    return lists


def weighted_rrf(
    lists: Iterable[RankedList],
    *,
    rrf_k: int = 60,
    top_n: int | None = None,
) -> list[FusedCandidate]:
    """Взвешенный RRF по списку списков.

    Возвращает кандидатов по убыванию оценки; при равных оценках порядок
    определяется идентификатором — иначе прогон датасета не воспроизводится.
    """
    if rrf_k <= 0:
        raise ValueError(f"rrf_k должен быть положительным, получено {rrf_k}")

    scores: dict[CandidateId, float] = {}
    provenance: dict[CandidateId, dict[str, int]] = {}

    for ranked in lists:
        if ranked.weight < 0:
            raise ValueError(f"отрицательный вес списка {ranked.name}: {ranked.weight}")
        for position, candidate_id in enumerate(ranked.items, start=1):
            seen = provenance.setdefault(candidate_id, {})
            if ranked.name in seen:
                # Дубль внутри списка: лучший ранг уже учтён.
                continue
            seen[ranked.name] = position
            scores[candidate_id] = scores.get(candidate_id, 0.0) + ranked.weight / (
                rrf_k + position
            )

    fused = [
        FusedCandidate(candidate_id, score, dict(provenance[candidate_id]))
        for candidate_id, score in scores.items()
    ]
    fused.sort(key=lambda candidate: (-candidate.score, candidate.candidate_id))
    return fused if top_n is None else fused[:top_n]
