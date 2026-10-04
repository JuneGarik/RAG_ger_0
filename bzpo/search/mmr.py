"""MMR с нормировкой обеих шкал (SPEC 5.2).

    выбор = argmax  λ · relevance'(d) − (1 − λ) · max_{s ∈ S} sim'(d, s)

Штрих означает min-max-нормировку по множеству кандидатов. Она обязательна:
шкалы несопоставимы по построению. Оценка реранкера после сигмоиды занимает
почти весь отрезок [0, 1], а косинус между чанками одного корпуса лежит в
узкой полосе 0.85–0.95. Без нормировки слагаемое разнообразия меняется на
сотые доли, слагаемое релевантности — на десятые, и MMR вырождается в порядок
реранкера: код формально есть, эффекта нет (recs.md п.52).

Флаг `normalize=False` оставлен ровно для одного потребителя — теста
вырождения. В рабочем коде он не используется.

Близость считается по `embedding_raw` (вектор до обогащения): обогащение
приписывает каждому фрагменту заказчика и год, из-за чего близость копий
типового блока искусственно падает и дедупликация с MMR перестают их видеть
(SPEC раздел 3).

Модуль чистый: ни базы, ни сети.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MMRCandidate:
    """Кандидат на вход MMR: оценка после реранкинга и boost + вектор для близости."""

    candidate_id: str
    relevance: float
    vector: Sequence[float]


def mmr(
    candidates: Sequence[MMRCandidate],
    *,
    lambda_: float = 0.7,
    top_k: int = 8,
    normalize: bool = True,
) -> list[MMRCandidate]:
    """Отобрать `top_k` кандидатов, балансируя релевантность и разнообразие.

    Возвращает кандидатов в порядке отбора. Порядок детерминирован: при равных
    оценках выбирается кандидат с меньшим индексом во входном списке.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda должна лежать в [0, 1], получено {lambda_}")
    if top_k <= 0:
        raise ValueError(f"top_k должен быть положительным, получено {top_k}")
    if not candidates:
        return []

    size = len(candidates)
    relevance = [candidate.relevance for candidate in candidates]
    similarity = _similarity_matrix(candidates)

    if normalize:
        relevance = _min_max(relevance)
        similarity = _normalize_similarity(similarity)

    selected: list[int] = []
    remaining = set(range(size))
    limit = min(top_k, size)

    while len(selected) < limit:
        best_index: int | None = None
        best_value = -math.inf
        for index in sorted(remaining):
            if selected:
                penalty = max(similarity[index][chosen] for chosen in selected)
            else:
                penalty = 0.0
            value = lambda_ * relevance[index] - (1.0 - lambda_) * penalty
            if value > best_value:
                best_value = value
                best_index = index
        assert best_index is not None
        selected.append(best_index)
        remaining.discard(best_index)

    return [candidates[index] for index in selected]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Косинусная близость. Векторы корпуса нормализованы, но опора на это
    не делается: ошибка нормализации иначе проявилась бы как тихий сдвиг MMR."""
    if len(left) != len(right):
        raise ValueError(f"размерности векторов различаются: {len(left)} и {len(right)}")
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(math.fsum(a * a for a in left))
    norm_right = math.sqrt(math.fsum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _similarity_matrix(candidates: Sequence[MMRCandidate]) -> list[list[float]]:
    size = len(candidates)
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            value = cosine(candidates[i].vector, candidates[j].vector)
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def _min_max(values: Sequence[float]) -> list[float]:
    """Нормировка в [0, 1]. Вырожденный случай (все значения равны) даёт нули:
    константа одинаково сдвигает всех кандидатов и на argmax не влияет."""
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0.0:
        return [0.0] * len(values)
    return [(value - low) / span for value in values]


def _normalize_similarity(matrix: list[list[float]]) -> list[list[float]]:
    """Min-max по внедиагональным элементам: диагональ (единицы) в отбор
    не участвует и, попав в статистику, сжала бы шкалу разнообразия."""
    size = len(matrix)
    if size < 2:
        return matrix
    off_diagonal = [matrix[i][j] for i in range(size) for j in range(size) if i != j]
    low = min(off_diagonal)
    high = max(off_diagonal)
    span = high - low
    if span == 0.0:
        return [[0.0] * size for _ in range(size)]
    return [
        [0.0 if i == j else (matrix[i][j] - low) / span for j in range(size)] for i in range(size)
    ]
