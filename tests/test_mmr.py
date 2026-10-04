"""Тесты MMR и обязательности нормировки (SPEC 5.2, PLAN 3.4).

Данные подобраны так, чтобы разница между нормированной и ненормированной
версией была не «немного другой», а качественной: без нормировки MMR
воспроизводит порядок реранкера дословно, то есть не работает.
"""

from __future__ import annotations

import math

import pytest

from bzpo.search.mmr import MMRCandidate, cosine, mmr


def unit(degrees: float) -> list[float]:
    """Единичный вектор под углом: косинус между двумя — косинус разности углов."""
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


def sample() -> list[MMRCandidate]:
    """Реалистичная картина: оценки реранкера широкие, косинусы — узкая полоса.

        A: угол 0,   relevance 0.95
        B: угол 8,   relevance 0.90   — почти дубль A, cos(A,B) = 0.9903
        C: угол 25,  relevance 0.85   — содержательно другой, cos(A,C) = 0.9063
        D: угол 45,  relevance 0.30   — далёкий и нерелевантный

    Косинусы: AB 0.99027, AC 0.90631, AD 0.70711,
              BC 0.95630, BD 0.79864, CD 0.93969
    """
    return [
        MMRCandidate("A", 0.95, unit(0)),
        MMRCandidate("B", 0.90, unit(8)),
        MMRCandidate("C", 0.85, unit(25)),
        MMRCandidate("D", 0.30, unit(45)),
    ]


def test_normalized_mmr_prefers_diverse_over_near_duplicate() -> None:
    """λ = 0.7, нормировка включена.

    Нормировка релевантности (min 0.30, max 0.95, размах 0.65):
        A 1.0000, B 0.9231, C 0.8462, D 0.0000
    Нормировка близости по внедиагональным (min 0.70711, max 0.99027,
    размах 0.28316):
        AB 1.0000, AC 0.70349, AD 0.0000, BC 0.88006, BD 0.32324, CD 0.82139

    Шаг 1: максимум релевантности — A.
    Шаг 2: B = 0.7*0.9231 - 0.3*1.00000 = 0.34615
           C = 0.7*0.8462 - 0.3*0.70349 = 0.38126   <- побеждает
           D = 0.7*0.0000 - 0.3*0.00000 = 0.00000
    Шаг 3: B = 0.7*0.9231 - 0.3*max(1.0, 0.88006) = 0.34615  <- побеждает
           D = 0.0 - 0.3*max(0.0, 0.82139) = -0.24642

    Итог: A, C, B — почти дубль B отодвинут за содержательно другой C.
    """
    selected = mmr(sample(), lambda_=0.7, top_k=3, normalize=True)
    assert [candidate.candidate_id for candidate in selected] == ["A", "C", "B"]


def test_without_normalization_mmr_degenerates_to_rerank_order() -> None:
    """Те же данные без нормировки, λ = 0.7.

    Шаг 2: B = 0.7*0.90 - 0.3*0.99027 = 0.63000 - 0.29708 = 0.33292  <- побеждает
           C = 0.7*0.85 - 0.3*0.90631 = 0.59500 - 0.27189 = 0.32311
           D = 0.7*0.30 - 0.3*0.70711 = 0.21000 - 0.21213 = -0.00213
    Шаг 3: C = 0.59500 - 0.3*max(0.90631, 0.95630) = 0.30811  <- побеждает
           D = 0.21000 - 0.3*max(0.70711, 0.79864) = -0.02959

    Итог A, B, C — ровно порядок реранкера. Слагаемое разнообразия меняется
    на сотые доли (косинусы в полосе 0.71–0.99), слагаемое релевантности —
    на десятые, и штраф за дублирование ни на что не влияет.
    """
    degenerate = mmr(sample(), lambda_=0.7, top_k=3, normalize=False)
    rerank_order = [
        candidate.candidate_id
        for candidate in sorted(sample(), key=lambda item: -item.relevance)
    ][:3]

    assert [candidate.candidate_id for candidate in degenerate] == ["A", "B", "C"]
    assert [candidate.candidate_id for candidate in degenerate] == rerank_order

    normalized = mmr(sample(), lambda_=0.7, top_k=3, normalize=True)
    assert [candidate.candidate_id for candidate in normalized] != rerank_order


def test_lambda_one_is_pure_relevance() -> None:
    selected = mmr(sample(), lambda_=1.0, top_k=4)
    assert [candidate.candidate_id for candidate in selected] == ["A", "B", "C", "D"]


def test_lambda_zero_is_pure_diversity() -> None:
    """При λ = 0 после первого выбора идёт самый непохожий на выбранное."""
    selected = mmr(sample(), lambda_=0.0, top_k=2)
    assert selected[1].candidate_id == "D"


def test_top_k_limits_and_never_repeats() -> None:
    selected = mmr(sample(), lambda_=0.7, top_k=2)
    assert len(selected) == 2
    assert len({candidate.candidate_id for candidate in selected}) == 2

    everything = mmr(sample(), lambda_=0.7, top_k=99)
    assert len(everything) == 4


def test_equal_relevance_does_not_break_selection() -> None:
    """Вырожденная нормировка (все оценки равны) не должна ломать отбор."""
    candidates = [
        MMRCandidate("A", 0.5, unit(0)),
        MMRCandidate("B", 0.5, unit(5)),
        MMRCandidate("C", 0.5, unit(60)),
    ]
    selected = mmr(candidates, lambda_=0.7, top_k=2)
    assert [candidate.candidate_id for candidate in selected] == ["A", "C"]


def test_empty_and_invalid_arguments() -> None:
    assert mmr([], lambda_=0.7, top_k=8) == []
    with pytest.raises(ValueError, match="lambda"):
        mmr(sample(), lambda_=1.5, top_k=2)
    with pytest.raises(ValueError, match="top_k"):
        mmr(sample(), lambda_=0.7, top_k=0)


def test_cosine_basics() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([3.0, 4.0], [3.0, 4.0]) == pytest.approx(1.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="размерности"):
        cosine([1.0], [1.0, 0.0])
