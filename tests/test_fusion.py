"""Тесты weighted-RRF (SPEC 5.2, PLAN 3.3).

Все ожидаемые значения посчитаны вручную и выписаны в комментариях: тест,
который сравнивает результат функции с результатом той же функции, ничего
не проверяет.
"""

from __future__ import annotations

import pytest

from bzpo.search.fusion import RankedList, standard_lists, weighted_rrf


def test_plain_rrf_matches_hand_calculation() -> None:
    """RRF без весов, k = 60, ранги с единицы.

    список A = [d1, d2, d3], список B = [d2, d4]

        d1 = 1/(60+1)              = 0.016393442622950820
        d2 = 1/(60+2) + 1/(60+1)   = 0.016129032258064516 + 0.016393442622950820
                                   = 0.032522474881015336
        d3 = 1/(60+3)              = 0.015873015873015872
        d4 = 1/(60+2)              = 0.016129032258064516

    Порядок: d2 > d1 > d4 > d3. d4 обходит d3, хотя стоит вторым в своём
    списке, — на втором ранге вклад больше, чем на третьем.
    """
    lists = [
        RankedList("A", ["d1", "d2", "d3"]),
        RankedList("B", ["d2", "d4"]),
    ]
    fused = weighted_rrf(lists, rrf_k=60)

    assert [candidate.candidate_id for candidate in fused] == ["d2", "d1", "d4", "d3"]
    scores = {candidate.candidate_id: candidate.score for candidate in fused}
    assert scores["d1"] == pytest.approx(0.016393442622950820)
    assert scores["d2"] == pytest.approx(0.032522474881015336)
    assert scores["d3"] == pytest.approx(0.015873015873015872)
    assert scores["d4"] == pytest.approx(0.016129032258064516)


def test_weighted_rrf_protects_query_candidates_from_hypothesis() -> None:
    """Вес 0.7 у списков гипотезы, k = 60.

        vector:query      = [c1, c2, c3]
        fts:query         = [c4, c1]
        vector:hypothesis = [c5, c2]
        fts:hypothesis    = [c5, c6]

        c1 = 1.0/61 + 1.0/62 = 0.0163934426 + 0.0161290323 = 0.0325224749
        c2 = 1.0/62 + 0.7/62 = 0.0161290323 + 0.0112903226 = 0.0274193548
        c5 = 0.7/61 + 0.7/61 = 0.0114754098 x 2            = 0.0229508197
        c4 = 1.0/61                                        = 0.0163934426
        c3 = 1.0/63                                        = 0.0158730159
        c6 = 0.7/62                                        = 0.0112903226

    Смысл весов виден на паре c5 / c2. Без весов c5 = 2/61 = 0.0327868852,
    c2 = 2/62 = 0.0322580645, и кандидат, найденный ТОЛЬКО гипотезой, обходит
    кандидата исходного запроса. С весами порядок обратный: гипотеза добавляет
    кандидатов, но не вытесняет найденное запросом (recs.md п.50).
    """
    lists = standard_lists(
        vector_query=["c1", "c2", "c3"],
        fts_query=["c4", "c1"],
        vector_hypothesis=["c5", "c2"],
        fts_hypothesis=["c5", "c6"],
        weight_hyde=0.7,
    )
    fused = weighted_rrf(lists, rrf_k=60)

    assert [candidate.candidate_id for candidate in fused] == ["c1", "c2", "c5", "c4", "c3", "c6"]
    scores = {candidate.candidate_id: candidate.score for candidate in fused}
    assert scores["c1"] == pytest.approx(0.0325224749)
    assert scores["c2"] == pytest.approx(0.0274193548)
    assert scores["c5"] == pytest.approx(0.0229508197)
    assert scores["c6"] == pytest.approx(0.0112903226)

    # Контрольный расчёт без весов: порядок c2 и c5 меняется на противоположный.
    unweighted = weighted_rrf(
        [RankedList(item.name, item.items, 1.0) for item in lists], rrf_k=60
    )
    order = [candidate.candidate_id for candidate in unweighted]
    assert order.index("c5") < order.index("c2")


def test_provenance_records_lists_and_ranks() -> None:
    """Провенанс несёт метрику вклада HyDE без отдельной разметки (SPEC 5.2)."""
    lists = standard_lists(
        vector_query=["c1", "c2"],
        fts_query=["c3", "c1"],
        vector_hypothesis=["c4"],
        fts_hypothesis=["c4"],
    )
    fused = {candidate.candidate_id: candidate for candidate in weighted_rrf(lists)}

    assert fused["c1"].provenance == {"vector:query": 1, "fts:query": 2}
    assert fused["c4"].provenance == {"vector:hypothesis": 1, "fts:hypothesis": 1}
    # Кандидат, найденный только гипотезой, опознаётся по составу списков.
    assert fused["c4"].source_lists == ("vector:hypothesis", "fts:hypothesis")


def test_hyde_disabled_leaves_two_lists() -> None:
    lists = standard_lists(vector_query=["c1"], fts_query=["c2"])
    assert [item.name for item in lists] == ["vector:query", "fts:query"]
    assert all(item.weight == 1.0 for item in lists)


def test_duplicate_inside_list_counts_once_by_best_rank() -> None:
    """Повтор внутри списка не должен удваивать вклад одного и того же чанка."""
    fused = weighted_rrf([RankedList("A", ["d1", "d2", "d1"])], rrf_k=60)
    scores = {candidate.candidate_id: candidate.score for candidate in fused}

    assert scores["d1"] == pytest.approx(1 / 61)
    assert fused[0].candidate_id == "d1"


def test_ties_are_broken_deterministically() -> None:
    """Одинаковые оценки — порядок по идентификатору, иначе прогон датасета
    не воспроизводится между запусками."""
    first = weighted_rrf([RankedList("A", ["b"]), RankedList("B", ["a"])])
    second = weighted_rrf([RankedList("A", ["a"]), RankedList("B", ["b"])])

    assert [candidate.candidate_id for candidate in first] == ["a", "b"]
    assert [candidate.candidate_id for candidate in second] == ["a", "b"]


def test_top_n_cuts_result() -> None:
    lists = standard_lists(vector_query=["c1", "c2", "c3"], fts_query=["c4"])
    assert len(weighted_rrf(lists, top_n=2)) == 2


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        weighted_rrf([RankedList("A", ["d1"])], rrf_k=0)
    with pytest.raises(ValueError, match="вес"):
        weighted_rrf([RankedList("A", ["d1"], weight=-1.0)])


def test_empty_input_gives_empty_result() -> None:
    assert weighted_rrf([]) == []
    assert weighted_rrf([RankedList("A", [])]) == []
