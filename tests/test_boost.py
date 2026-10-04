"""Тесты boost по метаданным (SPEC 5.2)."""

from __future__ import annotations

from datetime import date

import pytest

from bzpo.config import load_config
from bzpo.search.boost import BoostInput, BoostWeights, apply_boost

TODAY = date(2026, 8, 19)
WEIGHTS = BoostWeights(industry=0.05, won=0.05, fresh=0.03, fresh_horizon_years=5.0)


def test_weights_come_from_config() -> None:
    weights = BoostWeights.from_settings(load_config(use_yaml=False).search)
    assert (weights.industry, weights.won, weights.fresh) == (0.05, 0.05, 0.03)


def test_all_three_rules_multiply_score() -> None:
    """Свежесть годовалого документа: 1 - 1/5 = 0.8, вклад 0.03*0.8 = 0.024.

    Множитель = 1 + 0.05 + 0.05 + 0.024 = 1.124.
    """
    candidate = BoostInput(
        candidate_id="c1",
        score=0.50,
        industry="Энергетика",
        tender_result="won",
        doc_date=date(2025, 8, 19),
    )
    boosted = apply_boost(
        [candidate], weights=WEIGHTS, query_industry="энергетика", today=TODAY
    )[0]

    assert boosted.factor == pytest.approx(1.124, abs=1e-3)
    assert boosted.score == pytest.approx(0.50 * 1.124, abs=1e-3)
    assert boosted.applied == ("industry", "won", "fresh")
    assert boosted.base_score == 0.50


def test_missing_metadata_gives_no_penalty() -> None:
    """Пустое поле реестра означает «неизвестно», а не «нерелевантно»."""
    boosted = apply_boost(
        [BoostInput("c1", 0.70)], weights=WEIGHTS, query_industry="энергетика", today=TODAY
    )[0]

    assert boosted.factor == 1.0
    assert boosted.score == 0.70
    assert boosted.applied == ()


def test_industry_match_is_normalized() -> None:
    boosted = apply_boost(
        [BoostInput("c1", 1.0, industry="  Металлургия ")],
        weights=WEIGHTS,
        query_industry="металлургия",
        today=TODAY,
    )[0]
    assert "industry" in boosted.applied


def test_old_document_gets_no_freshness_bonus() -> None:
    boosted = apply_boost(
        [BoostInput("c1", 1.0, doc_date=date(2018, 1, 1))], weights=WEIGHTS, today=TODAY
    )[0]
    assert boosted.factor == 1.0


def test_lost_tender_is_not_boosted() -> None:
    boosted = apply_boost(
        [BoostInput("c1", 1.0, tender_result="lost")], weights=WEIGHTS, today=TODAY
    )[0]
    assert boosted.applied == ()


def test_future_date_does_not_exceed_fresh_bonus() -> None:
    """Дата из будущего — ошибка данных, а не рекорд свежести."""
    boosted = apply_boost(
        [BoostInput("c1", 1.0, doc_date=date(2030, 1, 1))], weights=WEIGHTS, today=TODAY
    )[0]
    assert boosted.factor == pytest.approx(1.03)


def test_result_is_sorted_by_boosted_score() -> None:
    """Boost способен переставить кандидатов — в этом и смысл шага."""
    candidates = [
        BoostInput("плохой-но-выигранный", 0.60, tender_result="won", industry="энергетика"),
        BoostInput("хороший-нейтральный", 0.63),
    ]
    order = [
        item.candidate_id
        for item in apply_boost(
            candidates, weights=WEIGHTS, query_industry="энергетика", today=TODAY
        )
    ]
    assert order == ["плохой-но-выигранный", "хороший-нейтральный"]


def test_invalid_horizon_raises() -> None:
    with pytest.raises(ValueError, match="горизонт"):
        apply_boost(
            [BoostInput("c1", 1.0)],
            weights=BoostWeights(fresh_horizon_years=0.0),
            today=TODAY,
        )
