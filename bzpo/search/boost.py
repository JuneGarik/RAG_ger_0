"""Аддитивный boost по метаданным реестра (SPEC 5.2).

    score' = score · (1 + w_ind·[отрасль совпала]
                        + w_won·[tender_result = 'won']
                        + w_fresh·max(0, 1 − возраст_лет / горизонт))

Место в конвейере строго определено: ПОСЛЕ реранкинга и порога «не найдено»,
ДО MMR. До реранкинга boost не имеет смысла (шкала кандидатов ещё не
сопоставима), после MMR — не влияет ни на что, отбор уже сделан.

Незаполненные метаданные дают нулевой вклад, а не штраф: реестр заполняется
людьми, и пустое поле означает «неизвестно», а не «нерелевантно». Формула
множительная (SPEC), но слагаемые внутри скобки аддитивны — при оценках
реранкера порядка единицы это те самые +0.05/+0.05/+0.03.

Модуль чистый: ни базы, ни сети; метаданные приходят от вызывающего кода.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # только для типов: модуль поиска не зависит от конфигурации в рантайме
    from bzpo.config import SearchConfig

DAYS_IN_YEAR = 365.25


@dataclass(frozen=True, slots=True)
class BoostWeights:
    """Веса из конфигурации (SPEC раздел 10, BOOST_W_*)."""

    industry: float = 0.05
    won: float = 0.05
    fresh: float = 0.03
    fresh_horizon_years: float = 5.0

    @classmethod
    def from_settings(cls, search: SearchConfig) -> BoostWeights:
        """Собрать веса из секции search конфигурации."""
        return cls(
            industry=search.boost_w_industry,
            won=search.boost_w_won,
            fresh=search.boost_w_fresh,
            fresh_horizon_years=search.boost_fresh_horizon_years,
        )


@dataclass(frozen=True, slots=True)
class BoostInput:
    """Кандидат после реранкинга вместе с метаданными его проекта и документа."""

    candidate_id: str
    score: float
    industry: str | None = None
    tender_result: str | None = None
    doc_date: date | None = None


@dataclass(frozen=True, slots=True)
class BoostedCandidate:
    """Результат: итоговая оценка, множитель и перечень сработавших правил."""

    candidate_id: str
    base_score: float
    score: float
    factor: float
    applied: tuple[str, ...]


def apply_boost(
    candidates: Sequence[BoostInput],
    *,
    weights: BoostWeights,
    query_industry: str | None = None,
    today: date | None = None,
) -> list[BoostedCandidate]:
    """Применить boost и пересортировать по убыванию итоговой оценки."""
    if weights.fresh_horizon_years <= 0:
        raise ValueError(
            f"горизонт свежести должен быть положительным, получено {weights.fresh_horizon_years}"
        )
    reference = today or date.today()

    boosted: list[BoostedCandidate] = []
    for candidate in candidates:
        bonus = 0.0
        applied: list[str] = []

        if _industry_matches(candidate.industry, query_industry):
            bonus += weights.industry
            applied.append("industry")

        if candidate.tender_result == "won":
            bonus += weights.won
            applied.append("won")

        freshness = _freshness(candidate.doc_date, reference, weights.fresh_horizon_years)
        if freshness > 0.0:
            bonus += weights.fresh * freshness
            applied.append("fresh")

        factor = 1.0 + bonus
        boosted.append(
            BoostedCandidate(
                candidate_id=candidate.candidate_id,
                base_score=candidate.score,
                score=candidate.score * factor,
                factor=factor,
                applied=tuple(applied),
            )
        )

    boosted.sort(key=lambda item: (-item.score, item.candidate_id))
    return boosted


def _industry_matches(candidate_industry: str | None, query_industry: str | None) -> bool:
    """Сравнение отраслей по нормализованной форме: реестр заполняют руками."""
    if not candidate_industry or not query_industry:
        return False
    return candidate_industry.strip().casefold() == query_industry.strip().casefold()


def _freshness(doc_date: date | None, reference: date, horizon_years: float) -> float:
    """Линейное затухание от 1 (сегодня) до 0 (горизонт и старше).

    Дата из будущего — ошибка данных, а не рекорд свежести: возраст обрезается
    снизу нулём, максимальный вклад такой же, как у сегодняшнего документа.
    """
    if doc_date is None:
        return 0.0
    age_years = max(0.0, (reference - doc_date).days / DAYS_IN_YEAR)
    return max(0.0, 1.0 - age_years / horizon_years)
