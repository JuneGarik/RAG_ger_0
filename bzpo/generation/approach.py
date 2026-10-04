"""Генерация описания подхода к новому проекту (сценарий Б, SPEC 5.3, тип approach_description).

Конвейер:
  1. Цель нового проекта разворачивается в подзапросы по шести граням подхода
     (SPEC 5.1: understanding, methodology, stages, artifacts, risks, outcome).
  2. По каждой грани — поиск (search.query); фрагменты сливаются без дублей,
     поочерёдно по граням, чтобы при усечении бюджета не выпала целая грань.
  3. Фрагменты нумеруются [N] и подписываются: проект, заказчик (с учётом
     публичности), документ, слайд. Широкие таблицы — построчно.
  4. Qwen (роль query) с системным промптом prompts/approach_system.txt.
  5. Постобработка: маркеры с несуществующим номером, числа без опоры во
     фрагментах, утверждения без маркера и без [вывод] — в блок «Не
     подтверждено»; список «Источники» собирается кодом по реально
     процитированным номерам (SPEC 5.3 п. 4), а не моделью.

Подзапросы граней пока строятся шаблонами строк; Jinja2-планы по типам задач
(SPEC 7) появятся вместе с классификатором запроса.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from bzpo.generation.tables import linearize_wide_tables
from bzpo.generation.validate import check_numbers
from bzpo.search.query import SearchHit

SYSTEM_PROMPT = Path(__file__).parent / "prompts" / "approach_system.txt"

# Грань -> (название для человека, что ищем). Порядок — порядок вывода в интерфейсе.
FACET_LABELS: dict[str, tuple[str, str]] = {
    "understanding": ("Понимание задачи",
                      "Цели, задачи и исходную ситуацию заказчиков в похожих проектах"),
    "methodology": ("Методология",
                    "Какими методами работали: интервью, опросы, оценки, бенчмаркинг"),
    "stages": ("Этапы", "Из каких этапов состояли проекты и в какой последовательности"),
    "artifacts": ("Артефакты", "Что передавали заказчику: отчёты, модели, стратегии, планы"),
    "risks": ("Риски", "Барьеры, ограничения и риски, с которыми сталкивались"),
    "outcome": ("Результат", "Выводы, рекомендации и достигнутый эффект"),
}

FACETS: dict[str, str] = {
    "understanding": "цели, задачи и исходная ситуация заказчика",
    "methodology": "методы и инструменты: интервью, опросы, оценка, бенчмаркинг",
    "stages": "этапы и последовательность работ проекта",
    "artifacts": "результаты и документы, переданные заказчику",
    "risks": "риски, барьеры и ограничения проекта",
    "outcome": "выводы, рекомендации и достигнутый результат",
}

_MARKER = re.compile(r"\[(\d+)\]")
# Модель пишет и «[вывод]», и «(вывод)» — оба варианта означают пометку переноса.
_INFERENCE = re.compile(r"[\[(]вывод[\])]", re.I)
_GAPS_HEADING = re.compile(r"^#+\s*\d*\.?\s*Пробелы", re.I)
_BASIS_HEADING = re.compile(r"^#+\s*\d*\.?\s*Опора", re.I)
_WORD = re.compile(r"[А-Яа-яЁё]{6,}")
# Слова, общие для любого описания проекта: совпадение по ним переносом не считается.
_GENERIC_STEMS = {"заказ", "компа", "проек", "прошл", "новог", "новый"}
# Доля основ строки из входных данных, начиная с которой строка — пересказ запроса.
_RESTATEMENT_SHARE = 0.6


@dataclass(frozen=True, slots=True)
class ApproachRequest:
    goal: str
    area: str | None = None
    situation: str | None = None

    def __post_init__(self) -> None:
        # Пустая цель даёт подзапросы вида «: риски…» — поиск отвечает мусором
        # с низкими оценками, а генерация всё равно идёт. Лучше упасть сразу.
        if not self.goal or not self.goal.strip():
            raise ValueError("цель проекта не задана")


@dataclass(frozen=True, slots=True)
class Fragment:
    number: int
    facet: str
    hit: SearchHit
    text: str  # то, что видит модель: метка + текст слайда

    @property
    def body(self) -> str:
        """Текст без метки — по нему проверяются числа ответа."""
        return self.text.split("\n", 1)[1] if "\n" in self.text else self.text


GENERAL = "general"  # утверждение без привязки к грани (цель, суть, пробелы)

# Раздел ответа -> грань: для строк без ссылок на фрагменты.
_SECTION_FACETS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Опора|Выбор", re.I), "understanding"),
    (re.compile(r"Этап", re.I), "stages"),
    (re.compile(r"Особенност", re.I), "methodology"),
    (re.compile(r"Риск", re.I), "risks"),
    (re.compile(r"Результат|контрольн", re.I), "outcome"),
)


@dataclass(frozen=True, slots=True)
class Issue:
    """Проблема, найденная проверкой, с гранью, к которой она относится."""

    kind: str  # number | no_source | transfer | bad_ref
    message: str
    facet: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class FacetScore:
    """Достоверность грани: опора поиска × чистота текста (эвристика, не вероятность).

    retrieval — средняя оценка реранкера фрагментов грани в контексте (0–1);
    grounding — доля утверждений грани, прошедших проверки (0–1).
    """

    facet: str
    retrieval: float | None
    claims: int
    issues: int

    @property
    def grounding(self) -> float | None:
        if self.claims == 0:
            return None
        return max(0.0, 1.0 - self.issues / self.claims)

    @property
    def confidence(self) -> int | None:
        if self.retrieval is None or self.grounding is None:
            return None
        return round(100 * self.retrieval * self.grounding)


@dataclass(slots=True)
class Section:
    text: str
    fragments: list[Fragment]
    cited: list[int]
    issues: list[Issue] = field(default_factory=list)
    claims_by_facet: dict[str, int] = field(default_factory=dict)

    @property
    def unverified(self) -> list[str]:
        return [issue.message for issue in self.issues]

    def scores(self) -> dict[str, FacetScore]:
        """Достоверность по граням в порядке FACETS (+ общая часть без грани)."""
        retrieval: dict[str, list[float]] = {}
        for fragment in self.fragments:
            retrieval.setdefault(fragment.facet, []).append(fragment.hit.rerank_score)
        issues: dict[str, int] = {}
        for issue in self.issues:
            issues[issue.facet] = issues.get(issue.facet, 0) + 1
        result = {}
        for facet in [*FACETS, GENERAL]:
            values = retrieval.get(facet)
            result[facet] = FacetScore(
                facet,
                sum(values) / len(values) if values else None,
                max(self.claims_by_facet.get(facet, 0), issues.get(facet, 0)),
                issues.get(facet, 0),
            )
        return result


def retrieval_confidence(hits: Sequence[SearchHit]) -> int | None:
    """Предварительная достоверность грани сразу после поиска — только опора поиска."""
    if not hits:
        return None
    return round(100 * sum(h.rerank_score for h in hits) / len(hits))


def facet_queries(request: ApproachRequest) -> dict[str, str]:
    subject = request.goal if not request.area else f"{request.goal} ({request.area})"
    return {facet: f"{subject}: {phrase}" for facet, phrase in FACETS.items()}


def assemble_context(
    hits_by_facet: Mapping[str, Sequence[SearchHit]],
    count_tokens: Callable[[str], int],
    budget: int,
) -> list[Fragment]:
    """Слить выдачу граней в нумерованные фрагменты в пределах бюджета токенов.

    Порядок — по кругу: первый фрагмент каждой грани, затем вторые и т. д.
    При нехватке бюджета отбрасываются хвосты граней, а не грань целиком.
    """
    queues = {facet: list(hits) for facet, hits in hits_by_facet.items()}
    seen: set[str] = set()
    fragments: list[Fragment] = []
    used = 0
    while any(queues.values()):
        for facet, queue in queues.items():
            while queue and queue[0].chunk_id in seen:
                queue.pop(0)
            if not queue:
                continue
            hit = queue.pop(0)
            seen.add(hit.chunk_id)
            text = _fragment_text(len(fragments) + 1, hit)
            cost = count_tokens(text)
            if used + cost > budget:
                continue
            fragments.append(Fragment(len(fragments) + 1, facet, hit, text))
            used += cost
    return fragments


BRIEF = (
    "\n\nКРАТКИЙ РЕЖИМ: тот же формат, но таблицы — не больше 3 строк, в разделах 2 и 4–7 "
    "по 1–2 пункта, суть подхода — 2 предложения. Ответ должен закончиться полностью."
)


def build_prompt(
    request: ApproachRequest, fragments: Sequence[Fragment], *, brief: bool = False
) -> str:
    situation = request.situation or "определи по материалам, что известно о похожих ситуациях"
    context = "\n\n".join(f.text for f in fragments)
    return (
        "ВХОДНЫЕ ДАННЫЕ\n"
        f"- Цель нового проекта: {request.goal}\n"
        f"- Область / отрасль: {request.area or 'не указана'}\n"
        f"- Ситуация нового заказчика: {situation}\n\n"
        f"ФРАГМЕНТЫ БАЗЫ ЗНАНИЙ\n{context}"
        + (BRIEF if brief else "")
    )


def postprocess(
    answer: str, fragments: Sequence[Fragment], request: ApproachRequest | None = None
) -> Section:
    """Проверить ответ модели против фрагментов (без LLM) и разнести итог по граням.

    Перенос: в разделе «Опора» (прошлые проекты) слова ситуации нового
    заказчика, которых нет в процитированном фрагменте, — модель приписала
    прошлому проекту обстоятельства нового. Строка, почти целиком
    пересказывающая входные данные, утверждением без источника не считается.

    Грань утверждения — та, по которой найдено большинство процитированных
    фрагментов; без ссылок — по разделу ответа (`_SECTION_FACETS`).
    """
    situation_stems = _stems(request.situation or "") - _GENERIC_STEMS if request else set()
    input_stems = (
        _stems(" ".join(filter(None, (request.goal, request.area, request.situation))))
        if request else set()
    )
    known = {f.number for f in fragments}
    bodies = {f.number: f.body for f in fragments}
    facet_of = {f.number: f.facet for f in fragments}
    issues: list[Issue] = []
    claims_by_facet: dict[str, int] = {}

    def facet_for(markers: Sequence[int], fallback: str) -> str:
        if not markers:
            return fallback
        counts: dict[str, int] = {}
        for n in markers:
            counts[facet_of[n]] = counts.get(facet_of[n], 0) + 1
        order = list(FACETS)
        return min(counts, key=lambda f: (-counts[f], order.index(f)))

    # Модель иногда дословно повторяет служебную строку промпта про источники.
    lines = [line for line in answer.strip().split("\n")
             if "список источников" not in line.lower()]
    cleaned: list[str] = []
    for line, section_facet in _with_sections(lines):
        def drop_unknown(match: re.Match[str], section_facet: str = section_facet) -> str:
            number = int(match.group(1))
            if number in known:
                return match.group(0)
            issues.append(Issue("bad_ref", f"ссылка [{number}] на несуществующий фрагмент "
                                           "удалена", section_facet))
            return ""

        cleaned.append(_MARKER.sub(drop_unknown, line))
    text = "\n".join(cleaned)

    for issue in check_numbers(text, bodies):
        if issue.cited:  # числа в [вывод] и без маркера ловит проверка ниже
            markers = [n for n in issue.cited if n in known]
            issues.append(Issue("number", f"{issue.reason}: «{_short(issue.sentence)}»",
                                facet_for(markers, GENERAL)))
    for line, in_basis, section_facet in _claims(text):
        markers = sorted({int(n) for n in _MARKER.findall(line)} & known)
        facet = facet_for(markers, section_facet)
        if not markers and not _INFERENCE.search(line):
            stems = _stems(line)
            if stems and len(stems & input_stems) / len(stems) >= _RESTATEMENT_SHARE:
                continue  # пересказ входных данных, а не утверждение о прошлом
            claims_by_facet[facet] = claims_by_facet.get(facet, 0) + 1
            issues.append(Issue("no_source", f"без источника и без [вывод]: «{_short(line)}»",
                                facet))
            continue
        claims_by_facet[facet] = claims_by_facet.get(facet, 0) + 1
        if in_basis and markers and situation_stems:
            cited_stems = _stems(" ".join(bodies[n] for n in markers))
            moved = sorted((_stems(line) & situation_stems) - cited_stems)
            if moved:
                issues.append(Issue(
                    "transfer",
                    f"возможен перенос ситуации нового заказчика в факт прошлого проекта "
                    f"(«{', '.join(moved)}…» нет во фрагментах {markers}): «{_short(line)}»",
                    facet,
                ))
    cited = sorted({int(n) for n in _MARKER.findall(text)} & known)
    return Section(text=text, fragments=list(fragments), cited=cited, issues=issues,
                   claims_by_facet=claims_by_facet)


def render(section: Section) -> str:
    by_number = {f.number: f for f in section.fragments}
    lines = [section.text, "", "### Источники"]
    documents: dict[tuple[str, str | None, str | None], list[str]] = {}
    for n in section.cited:
        hit = by_number[n].hit
        key = (hit.chunk.doc_title, hit.chunk.project, hit.customer)
        documents.setdefault(key, []).append(f"[{n}] {hit.chunk.locator}")
    for (title, project, customer), refs in documents.items():
        lines.append(f"- {title} — проект «{project}», заказчик: {customer}: {'; '.join(refs)}")
    if not section.cited:
        lines.append("— (ни одной ссылки на фрагменты)")
    lines += ["", "### Достоверность по граням",
              "| Грань | Достоверность | Опора поиска | Утверждений | Проблем |",
              "|---|---|---|---|---|"]
    for facet, score in section.scores().items():
        if facet == GENERAL and score.claims == 0:
            continue
        confidence = f"{score.confidence}%" if score.confidence is not None else "—"
        retrieval = f"{score.retrieval:.2f}" if score.retrieval is not None else "—"
        lines.append(f"| {facet_label(facet)} | {confidence} | {retrieval} | "
                     f"{score.claims} | {score.issues} |")
    if section.issues:
        lines += ["", "### Не подтверждено (проверить вручную)"]
        for facet in [*FACETS, GENERAL]:
            group = [i for i in section.issues if i.facet == facet]
            if group:
                lines += ["", f"#### {facet_label(facet)} ({len(group)})"]
                lines += [f"- {issue.message}" for issue in group]
    return "\n".join(lines)


def facet_label(facet: str) -> str:
    return FACET_LABELS[facet][0] if facet in FACET_LABELS else "Общее (цель, суть, выводы)"


# --------------------------------------------------------------- детали


def _fragment_text(number: int, hit: SearchHit) -> str:
    c = hit.chunk
    label = (f"[{number}] Проект «{c.project}»; заказчик: {hit.customer}; "
             f"{c.doc_title}, {c.locator}; раздел: {c.section_title}")
    return f"{label}\n{linearize_wide_tables(c.text_raw)}"


def _with_sections(lines: Sequence[str]) -> list[tuple[str, str]]:
    """Каждой строке — грань её раздела (до первого раздела — общая часть)."""
    result = []
    facet = GENERAL
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            facet = next((f for rx, f in _SECTION_FACETS if rx.search(stripped)), GENERAL)
        result.append((line, facet))
    return result


def _claims(text: str) -> list[tuple[str, bool, str]]:
    """Строки-утверждения: (строка, раздел «Опора»?, грань раздела).

    Без заголовков, шапок таблиц, коротких меток и вопросов-пробелов.
    """
    claims = []
    in_gaps = in_basis = False
    for raw, section_facet in _with_sections(text.split("\n")):
        line = raw.strip()
        if line.startswith("#"):
            in_gaps = bool(_GAPS_HEADING.match(line))
            in_basis = bool(_BASIS_HEADING.match(line))
            continue
        if in_gaps or line.rstrip(" .)").endswith("?"):
            continue  # вопросы к заказчику источника не требуют
        if not line or re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")] if line.startswith("|") else []
        if cells and any(c in ("№", "Этап", "Проект (заказчик)", "Источник") for c in cells):
            continue  # шапка таблицы
        if line.rstrip("*_ ").endswith(":"):
            continue  # подпись поля («**Суть подхода:**»), текст — на следующей строке
        if len(re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", line)) < 4:
            continue
        claims.append((line, in_basis, section_facet))
    return claims


def _stems(text: str) -> set[str]:
    """Грубые основы слов от 6 букв: «аренды», «аренду» -> «аренд»."""
    return {w.lower()[:5] for w in _WORD.findall(text)}


def _short(text: str, limit: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
