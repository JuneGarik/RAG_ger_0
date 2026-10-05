"""Черновик строки реестра по документу нового проекта: LLM заполняет, человек проверяет.

Последовательность:
  1. `select_context` — из md презентации берутся «паспортные» слайды (титул,
     сведения об исполнителе и специалистах, основание, резюме, контакты) и
     заголовки всех слайдов. Весь документ в окно не помещается и не нужен.
  2. `build_prompt` — LLM (роль enrichment, GigaChat Lightning) возвращает JSON:
     по каждому полю значение и номера слайдов, откуда оно взято.
  3. `verify` — детерминированная проверка: слайды есть в контексте, фамилии
     встречаются на процитированных слайдах, годы, даты и числа — тоже.
     Непрошедшее поле не выбрасывается, а помечается: решает человек.

LLM не заполняет то, что решает человек: код проекта, уровень публичности,
результат тендера (для отчёта он выводится правилом, см. `tender_result_for`).
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from bzpo.generation.validate import check_numbers
from bzpo.ingest.slides import SlideDeck

Kind = Literal["text", "names", "year", "date", "enum"]

# Поле реестра -> (что извлечь, тип проверки). Имена — как в колонках реестра.
FIELDS: dict[str, tuple[str, Kind]] = {
    "Год": ("год сдачи результата (число)", "year"),
    "Тип документа": (
        "report — отчёт по выполненному проекту (в том числе оформленный презентацией); "
        "kp — коммерческое предложение; act, contract, letter; presentation — прочие "
        "презентации, не отчёт и не КП",
        "enum",
    ),
    "Основная отрасль": ("отрасль заказчика, 1–3 слова", "text"),
    "Основная подотрасль": ("подотрасль заказчика, 1–4 слова", "text"),
    "Основное функциональное направление": ("предмет работ, 1–4 слова", "text"),
    "Клиент": ("заказчик: группа компаний или бренд", "text"),
    "Обзелич.название клиента": (
        "описание заказчика без названия, 3–7 слов, например «крупный девелопер Москвы»",
        "text",
    ),
    "Название": ("название проекта или документа с титула", "text"),
    "Описание проекта": ("что сделано, 1–3 предложения, только факты", "text"),
    "Подход": ("методы работы: интервью, опросы, анализ, бенчмарки; 1–2 предложения", "text"),
    "Key findings": ("главные выводы с цифрами, 2–3 предложения", "text"),
    "Страна": ("страна заказчика", "text"),
    "Практика": (
        "подразделения исполнителя, выполнявшие проект, — из должностей специалистов "
        "(«Партнер практики „X“» -> X); через «; »",
        "text",
    ),
    "Курирующий партнер": ("ФИО всех, у кого в должности «Партнер», через «; »", "names"),
    "Директор": ("ФИО всех, у кого в должности «Директор», через «; »", "names"),
    "Менеджер проекта": (
        "ФИО всех, у кого в должности «Руководитель проектов» или «Менеджер», через «; »",
        "names",
    ),
    "Команда": ("ФИО остальных специалистов (консультанты, аналитики), через «; »", "names"),
    "Дата начала": ("дата начала работ или договора, ГГГГ-ММ-ДД", "date"),
    "Дата окончания": ("дата окончания или отчётная дата, ГГГГ-ММ-ДД", "date"),
    "Услуга": ("вид услуги, 1–3 слова", "text"),
    "Субуслуга": ("конкретная услуга, 2–5 слов", "text"),
    "Отрасль": ("как «Основная отрасль»", "text"),
    "Название проекта (по договору)": ("предмет договора, если указан", "text"),
    "Клиент (по договору)": ("юрлицо заказчика по договору", "names"),
    "Описание клиента": ("кто заказчик: масштаб, сегменты, география; 1–3 предложения", "text"),
}
DOC_TYPES = ("kp", "report", "act", "contract", "letter", "presentation")
# Поля, которые по определению сочиняются, а не извлекаются: источник не обязателен.
_GENERATIVE = {"Обзелич.название клиента"}

_PASSPORT = re.compile(
    r"титул|резюме|итог|вывод|сведения|специалист|основани|контакт|подпис|о компании|"
    r"о заказчике|содержание результата|команда проекта",
    re.I,
)
_SKIP = re.compile(r"термин|сокращени|глоссари|список приложений|содержание отчета", re.I)


@dataclass(frozen=True, slots=True)
class DraftContext:
    text: str
    slides: dict[int, str]  # номер слайда -> полный текст, показанный модели
    titles: dict[int, str]  # заголовки всех слайдов: у этих отчётов заголовок — вывод с цифрами

    def source(self, slide: int) -> str | None:
        """Что модель видела на слайде: полный текст или хотя бы заголовок."""
        return self.slides.get(slide) or self.titles.get(slide)


@dataclass(slots=True)
class DraftField:
    name: str
    value: str | int | None
    slides: list[int]
    problems: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return not self.problems


def select_context(
    deck: SlideDeck, count_tokens: Callable[[str], int], budget: int = 9000
) -> DraftContext:
    """Паспортные слайды целиком + заголовки всех слайдов, в пределах бюджета."""
    chunks = deck.chunks
    titles = "\n".join(f"{c.slide}. {c.title}" for c in chunks)
    meta = "\n".join(f"{k}: {v}" for k, v in deck.meta.items() if k != "chunking")
    used = count_tokens(titles) + count_tokens(meta)

    passport = [c for c in chunks if _PASSPORT.search(f"{c.title} {c.slide_type or ''}")]
    head_tail = chunks[:8] + chunks[-2:]
    picked: dict[int, str] = {}
    for chunk in [*passport, *head_tail]:
        if chunk.slide in picked or _SKIP.search(chunk.title):
            continue
        cost = count_tokens(chunk.text_raw)
        if used + cost > budget:
            continue
        picked[chunk.slide] = chunk.text_raw
        used += cost

    body = "\n\n".join(f"=== Слайд {n} ===\n{picked[n]}" for n in sorted(picked))
    text = (
        f"Метаданные файла:\n{meta}\n\nЗаголовки всех слайдов:\n{titles}\n\n"
        f"Полный текст ключевых слайдов:\n{body}"
    )
    return DraftContext(text=text, slides=picked, titles={c.slide: c.title for c in chunks})


def build_prompt(context: DraftContext) -> str:
    fields = "\n".join(f"- «{name}»: {hint}" for name, (hint, _) in FIELDS.items())
    return (
        "Заполни карточку проекта для реестра консалтинговой компании по документу ниже.\n"
        "Правила:\n"
        "1. Только факты из документа. Нет сведений — value: null. Не угадывай.\n"
        "2. Для каждого поля укажи номера слайдов, где стоит этот факт.\n"
        "3. Фамилии, названия, даты и числа переписывай точно как в документе.\n"
        "4. Даты — в формате ГГГГ-ММ-ДД.\n\n"
        f"Поля:\n{fields}\n\n"
        'Ответ — только JSON: {"<поле>": {"value": <значение или null>, "slides": [<номера>]}, ...}'
        f"\n\nДокумент:\n{context.text}"
    )


def parse(raw: str) -> dict[str, DraftField]:
    """Разобрать ответ модели. Неизвестные поля отбрасываются, недостающие — пустые."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ответ модели — не JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("ответ модели — не объект JSON")

    # Модель называет поля по-своему: «год» вместо «Год», «обобщ.название клиента»
    # вместо «Обзелич.название клиента» — сопоставление без регистра и знаков,
    # для опечаток — ближайшее имя (не ниже 0,8 по difflib).
    by_key = {_field_key(k): v for k, v in data.items()}
    result: dict[str, DraftField] = {}
    for name in FIELDS:
        key = _field_key(name)
        if key not in by_key:
            close = difflib.get_close_matches(key, list(by_key), n=1, cutoff=0.8)
            key = close[0] if close else key
        item = by_key.get(key)
        value: str | int | None = None
        slides: list[int] = []
        if isinstance(item, dict):
            raw_value = item.get("value")
            if isinstance(raw_value, int | str) and str(raw_value).strip() \
                    and str(raw_value).strip().lower() not in _EMPTY_VALUES:
                value = raw_value.strip() if isinstance(raw_value, str) else raw_value
            raw_slides = item.get("slides") or []
            if isinstance(raw_slides, list):
                slides = [int(s) for s in raw_slides if str(s).isdigit()]
        result[name] = DraftField(name=name, value=value, slides=slides)
    return result


def verify(fields: dict[str, DraftField], context: DraftContext) -> dict[str, DraftField]:
    """Проверить каждое заполненное поле по процитированным слайдам (без LLM)."""
    for draft in fields.values():
        if draft.value is None:
            continue
        kind = FIELDS[draft.name][1]
        foreign = [s for s in draft.slides if context.source(s) is None]
        if foreign:
            draft.problems.append(f"слайдов {foreign} нет в документе")
        cited = "\n".join(src for s in draft.slides if (src := context.source(s)))
        if not cited and draft.name not in _GENERATIVE:
            draft.problems.append("нет подтверждающего слайда")
            continue
        value = str(draft.value)
        if kind == "enum":
            if value not in DOC_TYPES:
                draft.problems.append(f"«{value}» не из списка {', '.join(DOC_TYPES)}")
            guess = guess_doc_type(" ".join(context.slides.values())
                                   + " " + " ".join(context.titles.values()))
            if guess and value != guess:
                # Явный перевес признаков в тексте надёжнее ответа модели: от типа
                # зависит результат тендера (отчёт -> won), ошибка здесь дорогая.
                draft.problems.append(f"модель указала «{value}», по признакам в тексте — "
                                      f"«{guess}»; поставлено «{guess}»")
                draft.value = guess
        elif kind == "names":
            missing = [n for n in _name_words(value) if not _mentions(cited, n)]
            if missing:
                draft.problems.append(f"на слайдах {draft.slides} нет: {', '.join(missing)}")
        elif kind == "year":
            if not re.fullmatch(r"(19|20)\d\d", value) or value not in cited:
                draft.problems.append(f"года {value} нет на слайдах {draft.slides}")
        elif kind == "date":
            problem = _check_date(value, cited)
            if problem:
                draft.problems.append(problem)
        elif draft.name not in _GENERATIVE:
            for issue in check_numbers(f"{value} [1]", {1: cited}):
                draft.problems.append(f"числа {issue.number} нет на слайдах {draft.slides}")
    return fields


# Должность -> поле реестра; порядок важен: «Руководитель проектов практики» не директор.
_ROLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"партн[её]р", re.I), "Курирующий партнер"),
    (re.compile(r"руководител[ья] проект|менеджер проект", re.I), "Менеджер проекта"),
    (re.compile(r"директор", re.I), "Директор"),
)
_ROLE_FIELDS = ("Курирующий партнер", "Директор", "Менеджер проекта", "Команда", "Практика")


def apply_team_table(
    fields: dict[str, DraftField], context: DraftContext
) -> dict[str, DraftField]:
    """Роли и практики — правилом по таблице «ФИО | Должность», а не моделью.

    Распределить людей по ролям LLM на 1.8B активных параметров не может
    надёжно (путает партнёра с директором), а таблица однозначна. Нет таблицы —
    остаются ответы модели с пометкой «проверить».
    """
    for slide, text in sorted(context.slides.items()):
        people = _team_rows(text)
        if not people:
            continue
        by_role: dict[str, list[str]] = {name: [] for name in _ROLE_FIELDS}
        practices: list[str] = []
        for person, position in people:
            role = next((f for rx, f in _ROLE_RULES if rx.search(position)), "Команда")
            by_role[role].append(person)
            practice = re.search(r"практики\s+[«\"]([^»\"]+)[»\"]", position)
            if practice and practice.group(1) not in practices:
                practices.append(practice.group(1))
        by_role["Практика"] = practices
        for name in _ROLE_FIELDS:
            value = "; ".join(by_role[name]) or None
            fields[name] = DraftField(name=name, value=value, slides=[slide] if value else [])
        break
    return fields


def _team_rows(text: str) -> list[tuple[str, str]]:
    lines = [line for line in text.split("\n") if line.startswith("|")]
    if len(lines) < 3:
        return []
    header = [c.strip().lower() for c in lines[0].strip("|").split("|")]
    name_col = next((i for i, c in enumerate(header) if "ф. и. о." in c or "фио" in c), None)
    role_col = next((i for i, c in enumerate(header) if "должност" in c), None)
    if name_col is None or role_col is None:
        return []
    rows = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > max(name_col, role_col) and cells[name_col]:
            rows.append((cells[name_col], cells[role_col]))
    return rows


_KP_SIGNALS = re.compile(
    r"коммерческ\w* предложени|предлагаем|стоимост\w* (работ|услуг)|сопроводительное письмо|"
    r"срок\w* (выполнения|оказания)|мы понимаем, что", re.I)
_REPORT_SIGNALS = re.compile(
    r"\bотч[её]т\b|результат\w* (услуг|работ)|основание для оказания услуг|"
    r"сведения об исполнителе", re.I)


def guess_doc_type(text: str) -> str | None:
    """Отчёт или КП по перевесу признаков в тексте; None — признаков мало или они спорят.

    Признаки встречаются в обоих типах (КП обещает «отчёт», отчёт упоминает
    «стоимость работ»), поэтому решает перевес не меньше чем втрое: на «ОЦО
    Ритейл» 8 признаков КП против 2, на отчёте ДЕВ 30 признаков отчёта против 3.
    """
    kp = len(_KP_SIGNALS.findall(text))
    report = len(_REPORT_SIGNALS.findall(text))
    if kp >= 3 and kp >= 3 * report:
        return "kp"
    if report >= 3 and report >= 3 * kp:
        return "report"
    return None


def tender_result_for(doc_types: Iterable[str]) -> str | None:
    """Отчёт или акт — работа выполнена, тендер выигран. Только КП — решает человек."""
    return "won" if {"report", "act"} & set(doc_types) else None


# --------------------------------------------------------------- детали


# Модель пишет «нет данных» строкой вместо null.
_EMPTY_VALUES = {"null", "none", "нет", "нет данных", "не указано", "-", "—", "н/д"}


def _field_key(name: str) -> str:
    return re.sub(r"[\W_]+", "", str(name).lower().replace("ё", "е"))


def _name_words(value: str) -> list[str]:
    """Все слова ФИО (не только фамилия): опечатка в отчестве — тоже ошибка."""
    return re.findall(r"[А-ЯЁA-Z][а-яёa-z-]{2,}", value)


def _mentions(text: str, surname: str) -> bool:
    # Основа без окончания: «Иванов» найдёт и «Иванова», и «Иванову».
    stem = surname[: max(4, len(surname) - 2)].lower()
    return stem in text.lower()


def _check_date(value: str, cited: str) -> str | None:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return f"«{value}» — не дата ГГГГ-ММ-ДД"
    forms = (parsed.isoformat(), parsed.strftime("%d.%m.%Y"), parsed.strftime("%d.%m.%y"))
    if any(form in cited for form in forms):
        return None
    months = ("января февраля марта апреля мая июня июля августа сентября октября ноября "
              "декабря").split()
    spelled = f"{parsed.day} {months[parsed.month - 1]} {parsed.year}"
    if spelled in cited:
        return None
    return f"даты {parsed.strftime('%d.%m.%Y')} нет на процитированных слайдах"
