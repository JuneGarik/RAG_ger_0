"""Проверка чисел в сгенерированном разделе против цитируемых фрагментов.

Правило: каждое число в предложении должно встречаться во фрагментах, на
которые это предложение ссылается маркерами [N]. Число с маркером, которого
нет в процитированных фрагментах, — выдумано или взято не из того источника;
число в предложении без маркера проверить не по чему.

Чего проверка НЕ ловит: перепутанные ячейки одной таблицы — неверное
значение, которое во фрагменте всё же есть, только в другой ячейке. Это
закрывается подачей таблицы (`tables.linearize_wide_tables`), а не здесь.

Целые от 0 до 10 не проверяются: количества в тексте пишут и словами
(«восемь критериев»), и сверка давала бы ложные срабатывания на каждом
абзаце. Годы, проценты, дроби и крупные числа проверяются всегда.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

_MARKER = re.compile(r"\[(\d+)\]")
# «1 470», «94 405,5», «2,1», «12», «0.36»; пробелы-разделители разрядов —
# обычный, неразрывный и узкий неразрывный.
_NUMBER = re.compile(r"(?<![\w,.])\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
_NUM = r"\d+(?:[.,]\d+)?"
# Диапазон «2,2–2,3» и значение со знаком «≥2,4» сверяются целиком: «2,3–2,4»
# из чисел 2,3 и 2,4, которые во фрагменте есть по отдельности, — искажение.
_COMPOUND = re.compile(rf"(?:[≤≥<>]\s*{_NUM})|(?:{_NUM}\s*[–—-]\s*{_NUM})")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\"(*-])|\n+")


@dataclass(frozen=True, slots=True)
class NumberIssue:
    sentence: str
    number: str
    cited: tuple[int, ...]  # пусто — у предложения нет маркера

    @property
    def reason(self) -> str:
        if not self.cited:
            return f"число {self.number} в предложении без ссылки на источник"
        refs = ", ".join(f"[{n}]" for n in self.cited)
        return f"числа {self.number} нет во фрагментах {refs}"


def check_numbers(answer: str, fragments: Mapping[int, str]) -> list[NumberIssue]:
    """Найти числа ответа, не подтверждённые процитированными фрагментами.

    `fragments` — текст фрагментов по их номерам [N] в промпте.
    """
    # Во фрагменте допустимы и составные значения целиком, и отдельные числа
    # (в т. ч. концы диапазонов): «выручка 128 650» можно взять из таблицы.
    allowed_by_fragment = {
        n: _compounds(text) | _numbers(text) for n, text in fragments.items()
    }
    issues: list[NumberIssue] = []
    for sentence in _sentences(answer):
        cited = tuple(sorted({int(n) for n in _MARKER.findall(sentence)}))
        allowed: set[str] = set()
        for n in cited:
            allowed |= allowed_by_fragment.get(n, set())
        body = _MARKER.sub(" ", sentence)
        compounds = _compounds(body)
        # Числа внутри составного значения проверяются в его составе, не поодиночке.
        rest = _numbers(_COMPOUND.sub(" ", body))
        for value in sorted((compounds | rest) - allowed):
            issues.append(NumberIssue(sentence=sentence.strip(), number=value, cited=cited))
    return issues


def _compounds(text: str) -> set[str]:
    result: set[str] = set()
    for raw in _COMPOUND.findall(text):
        value = re.sub(r"\s", "", raw).replace(",", ".").replace("—", "–").replace("-", "–")
        result.add(value)
    return result


def _sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_END.split(text) if part.strip()]


def _numbers(text: str) -> set[str]:
    result: set[str] = set()
    for raw in _NUMBER.findall(text):
        value = re.sub(r"[   ]", "", raw).replace(",", ".")
        if "." not in value and int(value) <= 10:
            continue
        # 2.10 и 2.1 — одно число; 045 и 45 — тоже.
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        result.add(value.lstrip("0") or "0")
    return result
