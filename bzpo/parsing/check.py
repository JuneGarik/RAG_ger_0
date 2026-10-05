"""Проверка md против исходного pptx (docs/PARSING_PLAN.md, раздел 6).

Читает XML pptx напрямую — независимо от парсера, поэтому годится и для md,
сделанного вручную или моделью. По каждому слайду:

- полнота текста: доля абзацев XML, чьи слова (от 3 букв) есть в блоке слайда
  md не меньше чем на 80%;
- числа без опоры: число в md, которого нет ни в тексте слайда, ни в данных
  его диаграмм (с округлениями). Поля интерпретации («Ключевой посыл»,
  «Визуал», «Тип слайда») и ссылки на другие слайды не проверяются;
- картинки: сколько изображений на слайде (их содержимое XML не даёт).

Числа в XML бывают разбиты на куски («571 3» + «59»): куски абзаца
склеиваются как есть, без разделителей.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

_A_T = re.compile(r"<a:t>([^<]*)</a:t>")
_PARAGRAPH = re.compile(r"<a:p>.*?</a:p>|<a:p .*?</a:p>", re.S)
_C_V = re.compile(r"<c:v>([^<]*)</c:v>")
_SLIDE_BLOCK = re.compile(r"<!-- slide: (\d+) -->(.*?)(?=<!-- slide: \d+ -->|\Z)", re.S)
_NUMBER = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")
_INTERPRETATION = re.compile(
    r"^\*\*(Ключевой посыл|Визуал|Тип слайда|Раздел|Подраздел|Источник значений):\*\*.*$", re.M
)
# Текст, распознанный моделью с картинки, в XML отсутствует по определению:
# его числа не «без опоры», а помечены как распознанные и проверяются человеком.
_IMAGE_TEXT = re.compile(r"^\*\*Текст с изображения[^\n]*$", re.M)
_SLIDE_REF = re.compile(r"\[Слайд \d+\]|слайд[а-яё]*\s+\d+(?:\s*[–-]\s*\d+)?|<!--.*?-->", re.I)
# Не данные: адреса ссылок, цвета ячеек тепловых карт («■ #54CFD6»), подписи
# картинок (имя фигуры и alt-текст, а не текст слайда).
_NOT_DATA = re.compile(
    r"\]\([^)]*\)|https?://\S+|#[0-9A-Fa-f]{6}\b|\*\[Изображение:[^\]]*\]\*"
)

COVERAGE_MIN = 0.95  # критерий приёмки по документу
_PARAGRAPH_COVERED = 0.8


@dataclass(slots=True)
class SlideCheck:
    slide: int
    paragraphs: int
    covered: int
    unsupported: list[str] = field(default_factory=list)
    missing_paragraphs: list[str] = field(default_factory=list)
    images: int = 0

    @property
    def coverage(self) -> float:
        return self.covered / self.paragraphs if self.paragraphs else 1.0


@dataclass(slots=True)
class DeckCheck:
    slides: list[SlideCheck]
    problems: list[str] = field(default_factory=list)  # структура: число слайдов и т. п.

    @property
    def coverage(self) -> float:
        total = sum(s.paragraphs for s in self.slides)
        return sum(s.covered for s in self.slides) / total if total else 1.0

    @property
    def unsupported(self) -> int:
        return sum(len(s.unsupported) for s in self.slides)

    @property
    def passed(self) -> bool:
        return not self.problems and self.coverage >= COVERAGE_MIN and self.unsupported == 0


def check(pptx: Path, markdown: str) -> DeckCheck:
    xml_slides = _xml_slides(pptx)
    blocks = {int(n): body for n, body in _SLIDE_BLOCK.findall(markdown)}
    result = DeckCheck(slides=[])
    if len(blocks) != len(xml_slides):
        result.problems.append(f"слайдов в md {len(blocks)}, в pptx {len(xml_slides)}")
    for number, (paragraphs, values, images) in enumerate(xml_slides, start=1):
        block = blocks.get(number)
        if block is None:
            result.problems.append(f"нет слайда {number} в md")
            result.slides.append(SlideCheck(number, len(paragraphs), 0, images=images))
            continue
        result.slides.append(_check_slide(number, paragraphs, values, images, block))
    return result


def _check_slide(
    number: int, paragraphs: list[str], values: list[str], images: int, block: str
) -> SlideCheck:
    md_words = {w.lower() for w in _WORD.findall(block)}
    covered = 0
    missing = []
    for paragraph in paragraphs:
        words = {w.lower() for w in _WORD.findall(paragraph)}
        if not words or len(words & md_words) / len(words) >= _PARAGRAPH_COVERED:
            covered += 1
        else:
            missing.append(paragraph)

    allowed = set()
    for paragraph in paragraphs:
        allowed |= _numbers(paragraph) | _numbers(paragraph.replace(" ", ""))
    for value in values:
        allowed |= _numbers(value) | _rounded(value)
    checked = _NOT_DATA.sub(" ", _SLIDE_REF.sub(" ", _INTERPRETATION.sub(" ", block)))
    checked = _IMAGE_TEXT.sub(" ", checked)
    unsupported = sorted(n for n in _numbers(checked) - allowed if not _trivial(n))
    return SlideCheck(number, len(paragraphs), covered, unsupported, missing, images)


def _xml_slides(pptx: Path) -> list[tuple[list[str], list[str], int]]:
    """По слайдам в порядке показа: абзацы текста, значения диаграмм, число картинок."""
    with zipfile.ZipFile(pptx) as z:
        rels = z.read("ppt/_rels/presentation.xml.rels").decode("utf-8")
        targets = {
            m.group(1): m.group(2)
            for m in re.finditer(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels)
        } | {
            m.group(2): m.group(1)
            for m in re.finditer(r'Target="([^"]+)"[^>]*Id="([^"]+)"', rels)
        }
        order = re.findall(r'<p:sldId [^>]*r:id="([^"]+)"',
                           z.read("ppt/presentation.xml").decode("utf-8"))
        result = []
        for rid in order:
            name = targets[rid].split("/")[-1]
            xml = z.read(f"ppt/slides/{name}").decode("utf-8")
            # Куски абзаца склеиваются как есть — так текст выглядит на слайде:
            # «571 3» + «59» -> «571 359», «72» + «,» + «75» -> «72,75».
            paragraphs = ["".join(_A_T.findall(p)) for p in _PARAGRAPH.findall(xml)]
            paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip()]
            rel_xml = z.read(f"ppt/slides/_rels/{name}.rels").decode("utf-8")
            values = []
            for chart in re.findall(r'Target="\.\./charts/(chart\d+\.xml)"', rel_xml):
                chart_xml = z.read(f"ppt/charts/{chart}").decode("utf-8")
                values += _C_V.findall(chart_xml) + _A_T.findall(chart_xml)
            # Заметки докладчика конвертер переносит в md («> **Заметки докладчика:**») —
            # их числа законны; в полноту слайда заметки не входят.
            for notes in re.findall(r'Target="\.\./notesSlides/(notesSlide\d+\.xml)"', rel_xml):
                notes_xml = z.read(f"ppt/notesSlides/{notes}").decode("utf-8")
                values += ["".join(_A_T.findall(p)) for p in _PARAGRAPH.findall(notes_xml)]
            images = xml.count("<p:pic>")
            result.append((paragraphs, values, images))
        return result


def _numbers(text: str) -> set[str]:
    result = set()
    for raw in _NUMBER.findall(text.replace(" ", " ").replace(" ", " ")):
        value = re.sub(r"\s", "", raw).replace(",", ".")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        integer, _, fraction = value.partition(".")
        integer = integer.lstrip("0") or "0"
        result.add(f"{integer}.{fraction}" if fraction else integer)
    return result


def _rounded(value: str) -> set[str]:
    """Как значение диаграммы может быть записано в md: округления, проценты, тысячи."""
    try:
        # 45765.499999999993 в XML — это 45765,5, записанное с шумом двоичной дроби.
        number = Decimal(value).quantize(Decimal("0.000001"))
    except InvalidOperation:
        return set()
    forms = set()
    for scaled in (number, number * 100, number / 1000):
        for digits in range(5):
            # Округление «как в отчётах» — половина вверх: 45765,4999… -> 45765, 45765,5 -> 45766.
            for mode in (ROUND_HALF_UP, ROUND_DOWN):
                rounded = scaled.quantize(Decimal(1).scaleb(-digits), rounding=mode)
                forms |= _numbers(f"{rounded:f}")
    return forms


def _trivial(number: str) -> bool:
    """Целые до 12 — нумерация пунктов, этапов и разделов, а не данные."""
    return "." not in number and int(number) <= 12
