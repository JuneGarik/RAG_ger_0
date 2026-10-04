"""Проход мультимодальной модели по слайдам, которые код прочитать не может (П2).

На вход — картинка слайда и уже извлечённый из XML текст. Модель (роль vision,
Qwen3-VL instruct) возвращает JSON:

- key_message, visual — интерпретация: идут в поиск, не цитируются;
- image_text — текст, видимый только на картинках (скриншоты, схемы-рисунки);
  помечается как распознанный моделью;
Подписи диаграмм и значения тепловых карт модель НЕ определяет: на них она
ошибалась (путала ряды легенды, читала 2 ячейки из 12), их ставит код по XML
(parsing/charts.py, parsing/heatmap.py).

Ответ проверяется кодом: числа в «посыле» и «визуале», которых нет на слайде,
попадают в замечания; текст с картинки, повторяющий извлечённый, отбрасывается.

Ответы кэшируются по картинке, версии промпта и модели: повторная
конвертация модель не гоняет.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bzpo.parsing.pptx import SlideData

PROMPT_VERSION = "vlm-slide-3"
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

PROMPT = """Это слайд консалтингового отчёта. Ниже — текст, уже извлечённый из файла
слайда программой (он точный: тексты, таблицы, данные и подписи диаграмм).
Посмотри на картинку слайда и верни ТОЛЬКО JSON:

{{
  "key_message": "главный вывод слайда одним-двумя предложениями",
  "visual": "как устроена графика: схемы, стрелки, выделения цветом, что с чем связано",
  "image_text": "текст с картинки, которого НЕТ в извлечённом тексте; иначе пустая строка"
}}

Правила:
- Пиши по-русски. Числа и названия переноси точно, ничего не придумывай.
- В image_text не повторяй извлечённый текст.

Извлечённый текст слайда:
{text}
"""


@dataclass(slots=True)
class SlideVision:
    slide: int
    key_message: str = ""
    visual: str = ""
    image_text: str = ""
    issues: list[str] = field(default_factory=list)
    cached: bool = False


def analyze(
    slide: SlideData, image: Path, client: Any, cache_dir: Path, model: str
) -> SlideVision:
    """Разобрать один слайд: ответ модели из кэша или новый, затем проверка."""
    picture = image.read_bytes()
    text = "\n\n".join([slide.title, *slide.blocks])
    key = hashlib.sha256(picture + PROMPT_VERSION.encode() + model.encode()).hexdigest()[:24]
    cache = cache_dir / f"{key}.json"
    cached = cache.exists()
    if cached:
        raw = cache.read_text(encoding="utf-8")
    else:
        prompt = PROMPT.format(text=text[:6000])
        result = client.generate(prompt, images=[base64.b64encode(picture).decode()],
                                 json_format=True, num_predict=900)
        raw = result.text
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(raw, encoding="utf-8")
    vision = _verify(slide, _parse(raw), text)
    vision.cached = cached
    return vision


def apply(slide: SlideData, vision: SlideVision) -> None:
    """Вписать результат в слайд: поля интерпретации, текст с картинок, подписи диаграммы."""
    slide.key_message = vision.key_message or None
    slide.visual = vision.visual or None
    if vision.image_text:
        slide.blocks.append(f"**Текст с изображения (распознан моделью, проверить):** "
                            f"{vision.image_text}")


# --------------------------------------------------------------- детали


def _parse(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            data = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            data = {}
    return data if isinstance(data, dict) else {}


def _verify(slide: SlideData, data: dict[str, Any], text: str) -> SlideVision:
    vision = SlideVision(slide.number)
    if not data:
        vision.issues.append("модель не вернула разбираемый JSON")
        return vision
    vision.key_message = _string(data.get("key_message"))
    vision.visual = _string(data.get("visual"))
    image_text = _string(data.get("image_text"))
    # Повтор уже извлечённого текста — не новая информация; одно-два слова с
    # фотографии или логотипа («COMPANY») — шум, а не содержание слайда.
    words = re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", image_text)
    if image_text and len(words) >= 3 and _share_in(image_text, text) < 0.8:
        vision.image_text = image_text

    slide_numbers = {n.replace(",", ".") for n in _NUMBER.findall(text)}
    for name, value in (("ключевой посыл", vision.key_message), ("визуал", vision.visual)):
        foreign = sorted({n.replace(",", ".") for n in _NUMBER.findall(value)} - slide_numbers)
        foreign = [n for n in foreign if not (n.isdigit() and int(n) <= 12)]
        if foreign:
            vision.issues.append(f"{name}: чисел {', '.join(foreign)} нет на слайде")

    return vision








def _string(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def _share_in(fragment: str, text: str) -> float:
    words = {w.lower() for w in re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", fragment)}
    if not words:
        return 1.0
    present = {w.lower() for w in re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", text)}
    return len(words & present) / len(words)
