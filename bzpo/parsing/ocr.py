"""Распознавание текста на рисунках документа (скриншоты слайдов в Word), роль ocr.

Нужен текст слайда дословно, а не пересказ. Сравнение на скриншотах отчёта НГК
(рисунок 49): специализированная GLM-OCR давала смешанные слова и опечатки
(«архитектURE», «пrozрачной», «инциматив») и зацикливалась на титульном слайде,
а Qwen3-VL с промптом дословной транскрипции — чистый текст без ошибок за то же
время. Поэтому по умолчанию роль ocr — Qwen3-VL instruct (config: model_ocr).

XML у скриншота нет, сверить распознанное не с чем — в md такой текст
помечается «распознан OCR — проверить». Повтор строки, уже встречавшейся выше,
отбрасывается: защита от зацикливания модели. Кэш — по картинке, модели и
версии промпта.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path
from typing import Any

PROMPT_VERSION = "ocr-2"
PROMPT = ("Перепиши ДОСЛОВНО весь текст со слайда на картинке в порядке чтения, по строкам. "
          "Таблицы — в Markdown. Не пересказывай, не исправляй, не добавляй ничего от себя. "
          "Только текст.")
OCR_CTX = 8192
OCR_TIMEOUT = 600.0  # с: ответ до 1200 токенов плюс возможная очередь в Ollama
# Текст слайда умещается в ~800 токенов; лимит выше обрывает зацикленный ответ.
OCR_MAX_TOKENS = 1200
_MIN_WORDS = 3


def transcribe(image: bytes, client: Any, cache_dir: Path, model: str) -> str:
    """Текст картинки (из кэша или новый); пустая строка — текста нет или это шум."""
    key = hashlib.sha256(image + PROMPT_VERSION.encode() + model.encode()).hexdigest()[:24]
    cache = cache_dir / f"{key}.txt"
    if cache.exists():
        raw = cache.read_text(encoding="utf-8")
    else:
        result = client.generate(PROMPT, images=[base64.b64encode(image).decode()],
                                 num_predict=OCR_MAX_TOKENS, num_ctx=OCR_CTX,
                                 timeout=OCR_TIMEOUT)
        raw = result.text
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(raw, encoding="utf-8")
    return clean(raw)


def clean(raw: str) -> str:
    """Без повторов строк и обёрток ```; без пустых строк — блок рисунка неделим.

    Строки таблицы (|…|) и короткие строки из цифр не дедуплицируются: в таблице
    одинаковые строки и значения законны.
    """
    seen: set[str] = set()
    lines: list[str] = []
    for line in raw.replace("```markdown", "").replace("```", "").split("\n"):
        text = line.strip()
        if not text:
            continue
        key = re.sub(r"\W+", "", text.lower())
        if not text.startswith("|") and len(key) > 3:
            if key in seen:
                continue
            seen.add(key)
        lines.append(text)
    result = "\n".join(lines)
    words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", result)
    return result if len(words) >= _MIN_WORDS else ""
