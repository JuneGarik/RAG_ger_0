"""Общие настройки тестов.

Путь к корню проекта добавляется в sys.path, чтобы `import bzpo` работал без
установки пакета: на этапе 0.5 CI ставит только зависимости, а не сам проект.

Тесты не обращаются ни к PostgreSQL, ни к внешнему сервису, ни к Ollama.
Всё сетевое проверяется на httpx.MockTransport.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Убрать переменные BZPO_* из окружения разработчика.

    Иначе экспортированный в оболочке BZPO_SEARCH__TOP_K_FINAL молча меняет
    результат тестов конфигурации — и «зелёное» перестаёт что-либо значить.
    """
    for name in list(os.environ):
        if name.startswith("BZPO_"):
            monkeypatch.delenv(name, raising=False)
