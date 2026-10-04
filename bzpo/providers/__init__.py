"""Модуль изоляции внешних зависимостей (SPEC 6.4).

Единственное место, откуда система обращается к сервису эмбеддингов/реранкинга
и к Ollama. Правило проверяется tests/test_boundaries.py.
"""

from bzpo.providers.base import (
    Embedder,
    ExternalProtocolError,
    ExternalServiceError,
    ExternalServiceUnavailable,
    Reranker,
)
from bzpo.providers.factory import Providers, build_providers

__all__ = [
    "Embedder",
    "ExternalProtocolError",
    "ExternalServiceError",
    "ExternalServiceUnavailable",
    "Providers",
    "Reranker",
    "build_providers",
]
