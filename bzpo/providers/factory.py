"""Фабрика провайдеров: external | local по конфигурации (SPEC 6.4).

local — модели в процессе приложения (профиль ноутбука, секция `local`);
external — HTTP-сервис (секция `external`).

Эмбеддер и реранкер собираются вместе и делят один HTTP-транспорт: семафор
`max_concurrency` и лимит 20 rps относятся к сервису целиком, а не к каждому
клиенту в отдельности. Два независимых клиента дали бы вдвое больший
фактический параллелизм при формально верной конфигурации.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bzpo.config import ConfigError, Settings
from bzpo.providers.base import Embedder, Reranker
from bzpo.providers.external import ExternalEmbedder, ExternalReranker, HttpBackend
from bzpo.providers.local import LocalEmbedder, LocalReranker


@dataclass(frozen=True, slots=True)
class Providers:
    """Собранная пара провайдеров и общий транспорт (может быть None у local)."""

    embedder: Embedder
    reranker: Reranker
    backend: HttpBackend | None

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()


def build_providers(settings: Settings, **backend_kwargs: Any) -> Providers:
    """Создать провайдеры по конфигурации.

    Отсутствие токена при provider=external — ошибка конфигурации, а не
    предупреждение: без него первый же запрос упадёт 401, но уже внутри
    пользовательского сценария.
    """
    config = settings.external
    if config.provider == "local":
        return Providers(
            embedder=LocalEmbedder(settings.local),
            reranker=LocalReranker(settings.local),
            backend=None,
        )

    if config.token is None:
        raise ConfigError(
            "external.provider=external, но токен сервиса не задан: "
            "укажите переменную окружения BZPO_EXTERNAL__TOKEN"
        )

    backend = HttpBackend(config, **backend_kwargs)
    return Providers(
        embedder=ExternalEmbedder(config, backend),
        reranker=ExternalReranker(config, backend),
        backend=backend,
    )
