"""Клиент Ollama для локальной генерации (SPEC раздел 10).

РЕШЕНИЕ (фиксируется): используется `POST /api/generate`, не `/api/chat`.
Причины:

* промпт собирается нашими шаблонами Jinja2 целиком (SPEC раздел 7), роли
  сообщений в нём не нужны; `/api/chat` добавил бы ещё один слой разметки
  между тем, что мы собрали, и тем, что увидела модель;
* рантайм-ассерт инварианта HyDE (SPEC 5.1) проверяет ровно ту строку, которая
  уходит в модель. С `/api/chat` проверяемый текст и отправляемый расходятся;
* маркеры [N] нумеруются по порядку фрагментов в промпте — единая строка
  промпта делает эту нумерацию проверяемой детерминированно.

Обязательные options (SPEC раздел 10, не значения по умолчанию Ollama):
`num_ctx=16384` (дефолт 2048 молча усечёт контекст с разделами-образцами —
дефект без симптомов), `num_predict=GEN_MAX_TOKENS`, `keep_alive=-1`
(модель резидентна, иначе перезагрузка 12 ГБ съедает бюджет латентности).
`temperature` и `seed` задаются на вызов: 0 для контрактных JSON-вызовов,
0.7 для гипотез HyDE.

Повторов здесь нет сознательно: повтор при невалидном JSON — забота слоя
разбора (SPEC 5.1, до 2 повторов и детерминированный фолбэк), а повторять
30-секундную генерацию из-за таймаута внутри 60-секундного бюджета запроса
бессмысленно.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from bzpo.config import OllamaConfig, OllamaRole
from bzpo.providers.base import ExternalServiceError, ExternalServiceUnavailable

logger = logging.getLogger(__name__)

GENERATE_PATH = "/api/generate"


class OllamaError(ExternalServiceError):
    """Ошибка обращения к Ollama. Отдельный класс — отдельное сообщение
    пользователю: «модель генерации недоступна» (SPEC 10а)."""


class OllamaUnavailable(OllamaError, ExternalServiceUnavailable):
    """Ollama не отвечает или вернула ошибку транспорта."""


@dataclass(frozen=True, slots=True)
class Generation:
    """Результат генерации вместе с фактическими счётчиками токенов."""

    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_duration_ms: float | None
    done_reason: str | None

    @property
    def truncated_by_length(self) -> bool:
        """Генерация оборвана по num_predict, а не по завершению мысли."""
        return self.done_reason == "length"


class OllamaClient:
    """Тонкий клиент /api/generate с обязательными параметрами из конфигурации.

    Роль задаётся при создании и без значения по умолчанию: конвейер индексации
    создаёт клиент `enrichment`, обработка запроса — `query`. Модель по роли
    берётся из конфигурации, вызывающий код имён моделей не знает.
    """

    def __init__(
        self,
        config: OllamaConfig,
        *,
        role: OllamaRole,
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._model = config.model_for(role)
        self._own_client = client is None
        self._client = client or httpx.Client()
        self._timeout = httpx.Timeout(
            connect=config.timeout_connect,
            read=config.timeout_read,
            write=config.timeout_connect,
            pool=config.timeout_connect,
        )

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        num_predict: int | None = None,
        json_format: bool = False,
        stop: list[str] | None = None,
        system: str | None = None,
        images: list[str] | None = None,
    ) -> Generation:
        """Один синхронный вызов генерации.

        `json_format=True` включает режим строгого JSON у Ollama — им пользуются
        контрактные вызовы разбора запроса (SPEC 5.1). Это не отменяет проверку
        схемы на нашей стороне: формат гарантирует синтаксис, а не поля.

        `system` — правила стиля и запреты (bzpo/generation/prompts/). Нумерация
        фрагментов [N] живёт только в `prompt`, чтобы проверяемая строка
        контекста оставалась одной.
        """
        payload = self._payload(prompt, temperature=temperature, seed=seed,
                                num_predict=num_predict, json_format=json_format,
                                stop=stop, system=system)
        if images:
            # Картинки в base64 — для мультимодальной модели (роль vision).
            payload["images"] = images
        data = self._post(GENERATE_PATH, payload)
        return self._parse(data)

    def stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        num_predict: int | None = None,
        system: str | None = None,
    ) -> Iterator[str | Generation]:
        """Потоковая генерация: куски текста по мере готовности, последним — Generation.

        Для интерфейса: длинный раздел пишется 1–2 минуты, и пользователь видит
        текст сразу, а не после окончания. Итоговый Generation несёт те же
        счётчики, что и `generate` (done_reason, токены), — по нему видна обрезка.
        """
        payload = self._payload(prompt, temperature=temperature, seed=None,
                                num_predict=num_predict, json_format=False, stop=None,
                                system=system)
        payload["stream"] = True
        url = self._config.base_url.rstrip("/") + GENERATE_PATH
        parts: list[str] = []
        try:
            with self._client.stream("POST", url, json=payload, timeout=self._timeout) as response:
                if response.status_code != httpx.codes.OK:
                    response.read()
                    raise OllamaError(
                        f"Ollama вернула HTTP {response.status_code} на {url}: "
                        f"{response.text[:300]}"
                    )
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError as exc:
                        raise OllamaError(f"Ollama вернула невалидный JSON: {line[:300]}") from exc
                    if "error" in data:
                        raise OllamaError(f"ошибка Ollama: {data['error']}")
                    piece = data.get("response", "")
                    if piece:
                        parts.append(piece)
                        yield piece
                    if data.get("done"):
                        yield self._parse({**data, "response": "".join(parts)})
                        return
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(
                f"модель генерации недоступна: {url}, {type(exc).__name__}: {exc}"
            ) from exc
        raise OllamaError("поток Ollama оборвался без завершающего сообщения")

    def _payload(
        self,
        prompt: str,
        *,
        temperature: float,
        seed: int | None,
        num_predict: int | None,
        json_format: bool,
        stop: list[str] | None,
        system: str | None,
    ) -> dict[str, Any]:
        resolved_predict = num_predict if num_predict is not None else self._config.num_predict
        if resolved_predict is None:
            # Settings подставляет num_predict из gen_max_tokens; None здесь означает,
            # что конфигурация собрана в обход валидации, и модель получила бы
            # безлимитную генерацию — при keep_alive=-1 это зависший запрос.
            raise ValueError(
                "num_predict не задан: OllamaConfig должен приходить из Settings "
                "(там он подставляется из generation.gen_max_tokens)"
            )

        options: dict[str, Any] = {
            "num_ctx": self._config.num_ctx,
            "num_predict": resolved_predict,
            "temperature": temperature,
        }
        if seed is not None:
            options["seed"] = seed
        if stop:
            options["stop"] = stop

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": self._config.think,
            "keep_alive": self._config.keep_alive,
            "options": options,
        }
        if json_format:
            payload["format"] = "json"
        if system is not None:
            # Заменяет системный промпт из Modelfile модели, а не дополняет его.
            payload["system"] = system

        return payload

    def installed_models(self) -> list[str] | None:
        """Модели, скачанные в Ollama; None — Ollama не отвечает."""
        url = self._config.base_url.rstrip("/") + "/api/tags"
        try:
            response = self._client.get(url, timeout=self._config.timeout_connect)
            response.raise_for_status()
            return [m.get("name", "") for m in response.json().get("models", [])]
        except (httpx.HTTPError, ValueError):
            return None

    @property
    def model(self) -> str:
        return self._model

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --------------------------------------------------------------- детали

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        url = self._config.base_url.rstrip("/") + path
        try:
            response = self._client.post(url, json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise OllamaUnavailable(
                f"модель генерации недоступна: {url}, {type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code != httpx.codes.OK:
            raise OllamaError(
                f"Ollama вернула HTTP {response.status_code} на {url}: "
                f"{response.text[:300]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise OllamaError(
                f"Ollama вернула невалидный JSON на {url}: {response.text[:300]}"
            ) from exc

    def _parse(self, data: Any) -> Generation:
        if not isinstance(data, dict) or not isinstance(data.get("response"), str):
            raise OllamaError(
                f"неожиданная схема ответа Ollama: ожидалось поле «response», получено {data!r}"
            )
        prompt_tokens = _as_optional_int(data.get("prompt_eval_count"))
        done_reason = data.get("done_reason")
        total_ns = _as_optional_int(data.get("total_duration"))

        # Тихое усечение промпта — главный дефект без симптомов из SPEC раздела 10.
        # Если промпт съел почти всё окно, ответ обрежется, а внешне всё «работает».
        if prompt_tokens is not None and prompt_tokens >= self._config.num_ctx * 0.95:
            logger.warning(
                "промпт занял %d токенов при num_ctx=%d: контекст на грани усечения",
                prompt_tokens,
                self._config.num_ctx,
            )

        return Generation(
            text=data["response"],
            model=str(data.get("model", self._model)),
            prompt_tokens=prompt_tokens,
            completion_tokens=_as_optional_int(data.get("eval_count")),
            total_duration_ms=None if total_ns is None else total_ns / 1e6,
            done_reason=done_reason if isinstance(done_reason, str) else None,
        )


def _as_optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)
