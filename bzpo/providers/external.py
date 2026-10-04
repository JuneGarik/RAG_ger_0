"""Внешний сервис эмбеддингов и реранкинга (SPEC 6.1–6.4).

Единственное место в проекте, где выполняются HTTP-обращения к сервису
эмбеддингов/реранкинга; правило проверяется tests/test_boundaries.py.

Что здесь решается и почему именно здесь, а не в вызывающем коде:

* **батчи и усечение.** Конвейеру индексации и поиску незачем знать пределы
  сервиса; они меняются вместе с сервисом, а не вместе с логикой поиска.
* **порядок ответа.** Ответ сортируется по полю `index`: порядок элементов
  массива не гарантирован контрактом, а молча перепутанные векторы дают
  дефект, который не проявится нигде, кроме метрик качества.
* **повторы и лимиты.** 1 + `retry_attempts` попыток с задержками 1/3/9 с,
  `Retry-After` при 429, семафор `max_concurrency` и спейсинг под 20 rps.
  Ограничители общие на клиента и потокобезопасные: грани запроса ходят в
  сервис параллельно (SPEC раздел 11), лимит обязан быть общим.
* **нормализация L2=1** — пороги дедупликации и графа определены на
  нормализованных векторах (SPEC раздел 3).
* **кэш векторов запросов** с ключом (хэш текста, модель, ревизия модели):
  без ревизии кэш после обновления модели молча отдаёт несовместимые векторы.

Время (`sleep`, `monotonic`) инжектируется: иначе тест повторов ждал бы
13 секунд реального времени и его перестали бы гонять.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from bzpo.config import ExternalServiceConfig
from bzpo.providers.base import (
    ExternalProtocolError,
    ExternalServiceError,
    ExternalServiceUnavailable,
)

logger = logging.getLogger(__name__)

#: Сколько символов тела показывать в тексте ошибки разбора.
_BODY_PREVIEW = 300


class _RateLimiter:
    """Равномерный спейсинг запросов под лимит rps.

    Именно спейсинг, а не «20 в первую секунду и тишина»: сервис общий,
    залповая нагрузка от одного клиента отбирает окно у остальных.
    """

    def __init__(
        self,
        rps: float,
        monotonic: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        self._interval = 1.0 / rps
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_free: float | None = None

    def acquire(self) -> float:
        """Дождаться своего окна. Возвращает длительность ожидания (для тестов)."""
        with self._lock:
            now = self._monotonic()
            if self._next_free is None or now >= self._next_free:
                self._next_free = now + self._interval
                wait = 0.0
            else:
                wait = self._next_free - now
                self._next_free += self._interval
        if wait > 0:
            self._sleep(wait)
        return wait


class HttpBackend:
    """Транспорт: таймауты, повторы, лимит параллелизма, разбор JSON."""

    def __init__(
        self,
        config: ExternalServiceConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        import time  # локально: модуль времени нужен только для дефолтов

        self._config = config
        self._sleep = sleep or time.sleep
        self._own_client = client is None
        self._client = client or httpx.Client()
        self._timeout = httpx.Timeout(
            connect=config.timeout_connect,
            read=config.timeout_read,
            write=config.timeout_connect,
            pool=config.timeout_connect,
        )
        self._semaphore = threading.BoundedSemaphore(config.max_concurrency)
        self._limiter = _RateLimiter(
            config.rate_limit_rps, monotonic or time.monotonic, self._sleep
        )

    # ------------------------------------------------------------------ HTTP

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        """POST с повторами. Возвращает разобранный JSON."""
        return self._request("POST", path, payload)

    def get_json(self, path: str) -> Any:
        """GET с повторами (синтетическая проба доступности, SPEC 6.3)."""
        return self._request("GET", path, None)

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def __enter__(self) -> HttpBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --------------------------------------------------------------- детали

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> Any:
        url = self._config.base_url.rstrip("/") + path
        attempts = self._config.retry_attempts + 1
        delay = 0.0
        last_reason = "неизвестно"

        for attempt in range(attempts):
            if attempt:
                logger.warning(
                    "внешний сервис %s: попытка %d/%d через %.1f с (%s)",
                    path,
                    attempt + 1,
                    attempts,
                    delay,
                    last_reason,
                )
                self._sleep(delay)
            try:
                response = self._send(method, url, payload)
            except httpx.HTTPError as exc:
                last_reason = f"{type(exc).__name__}: {exc}"
                if attempt == attempts - 1:
                    raise ExternalServiceUnavailable(
                        f"внешний сервис недоступен: {method} {url}, "
                        f"{attempts} попыток, последняя ошибка — {last_reason}"
                    ) from exc
                delay = self._retry_delay(attempt)
                continue

            if response.status_code == httpx.codes.OK:
                return self._decode(response, url)

            if self._is_retryable(response.status_code):
                last_reason = f"HTTP {response.status_code}"
                if attempt == attempts - 1:
                    raise ExternalServiceUnavailable(
                        f"внешний сервис недоступен: {method} {url}, "
                        f"{attempts} попыток, последний ответ — {last_reason} "
                        f"{_preview(response)}"
                    )
                delay = self._delay_after_status(response, attempt)
                continue

            # Прочие 4xx повторять бессмысленно: это наш запрос или наш токен.
            raise ExternalServiceError(
                f"внешний сервис отклонил запрос: {method} {url} -> "
                f"HTTP {response.status_code} {_preview(response)}"
            )

        raise AssertionError("недостижимо: цикл повторов всегда завершается явно")

    def _send(self, method: str, url: str, payload: dict[str, Any] | None) -> httpx.Response:
        with self._semaphore:
            self._limiter.acquire()
            return self._client.request(
                method, url, json=payload, headers=self._headers(), timeout=self._timeout
            )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.token is not None:
            headers["Authorization"] = f"Bearer {self._config.token.get_secret_value()}"
        return headers

    @staticmethod
    def _is_retryable(status: int) -> bool:
        return status == httpx.codes.TOO_MANY_REQUESTS or status >= 500

    def _retry_delay(self, attempt: int) -> float:
        delays = self._config.retry_delays
        if not delays:
            return 0.0
        return delays[min(attempt, len(delays) - 1)]

    def _delay_after_status(self, response: httpx.Response, attempt: int) -> float:
        """Задержка перед повтором: Retry-After при 429, иначе штатная 1/3/9."""
        default = self._retry_delay(attempt)
        if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
            return default
        raw = response.headers.get("Retry-After")
        if raw is None:
            return default
        try:
            seconds = float(raw.strip())
        except ValueError:
            # Формат HTTP-date поддерживаем, но не разбираем сами: разница между
            # часами клиента и сервера делает такой расчёт менее надёжным,
            # чем штатная задержка.
            logger.warning("Retry-After в неожиданном формате: %r, беру %.1f с", raw, default)
            return default
        if seconds < 0:
            return default
        return min(seconds, self._config.retry_after_max)

    def _decode(self, response: httpx.Response, url: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise ExternalProtocolError(
                f"внешний сервис вернул невалидный JSON: {url} -> {_preview(response)}"
            ) from exc


def _preview(response: httpx.Response) -> str:
    try:
        body = response.text
    except Exception:  # pragma: no cover - защита от экзотических кодировок
        return "<тело не читается>"
    body = body.strip().replace("\n", " ")
    if len(body) > _BODY_PREVIEW:
        body = body[:_BODY_PREVIEW] + "…"
    return f"«{body}»"


# --------------------------------------------------------------- эмбеддинги


class ExternalEmbedder:
    """Реализация Embedder поверх POST {embed_path} (SPEC 6.1)."""

    def __init__(
        self,
        config: ExternalServiceConfig,
        backend: HttpBackend | None = None,
        **backend_kwargs: Any,
    ) -> None:
        self._config = config
        self._backend = backend or HttpBackend(config, **backend_kwargs)
        self._cache: OrderedDict[tuple[str, str, str], list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Векторы в порядке входа. Кэш не используется: это поток документов."""
        if not texts:
            return []
        prepared = [self._truncate(text) for text in texts]
        batch_size = self._config.embed_batch
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), batch_size):
            vectors.extend(self._embed_batch(prepared[start : start + batch_size]))
        return vectors

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Векторы запросов с кэшем (SPEC 6.3).

        Кэшируются только запросы: тексты корпуса приходят по одному разу,
        а гипотезы HyDE недетерминированы и кэшу не подлежат — вызывающий код
        обязан вести их через `embed`.
        """
        result: list[list[float] | None] = [None] * len(texts)
        missing_index: list[int] = []
        missing_text: list[str] = []

        for position, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is None:
                missing_index.append(position)
                missing_text.append(text)
            else:
                result[position] = cached

        if missing_text:
            for position, vector in zip(missing_index, self.embed(missing_text), strict=True):
                self._cache_put(texts[position], vector)
                result[position] = list(vector)

        filled: list[list[float]] = []
        for position, vector in enumerate(result):
            if vector is None:  # pragma: no cover - защита от правки логики кэша
                raise ExternalProtocolError(f"вектор запроса #{position} не получен")
            filled.append(vector)
        return filled

    def embed_query(self, text: str) -> list[float]:
        """Вектор одного запроса (с кэшем)."""
        return self.embed_queries([text])[0]

    def health(self) -> bool:
        """Синтетическая проба доступности. Планировщик пробы — на стороне вызывающего."""
        try:
            self._backend.get_json(self._config.health_path)
        except ExternalServiceError as exc:
            logger.warning("проба доступности не прошла: %s", exc)
            return False
        return True

    def close(self) -> None:
        self._backend.close()

    # --------------------------------------------------------------- детали

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {
            "model": self._config.embed_model,
            "input": batch,
            "encoding_format": "float",
        }
        data = self._backend.post_json(self._config.embed_path, payload)
        raw = _parse_embedding_response(data, len(batch), self._config.embed_dim)
        return [_l2_normalize(vector) for vector in raw]

    def _truncate(self, text: str) -> str:
        limit = self._config.max_input_chars
        if len(text) <= limit:
            return text
        # Усечение с конца (SPEC 5.2): начало фрагмента несёт заголовок раздела
        # и подпись таблицы, они информативнее хвоста.
        logger.debug("вход усечён до %d символов (было %d)", limit, len(text))
        return text[:limit]

    def _cache_key(self, text: str) -> tuple[str, str, str]:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return digest, self._config.embed_model, self._config.embed_model_revision

    def _cache_get(self, text: str) -> list[float] | None:
        if self._config.query_cache_size == 0:
            return None
        key = self._cache_key(text)
        with self._cache_lock:
            vector = self._cache.get(key)
            if vector is None:
                return None
            self._cache.move_to_end(key)
            return list(vector)

    def _cache_put(self, text: str, vector: list[float]) -> None:
        if self._config.query_cache_size == 0:
            return
        key = self._cache_key(text)
        with self._cache_lock:
            self._cache[key] = list(vector)
            self._cache.move_to_end(key)
            while len(self._cache) > self._config.query_cache_size:
                self._cache.popitem(last=False)


# ----------------------------------------------------------------- реранкер


class ExternalReranker:
    """Реализация Reranker для обоих диалектов SPEC 6.2.

    Диалект берётся из конфигурации. Автоопределения нет сознательно: молчаливое
    переключение формата разбора маскирует смену контракта сервиса.
    """

    def __init__(
        self,
        config: ExternalServiceConfig,
        backend: HttpBackend | None = None,
        **backend_kwargs: Any,
    ) -> None:
        self._config = config
        self._backend = backend or HttpBackend(config, **backend_kwargs)

    def rerank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        """Ранжирование кандидатов по убыванию оценки.

        Кандидаты режутся на батчи `rerank_batch` (сколько пар сервис принимает
        за вызов — блокирующий вопрос SPEC 6.2), но индексы в результате всегда
        глобальные: локальная нумерация батча смещается на его начало. Оценки
        кросс-энкодера абсолютны, поэтому сортировка сквозная по всем батчам.
        """
        if not docs:
            return []
        prepared_query = self._truncate(query)
        prepared_docs = [self._truncate(doc) for doc in docs]
        batch_size = self._config.rerank_batch

        scored: list[tuple[int, float]] = []
        for start in range(0, len(prepared_docs), batch_size):
            batch = prepared_docs[start : start + batch_size]
            for local_index, score in self._rerank_batch(prepared_query, batch):
                scored.append((start + local_index, score))

        # Вторичный ключ — индекс: при равных оценках порядок детерминирован.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored

    def close(self) -> None:
        self._backend.close()

    # --------------------------------------------------------------- детали

    def _rerank_batch(self, query: str, batch: list[str]) -> list[tuple[int, float]]:
        if self._config.rerank_dialect == "tei":
            payload: dict[str, Any] = {"query": query, "texts": batch, "raw_scores": False}
        else:
            payload = {"model": self._config.rerank_model, "query": query, "documents": batch}
        data = self._backend.post_json(self._config.effective_rerank_path, payload)
        return _parse_rerank_response(data, len(batch), self._config.rerank_dialect)

    def _truncate(self, text: str) -> str:
        limit = self._config.max_input_chars
        return text if len(text) <= limit else text[:limit]


# ------------------------------------------------------------------- разбор


def _parse_embedding_response(data: Any, expected: int, expected_dim: int) -> list[list[float]]:
    """Разбор ответа эмбеддера с восстановлением порядка по полю index."""
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ExternalProtocolError(
            "неожиданная схема ответа эмбеддера: ожидался объект с массивом "
            f"«data», получено {_describe(data)}"
        )
    items = data["data"]
    if len(items) != expected:
        raise ExternalProtocolError(
            f"эмбеддер вернул {len(items)} векторов на {expected} текстов — "
            "молчаливая потеря части входа недопустима"
        )

    ordered: dict[int, list[float]] = {}
    for position, item in enumerate(items):
        if not isinstance(item, dict) or "embedding" not in item or "index" not in item:
            raise ExternalProtocolError(
                f"неожиданная схема элемента #{position} ответа эмбеддера: {_describe(item)}"
            )
        index = item["index"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < expected:
            raise ExternalProtocolError(
                f"элемент #{position} ответа эмбеддера имеет недопустимый index={index!r}"
            )
        if index in ordered:
            raise ExternalProtocolError(f"index={index} встречается в ответе эмбеддера дважды")
        ordered[index] = _as_float_vector(item["embedding"], f"вектор index={index}")

    vectors = [ordered[index] for index in range(expected)]
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dim:
            raise ExternalProtocolError(
                f"размерность вектора index={index} равна {len(vector)}, ожидалось "
                f"{expected_dim}: схема pgvector рассчитана на фиксированную размерность"
            )
    return vectors


def _parse_rerank_response(data: Any, expected: int, dialect: str) -> list[tuple[int, float]]:
    """Разбор ответа реранкера обоих диалектов -> список (index, score)."""
    if dialect == "tei":
        items = data
        score_key = "score"
        shape = "массив объектов {index, score}"
    else:
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ExternalProtocolError(
                "неожиданная схема ответа реранкера (диалект cohere): ожидался объект с "
                f"массивом «results», получено {_describe(data)}"
            )
        items = data["results"]
        score_key = "relevance_score"
        shape = "объект {results: [{index, relevance_score}]}"

    if not isinstance(items, list):
        raise ExternalProtocolError(
            f"неожиданная схема ответа реранкера (диалект {dialect}): ожидался {shape}, "
            f"получено {_describe(data)}"
        )
    if len(items) != expected:
        raise ExternalProtocolError(
            f"реранкер вернул {len(items)} оценок на {expected} документов — "
            "часть кандидатов молча потеряна"
        )

    seen: set[int] = set()
    pairs: list[tuple[int, float]] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict) or "index" not in item or score_key not in item:
            raise ExternalProtocolError(
                f"неожиданная схема элемента #{position} ответа реранкера "
                f"(диалект {dialect}): {_describe(item)}"
            )
        index = item["index"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < expected:
            raise ExternalProtocolError(
                f"элемент #{position} ответа реранкера имеет недопустимый index={index!r}"
            )
        if index in seen:
            raise ExternalProtocolError(f"index={index} встречается в ответе реранкера дважды")
        seen.add(index)
        pairs.append((index, _as_float(item[score_key], f"оценка index={index}")))
    return pairs


def _as_float(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExternalProtocolError(f"{what}: ожидалось число, получено {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ExternalProtocolError(f"{what}: получено нечисловое значение {value!r}")
    return number


def _as_float_vector(value: Any, what: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ExternalProtocolError(f"{what}: ожидался непустой массив чисел, получено {value!r}")
    return [_as_float(item, what) for item in value]


def _l2_normalize(vector: list[float]) -> list[float]:
    """L2=1. Нулевой вектор — ошибка: нормализовать его нельзя, а пороги
    косинусной близости на нём неопределены."""
    norm = math.sqrt(math.fsum(component * component for component in vector))
    if norm == 0.0:
        raise ExternalProtocolError(
            "эмбеддер вернул нулевой вектор: косинусные пороги на нём неопределены"
        )
    return [component / norm for component in vector]


def _describe(value: Any) -> str:
    text = repr(value)
    if len(text) > _BODY_PREVIEW:
        text = text[:_BODY_PREVIEW] + "…"
    return text
