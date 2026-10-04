"""Локальные Embedder/Reranker: модели в процессе приложения (SPEC 6.4).

Профиль — ноутбук Apple Silicon с 16 ГБ общей памяти: эмбеддер
Giga-Embeddings-instruct-480M и реранкер bge-reranker-v2-m3 через
sentence-transformers на MPS. Контракт тот же, что у внешнего сервиса:
векторы L2=1 в порядке входа, реранкинг по убыванию оценки с глобальными
индексами, кэш векторов запросов с ревизией модели в ключе.

Отличие от внешнего сервиса, которое видно вызывающему коду, одно и
намеренное: `embed_queries` добавляет к запросу инструкцию модели
(`local.query_instruction`), а `embed` кодирует текст как есть. Поэтому
векторы запросов обязаны идти через `embed_queries`, а тексты корпуса и
гипотезы HyDE (они по форме — документ) — через `embed`.

torch и sentence-transformers импортируются лениво, при первом обращении к
модели: тесты и CI работают без них, а приложение с provider=external их не
требует вовсе. Установка: `pip install -e ".[local]"`.

Проверка на живых моделях: `python -m bzpo.providers.local`.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Protocol

from bzpo.config import LocalModelsConfig
from bzpo.providers.base import ExternalProtocolError, ExternalServiceError

logger = logging.getLogger(__name__)


class LocalModelError(ExternalServiceError):
    """Модель не загрузилась или вернула результат не по контракту."""


class _EncoderModel(Protocol):
    """Подмножество SentenceTransformer, которым пользуется эмбеддер."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


class _CrossEncoderModel(Protocol):
    """Подмножество CrossEncoder, которым пользуется реранкер."""

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any: ...


# ----------------------------------------------------------------- загрузка


def resolve_device(requested: str) -> str:
    """auto -> mps, если доступен, иначе cpu."""
    if requested != "auto":
        return requested
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _torch_dtype(name: str) -> Any:
    import torch

    return getattr(torch, name)


def _load_encoder(config: LocalModelsConfig) -> _EncoderModel:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise LocalModelError(
            "sentence-transformers не установлен: pip install -e \".[local]\""
        ) from exc
    device = resolve_device(config.device)
    logger.info("загрузка эмбеддера %s на %s", config.embed_model, device)
    try:
        model = SentenceTransformer(
            config.embed_model,
            revision=config.embed_model_revision,
            device=device,
            trust_remote_code=config.trust_remote_code,
            model_kwargs={"torch_dtype": _torch_dtype(config.dtype)},
        )
    except Exception as exc:  # noqa: BLE001 - причина уходит в текст ошибки
        raise LocalModelError(f"эмбеддер {config.embed_model} не загрузился: {exc}") from exc
    model.max_seq_length = config.embed_max_tokens
    return model


def _load_cross_encoder(config: LocalModelsConfig) -> _CrossEncoderModel:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise LocalModelError(
            "sentence-transformers не установлен: pip install -e \".[local]\""
        ) from exc
    device = resolve_device(config.device)
    logger.info("загрузка реранкера %s на %s", config.rerank_model, device)
    try:
        return CrossEncoder(
            config.rerank_model,
            max_length=config.rerank_max_tokens,
            device=device,
            model_kwargs={"torch_dtype": _torch_dtype(config.dtype)},
        )
    except Exception as exc:  # noqa: BLE001
        raise LocalModelError(f"реранкер {config.rerank_model} не загрузился: {exc}") from exc


# ----------------------------------------------------------------- эмбеддер


class LocalEmbedder:
    """Embedder поверх SentenceTransformer.

    Модель грузится при первом вызове, а не в конструкторе: фабрика собирает
    провайдеры при старте любого CLI, и загрузка 1 ГБ весов ради `--help`
    была бы лишней.
    """

    def __init__(self, config: LocalModelsConfig, model: _EncoderModel | None = None) -> None:
        self._config = config
        self._model = model
        self._load_lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str, str], list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Векторы текстов корпуса (без инструкции), в порядке входа."""
        return self._encode(list(texts), prompt=None)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Векторы запросов: с инструкцией модели и кэшем."""
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
            vectors = self._encode(missing_text, prompt=self._config.query_instruction)
            for position, vector in zip(missing_index, vectors, strict=True):
                self._cache_put(texts[position], vector)
                result[position] = vector

        return [vector for vector in result if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        """Вектор одного запроса (с инструкцией и кэшем)."""
        return self.embed_queries([text])[0]

    def count_tokens(self, text: str) -> int:
        """Длина текста в токенах модели — в тех единицах, в которых её режет max_seq_length."""
        tokenizer = getattr(self._get_model(), "tokenizer", None)
        if tokenizer is None:
            raise LocalModelError(f"у модели {self._config.embed_model} нет токенизатора")
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    def health(self) -> bool:
        """Модель загружается и кодирует тестовую строку."""
        try:
            self._encode(["проверка"], prompt=None)
        except ExternalServiceError as exc:
            logger.warning("проба эмбеддера не прошла: %s", exc)
            return False
        return True

    def close(self) -> None:
        self._model = None

    # --------------------------------------------------------------- детали

    def _get_model(self) -> _EncoderModel:
        with self._load_lock:
            if self._model is None:
                self._model = _load_encoder(self._config)
            return self._model

    def _encode(self, texts: list[str], prompt: str | None) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict[str, Any] = {
            "batch_size": self._config.embed_batch,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if prompt:
            kwargs["prompt"] = prompt
        raw = self._get_model().encode(texts, **kwargs)
        vectors = [[float(value) for value in row] for row in raw]
        if len(vectors) != len(texts):
            raise ExternalProtocolError(
                f"эмбеддер вернул {len(vectors)} векторов на {len(texts)} текстов"
            )
        for position, vector in enumerate(vectors):
            if len(vector) != self._config.embed_dim:
                raise ExternalProtocolError(
                    f"вектор #{position}: размерность {len(vector)}, "
                    f"ожидалась {self._config.embed_dim}"
                )
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isfinite(norm) or abs(norm - 1.0) > 0.05:
                # NaN на MPS в половинной точности — известный класс сбоев;
                # молча записанный NaN-вектор ломает поиск целиком.
                raise ExternalProtocolError(
                    f"вектор #{position} не нормализован (норма {norm}); "
                    "попробуйте local.dtype: float32"
                )
            # bfloat16 держит ~3 значащих цифры: норма после нормализации в модели
            # отходит от 1 на доли процента. Доводим до L2=1 в float64 — пороги
            # дедупликации и графа определены на нормализованных векторах.
            vectors[position] = [value / norm for value in vector]
        return vectors

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


class LocalReranker:
    """Reranker поверх CrossEncoder.

    Для bge-reranker-v2-m3 (одна метка) sentence-transformers применяет
    сигмоиду, оценки лежат в [0, 1] — в той же шкале, что `rerank_threshold`.
    Оценка вне [0, 1] значит, что модель заменили на выдающую логиты; это
    ошибка, а не повод молча откалибровать порог заново.
    """

    def __init__(
        self, config: LocalModelsConfig, model: _CrossEncoderModel | None = None
    ) -> None:
        self._config = config
        self._model = model
        self._load_lock = threading.Lock()

    def rerank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        if not docs:
            return []
        pairs = [(query, doc) for doc in docs]
        raw = self._get_model().predict(
            pairs,
            batch_size=self._config.rerank_batch,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = [float(value) for value in raw]
        if len(scores) != len(docs):
            raise ExternalProtocolError(
                f"реранкер вернул {len(scores)} оценок на {len(docs)} документов"
            )
        for position, score in enumerate(scores):
            if not (math.isfinite(score) and 0.0 <= score <= 1.0):
                raise ExternalProtocolError(
                    f"оценка #{position} = {score} вне [0, 1]: модель отдаёт логиты?"
                )
        ranked = list(enumerate(scores))
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    def close(self) -> None:
        self._model = None

    def _get_model(self) -> _CrossEncoderModel:
        with self._load_lock:
            if self._model is None:
                self._model = _load_cross_encoder(self._config)
            return self._model


# ------------------------------------------------------------- самопроверка


def _selfcheck() -> None:
    """Загрузить обе модели и прогнать короткий пример (скачивает веса при первом запуске)."""
    import time

    from bzpo.config import load_config

    config = load_config().local
    print(f"устройство: {resolve_device(config.device)}, dtype: {config.dtype}")

    embedder = LocalEmbedder(config)
    started = time.perf_counter()
    docs = [
        "Внедрение системы электронного документооборота для регионального оператора.",
        "Отчёт о поставке серверного оборудования в центр обработки данных.",
        "Рецепт пирога с яблоками.",
    ]
    doc_vectors = embedder.embed(docs)
    query_vector = embedder.embed_query("опыт внедрения СЭД")
    print(f"эмбеддер: {len(doc_vectors[0])} измерений, {time.perf_counter() - started:.1f} с")
    for doc, vector in zip(docs, doc_vectors, strict=True):
        similarity = sum(a * b for a, b in zip(query_vector, vector, strict=True))
        print(f"  {similarity:.3f}  {doc}")

    reranker = LocalReranker(config)
    started = time.perf_counter()
    ranked = reranker.rerank("опыт внедрения СЭД", docs)
    print(f"реранкер: {time.perf_counter() - started:.1f} с")
    for index, score in ranked:
        print(f"  {score:.3f}  {docs[index]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selfcheck()
