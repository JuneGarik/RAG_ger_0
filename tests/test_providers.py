"""Тесты модуля изоляции на моке (PLAN 0.5.4 — критерий закрытия этапа 0.5).

Внешний сервис не поднимается: весь HTTP идёт через httpx.MockTransport.
Время инжектируется фейковыми часами, поэтому проверка задержек 1/3/9 с
выполняется мгновенно.
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

import httpx
import pytest

from bzpo.config import (
    ConfigError,
    ExternalServiceConfig,
    LocalModelsConfig,
    OllamaConfig,
    Settings,
)
from bzpo.providers.base import (
    Embedder,
    ExternalProtocolError,
    ExternalServiceError,
    ExternalServiceUnavailable,
    Reranker,
)
from bzpo.providers.external import (
    ExternalEmbedder,
    ExternalReranker,
    HttpBackend,
    _RateLimiter,
)
from bzpo.providers.factory import build_providers
from bzpo.providers.local import LocalEmbedder, LocalReranker
from bzpo.providers.ollama import OllamaClient, OllamaError, OllamaUnavailable

# Паузы спейсинга rps на порядки меньше задержек повторов; при разборе
# записанных пауз интересны только повторы.
RETRY_SLEEP_FLOOR = 0.1


class FakeClock:
    """Часы без ожидания: sleep только двигает стрелку и пишет в журнал."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    @property
    def retry_sleeps(self) -> list[float]:
        return [value for value in self.sleeps if value >= RETRY_SLEEP_FLOOR]


def external_config(**overrides: Any) -> ExternalServiceConfig:
    defaults: dict[str, Any] = {
        "base_url": "http://service.test",
        "embed_dim": 3,
        # Спейсинг под 20 rps проверяется отдельно, здесь он только мешает.
        "rate_limit_rps": 1_000_000.0,
    }
    return ExternalServiceConfig(**{**defaults, **overrides})


def make_embedder(
    handler: Any, config: ExternalServiceConfig | None = None
) -> tuple[ExternalEmbedder, FakeClock]:
    config = config or external_config()
    clock = FakeClock()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = HttpBackend(config, client=client, sleep=clock.sleep, monotonic=clock.monotonic)
    return ExternalEmbedder(config, backend), clock


def make_reranker(
    handler: Any, config: ExternalServiceConfig | None = None
) -> tuple[ExternalReranker, FakeClock]:
    config = config or external_config()
    clock = FakeClock()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = HttpBackend(config, client=client, sleep=clock.sleep, monotonic=clock.monotonic)
    return ExternalReranker(config, backend), clock


def embedding_payload(texts: list[str], *, shuffled: bool = False) -> dict[str, Any]:
    """Ответ эмбеддера: вектор кодирует номер текста как отношение v[0]/v[1]."""
    items = []
    for position, text in enumerate(texts):
        number = float(text.split("#")[-1])
        items.append({"index": position, "embedding": [number, 1.0, 1.0]})
    if shuffled:
        items.reverse()
    return {"data": items, "usage": {"prompt_tokens": len(texts)}}


# --------------------------------------------------------------- эмбеддинги


def test_embedder_satisfies_protocol() -> None:
    embedder, _ = make_embedder(lambda request: httpx.Response(200, json={"data": []}))
    assert isinstance(embedder, Embedder)


def test_embed_batches_and_preserves_order() -> None:
    """300 текстов при батче 128 — три вызова, порядок входа сохранён."""
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sizes.append(len(payload["input"]))
        assert payload["model"] == "BAAI/bge-m3"
        assert payload["encoding_format"] == "float"
        # Порядок элементов ответа контрактом не гарантирован — отдаём наоборот.
        return httpx.Response(200, json=embedding_payload(payload["input"], shuffled=True))

    embedder, _ = make_embedder(handler, external_config(embed_batch=128))
    texts = [f"фрагмент #{number}" for number in range(300)]
    vectors = embedder.embed(texts)

    assert sizes == [128, 128, 44]
    assert len(vectors) == 300
    for number, vector in enumerate(vectors):
        assert vector[0] / vector[1] == pytest.approx(number, abs=1e-9)


def test_embed_sorts_by_index_not_by_position() -> None:
    """Перемешанный порядок index в ответе не должен путать векторы."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 2, "embedding": [2.0, 1.0, 1.0]},
                    {"index": 0, "embedding": [0.0, 1.0, 1.0]},
                    {"index": 1, "embedding": [1.0, 1.0, 1.0]},
                ]
            },
        )

    embedder, _ = make_embedder(handler)
    vectors = embedder.embed(["а", "б", "в"])
    assert [round(vector[0] / vector[1], 6) for vector in vectors] == [0.0, 1.0, 2.0]


def test_embed_normalizes_to_unit_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0, 0.0]}]})

    embedder, _ = make_embedder(handler)
    vector = embedder.embed(["текст"])[0]

    assert vector == pytest.approx([0.6, 0.8, 0.0])
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


def test_embed_truncates_long_input() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured.extend(payload["input"])
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]})

    embedder, _ = make_embedder(handler, external_config(max_input_chars=10))
    embedder.embed(["длинный текст " * 100])

    assert len(captured[0]) == 10
    assert captured[0] == "длинный те"


def test_embed_rejects_broken_json() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, content=b"{'data': [", headers={"content-type": "application/json"})

    embedder, _ = make_embedder(handler)
    with pytest.raises(ExternalProtocolError) as excinfo:
        embedder.embed(["текст"])

    assert "невалидный JSON" in str(excinfo.value)
    # Битый ответ не повторяем: повтор даст ровно то же самое.
    assert len(calls) == 1


def test_embed_rejects_unexpected_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"vectors": [[0.1, 0.2, 0.3]]})

    embedder, _ = make_embedder(handler)
    with pytest.raises(ExternalProtocolError) as excinfo:
        embedder.embed(["текст"])
    assert "неожиданная схема" in str(excinfo.value)


def test_embed_rejects_missing_vectors() -> None:
    """Ответ короче входа — молчаливая потеря части корпуса."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]})

    embedder, _ = make_embedder(handler)
    with pytest.raises(ExternalProtocolError) as excinfo:
        embedder.embed(["а", "б"])
    assert "1 векторов на 2 текстов" in str(excinfo.value)


def test_embed_rejects_wrong_dimension() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    embedder, _ = make_embedder(handler)
    with pytest.raises(ExternalProtocolError) as excinfo:
        embedder.embed(["текст"])
    assert "размерность" in str(excinfo.value)


def test_embed_rejects_zero_vector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0, 0.0, 0.0]}]})

    embedder, _ = make_embedder(handler)
    with pytest.raises(ExternalProtocolError):
        embedder.embed(["текст"])


def test_query_cache_avoids_repeated_calls() -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload["input"])
        return httpx.Response(200, json=embedding_payload(payload["input"]))

    embedder, _ = make_embedder(handler)
    first = embedder.embed_query("запрос #7")
    second = embedder.embed_query("запрос #7")

    assert first == second
    assert len(calls) == 1

    # Второй запрос кэшем не покрыт — уходит в сервис отдельным вызовом.
    embedder.embed_queries(["запрос #7", "запрос #8"])
    assert calls[1] == ["запрос #8"]


def test_query_cache_key_includes_model_revision() -> None:
    """Без ревизии кэш после обновления модели отдаёт несовместимые векторы."""
    old, _ = make_embedder(
        lambda request: httpx.Response(200, json={"data": []}),
        external_config(embed_model_revision="rev-1"),
    )
    new, _ = make_embedder(
        lambda request: httpx.Response(200, json={"data": []}),
        external_config(embed_model_revision="rev-2"),
    )
    assert old._cache_key("текст") != new._cache_key("текст")


# ------------------------------------------------------------------ реранкер


def test_reranker_satisfies_protocol() -> None:
    reranker, _ = make_reranker(lambda request: httpx.Response(200, json=[]))
    assert isinstance(reranker, Reranker)


def test_rerank_tei_dialect() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        # Порядок элементов перемешан намеренно.
        return httpx.Response(
            200,
            json=[
                {"index": 1, "score": 0.10},
                {"index": 2, "score": 0.93},
                {"index": 0, "score": 0.55},
            ],
        )

    reranker, _ = make_reranker(handler, external_config(rerank_dialect="tei"))
    result = reranker.rerank("выручка выросла", ["док-0", "док-1", "док-2"])

    assert captured["path"] == "/rerank"
    assert captured["payload"] == {
        "query": "выручка выросла",
        "texts": ["док-0", "док-1", "док-2"],
        "raw_scores": False,
    }
    assert result == [(2, 0.93), (0, 0.55), (1, 0.10)]


def test_rerank_cohere_dialect() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.93},
                    {"index": 0, "relevance_score": 0.55},
                    {"index": 1, "relevance_score": 0.10},
                ]
            },
        )

    reranker, _ = make_reranker(handler, external_config(rerank_dialect="cohere"))
    result = reranker.rerank("выручка выросла", ["док-0", "док-1", "док-2"])

    assert captured["path"] == "/v1/rerank"
    assert captured["payload"] == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "выручка выросла",
        "documents": ["док-0", "док-1", "док-2"],
    }
    assert result == [(2, 0.93), (0, 0.55), (1, 0.10)]


def test_rerank_batches_keep_global_indices() -> None:
    """120 кандидатов при батче 50 — три вызова, нумерация остаётся сквозной."""
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        texts = payload["texts"]
        sizes.append(len(texts))
        # Оценка кодирует глобальный номер документа: 0.001 * номер.
        items = [
            {"index": local, "score": 0.001 * float(text.split("#")[-1])}
            for local, text in enumerate(texts)
        ]
        items.reverse()
        return httpx.Response(200, json=items)

    reranker, _ = make_reranker(handler, external_config(rerank_batch=50))
    docs = [f"кандидат #{number}" for number in range(120)]
    result = reranker.rerank("запрос", docs)

    assert sizes == [50, 50, 20]
    assert len(result) == 120
    assert {index for index, _ in result} == set(range(120))
    # Оценка монотонна по номеру, значит порядок обязан быть строго убывающим.
    assert [index for index, _ in result] == list(range(119, -1, -1))
    assert result[0][1] == pytest.approx(0.119)


def test_rerank_rejects_unexpected_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "score": 0.5}]})

    reranker, _ = make_reranker(handler, external_config(rerank_dialect="cohere"))
    with pytest.raises(ExternalProtocolError) as excinfo:
        reranker.rerank("запрос", ["док"])
    assert "неожиданная схема" in str(excinfo.value)


def test_rerank_empty_input_makes_no_call() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=[])

    reranker, _ = make_reranker(handler)
    assert reranker.rerank("запрос", []) == []
    assert not calls


# -------------------------------------------------------- повторы и лимиты


def test_429_with_retry_after_is_honoured() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"error": "slow down"})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]})

    embedder, clock = make_embedder(handler)
    vectors = embedder.embed(["текст"])

    assert len(vectors) == 1
    assert len(attempts) == 2
    # Взята задержка сервиса (2 с), а не штатная первая (1 с).
    assert clock.retry_sleeps == [2.0]


def test_429_without_retry_after_uses_default_delays() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(429, json={"error": "rate limit"})

    embedder, clock = make_embedder(handler)
    with pytest.raises(ExternalServiceUnavailable):
        embedder.embed(["текст"])

    assert len(attempts) == 4  # первая попытка + три повтора
    assert clock.retry_sleeps == [1.0, 3.0, 9.0]


def test_retry_after_is_capped() -> None:
    """Сервис может попросить ждать час — внутри запроса пользователя это отказ."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "3600"}, json={})

    embedder, clock = make_embedder(handler, external_config(retry_after_max=30.0))
    with pytest.raises(ExternalServiceUnavailable):
        embedder.embed(["текст"])
    assert clock.retry_sleeps == [30.0, 30.0, 30.0]


def test_timeout_retries_then_unavailable() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ReadTimeout("сервис не ответил", request=request)

    embedder, clock = make_embedder(handler)
    with pytest.raises(ExternalServiceUnavailable) as excinfo:
        embedder.embed(["текст"])

    assert len(attempts) == 4
    assert clock.retry_sleeps == [1.0, 3.0, 9.0]
    assert "ReadTimeout" in str(excinfo.value)


def test_connect_error_retries_then_unavailable() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("соединение отклонено", request=request)

    embedder, _ = make_embedder(handler)
    with pytest.raises(ExternalServiceUnavailable):
        embedder.embed(["текст"])
    assert len(attempts) == 4


def test_server_error_retries_then_unavailable() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(503, text="upstream down")

    embedder, _ = make_embedder(handler)
    with pytest.raises(ExternalServiceUnavailable):
        embedder.embed(["текст"])
    assert len(attempts) == 4


def test_client_error_is_not_retried() -> None:
    """401 повторять бессмысленно: это наш токен, а не сбой сервиса."""
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(401, text="unauthorized")

    embedder, _ = make_embedder(handler)
    with pytest.raises(ExternalServiceError) as excinfo:
        embedder.embed(["текст"])

    assert not isinstance(excinfo.value, ExternalServiceUnavailable)
    assert len(attempts) == 1


def test_authorization_header_is_sent() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]})

    embedder, _ = make_embedder(handler, external_config(token="s3cret"))
    embedder.embed(["текст"])
    assert captured["auth"] == "Bearer s3cret"


def test_concurrency_never_exceeds_limit() -> None:
    """Семафор общий на клиента: грани запроса ходят в сервис параллельно."""
    lock = threading.Lock()
    state = {"current": 0, "peak": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        time.sleep(0.02)
        with lock:
            state["current"] -= 1
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]})

    config = external_config(max_concurrency=2)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = HttpBackend(config, client=client)
    embedder = ExternalEmbedder(config, backend)

    threads = [threading.Thread(target=lambda: embedder.embed(["текст"])) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["peak"] <= 2


def test_rate_limiter_spaces_requests() -> None:
    clock = FakeClock()
    limiter = _RateLimiter(10.0, clock.monotonic, clock.sleep)

    assert limiter.acquire() == 0.0
    assert limiter.acquire() == pytest.approx(0.1)
    assert limiter.acquire() == pytest.approx(0.1)
    assert clock.sleeps == pytest.approx([0.1, 0.1])


def test_health_probe_reports_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="down")

    embedder, _ = make_embedder(handler)
    assert embedder.health() is False


# --------------------------------------------------------- local и фабрика


class FakeEncoder:
    """Подставной SentenceTransformer: вектор [номер текста, 1, 1, ...] / норма."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append({"sentences": list(sentences), **kwargs})
        out = []
        for text in sentences:
            raw = [float(text.split("#")[-1]) if "#" in text else 0.0] + [1.0] * (self.dim - 1)
            norm = math.sqrt(sum(value * value for value in raw))
            out.append([value / norm for value in raw])
        return out


class FakeCrossEncoder:
    """Подставной CrossEncoder: оценка = номер документа / 100."""

    def __init__(self, scores: list[float] | None = None) -> None:
        self.scores = scores
        self.calls = 0

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.calls += 1
        if self.scores is not None:
            return list(self.scores)
        return [float(doc.split("#")[-1]) / 100 for _, doc in sentences]


def local_config(**overrides: Any) -> LocalModelsConfig:
    return LocalModelsConfig(**{"embed_dim": 4, **overrides})


def test_local_embedder_satisfies_protocol() -> None:
    assert isinstance(LocalEmbedder(local_config(), FakeEncoder()), Embedder)
    assert isinstance(LocalReranker(local_config(), FakeCrossEncoder()), Reranker)


def test_local_embed_documents_without_instruction() -> None:
    model = FakeEncoder()
    vectors = LocalEmbedder(local_config(), model).embed(["док #1", "док #2"])

    assert len(vectors) == 2
    assert "prompt" not in model.calls[0]
    assert model.calls[0]["normalize_embeddings"] is True
    for vector in vectors:
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_local_queries_get_instruction_and_cache() -> None:
    model = FakeEncoder()
    embedder = LocalEmbedder(local_config(query_instruction="Instruct: x\nQuery: "), model)

    first = embedder.embed_queries(["запрос #1", "запрос #2"])
    second = embedder.embed_queries(["запрос #2", "запрос #3"])

    assert model.calls[0]["prompt"] == "Instruct: x\nQuery: "
    # Второй вызов кодирует только незакэшированный запрос.
    assert model.calls[1]["sentences"] == ["запрос #3"]
    assert second[0] == first[1]


def test_local_embed_rejects_wrong_dimension() -> None:
    embedder = LocalEmbedder(local_config(embed_dim=1024), FakeEncoder(dim=4))
    with pytest.raises(ExternalProtocolError):
        embedder.embed(["док #1"])


def test_local_embed_rejects_nan() -> None:
    class NanEncoder(FakeEncoder):
        def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
            return [[math.nan] * self.dim for _ in sentences]

    with pytest.raises(ExternalProtocolError) as excinfo:
        LocalEmbedder(local_config(), NanEncoder()).embed(["док"])
    assert "float32" in str(excinfo.value)


def test_local_embed_renormalizes_bf16_drift() -> None:
    # Норма 1.0026 — реальный выход Giga-Embeddings в bfloat16 на MPS.
    class DriftEncoder(FakeEncoder):
        def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
            return [[1.002577 / math.sqrt(self.dim)] * self.dim for _ in sentences]

    [vector] = LocalEmbedder(local_config(), DriftEncoder()).embed(["док"])
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0, abs=1e-9)


def test_local_rerank_sorts_with_global_indices() -> None:
    reranker = LocalReranker(local_config(), FakeCrossEncoder())
    ranked = reranker.rerank("запрос", ["д #5", "д #30", "д #10"])
    assert ranked == [(1, 0.30), (2, 0.10), (0, 0.05)]


def test_local_rerank_rejects_logits() -> None:
    reranker = LocalReranker(local_config(), FakeCrossEncoder(scores=[3.2, -1.0]))
    with pytest.raises(ExternalProtocolError):
        reranker.rerank("запрос", ["а", "б"])


def test_local_empty_input_does_not_load_model() -> None:
    # model=None и пустой вход: загрузчик не вызывается, torch не нужен.
    assert LocalEmbedder(local_config()).embed([]) == []
    assert LocalReranker(local_config()).rerank("запрос", []) == []


def test_factory_switches_by_config() -> None:
    local = build_providers(Settings(external={"provider": "local"}))
    assert isinstance(local.embedder, LocalEmbedder)
    assert local.backend is None
    local.close()

    external = build_providers(Settings(external={"provider": "external", "token": "t"}))
    assert isinstance(external.embedder, ExternalEmbedder)
    assert isinstance(external.reranker, ExternalReranker)
    # Транспорт общий: лимит параллелизма относится к сервису, а не к клиенту.
    assert external.embedder._backend is external.reranker._backend
    external.close()


def test_factory_requires_token_for_external() -> None:
    # provider задан явно: config.yaml рабочей копии стоит в профиле ноутбука (local).
    with pytest.raises(ConfigError) as excinfo:
        build_providers(Settings(external={"provider": "external"}))
    assert "BZPO_EXTERNAL__TOKEN" in str(excinfo.value)


# ------------------------------------------------------------------- Ollama


def ollama_config(**overrides: Any) -> OllamaConfig:
    defaults: dict[str, Any] = {"base_url": "http://ollama.test", "num_predict": 3000}
    return OllamaConfig(**{**defaults, **overrides})


def test_ollama_sends_mandatory_options() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gigachat",
                "response": "текст ответа",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1200,
                "eval_count": 300,
                "total_duration": 5_000_000_000,
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ollama = OllamaClient(ollama_config(), role="query", client=client)
    result = ollama.generate("промпт", temperature=0.0, seed=42)

    assert captured["path"] == "/api/generate"
    payload = captured["payload"]
    assert payload["stream"] is False
    assert payload["keep_alive"] == -1
    assert payload["options"]["num_ctx"] == 16384
    assert payload["options"]["num_predict"] == 3000
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["seed"] == 42
    assert payload["think"] is False

    assert result.text == "текст ответа"
    assert result.prompt_tokens == 1200
    assert result.completion_tokens == 300
    assert result.total_duration_ms == pytest.approx(5000.0)
    assert result.truncated_by_length is False


@pytest.mark.parametrize(
    ("role", "expected"),
    [("query", "модель-запросов"), ("enrichment", "модель-обогащения")],
)
def test_ollama_picks_model_by_role(role: str, expected: str) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "", "done": True})

    config = ollama_config(model_query="модель-запросов", model_enrichment="модель-обогащения")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    OllamaClient(config, role=role, client=client).generate("промпт")
    assert captured["payload"]["model"] == expected


def test_ollama_system_prompt_only_when_given() -> None:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "", "done": True})

    ollama = OllamaClient(
        ollama_config(), role="query", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    ollama.generate("промпт", system="правила стиля")
    ollama.generate("промпт")
    assert payloads[0]["system"] == "правила стиля"
    # Без явного system в запросе нет поля: остаётся системный промпт из Modelfile.
    assert "system" not in payloads[1]


def test_ollama_json_format_flag() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "{}", "done": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    OllamaClient(ollama_config(), role="query", client=client).generate("промпт", json_format=True)
    assert captured["payload"]["format"] == "json"


def test_ollama_reports_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("нет соединения", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(OllamaUnavailable):
        OllamaClient(ollama_config(), role="query", client=client).generate("промпт")


def test_ollama_reports_bad_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(OllamaError) as excinfo:
        OllamaClient(ollama_config(), role="query", client=client).generate("промпт")
    assert "404" in str(excinfo.value)


def test_ollama_requires_num_predict() -> None:
    """OllamaConfig без num_predict приходит только в обход Settings — это ошибка."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - не вызывается
        return httpx.Response(200, json={"response": "", "done": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="num_predict"):
        OllamaClient(OllamaConfig(), role="query", client=client).generate("промпт")
