"""Конфигурация БЗПО (SPEC раздел 10).

Механизм: `config.yaml` + переменные окружения с префиксом `BZPO_`, валидация
при старте, хэш итоговой конфигурации пишется в каждую трассу (SPEC раздел 8).

Три правила закреплены здесь кодом, а не соглашением:

1. Секреты (токен внешнего сервиса, DSN базы) читаются ТОЛЬКО из окружения.
   Секрет, найденный в config.yaml, — ошибка загрузки, а не предупреждение:
   файл лежит в git, и однажды попавший туда токен оттуда уже не убрать.
2. Невалидная конфигурация роняет старт. Молчаливый дефолт опаснее отказа:
   дефекты этого класса (num_ctx=2048 у Ollama, candidates > per_list_k)
   не дают симптомов, они просто тихо ухудшают выдачу.
3. `config_hash` считается по конфигурации без секретов и не меняется при
   ротации токена — иначе трассы разных дней несравнимы на ровном месте.

Имена полей повторяют имена параметров SPEC раздела 10 в нижнем регистре,
чтобы поиск по имени параметра из спецификации приводил прямо в код.
Переопределение из окружения — через двойное подчёркивание:
`BZPO_SEARCH__TOP_K_FINAL=5`, `BZPO_EXTERNAL__TOKEN=...`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import typing
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config.yaml"

#: Ключи, которые запрещено держать в config.yaml на любом уровне вложенности.
FORBIDDEN_FILE_KEYS = frozenset({"token", "dsn", "password", "api_key", "secret"})

Publicity = Literal["open", "anon", "closed"]


class ConfigError(RuntimeError):
    """Ошибка загрузки конфигурации: битый YAML, секрет в файле, нечитаемый путь."""


class _Section(BaseModel):
    """Базовый класс секции. extra=forbid ловит опечатки в config.yaml сразу."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ChunkingConfig(_Section):
    """Чанкинг, дедупликация, векторизация (SPEC 4.2, раздел 10)."""

    chunk_target: Annotated[int, Field(gt=0)] = 500
    chunk_max: Annotated[int, Field(gt=0)] = 1000
    chunk_min: Annotated[int, Field(gt=0)] = 100
    chunk_overlap: Annotated[float, Field(ge=0.0, lt=0.5)] = 0.15
    # Предел длины входа эмбеддера в токенах; фактическое значение — из probe_api.
    chunk_hard_limit_embed: Annotated[int, Field(gt=0)] = 8192
    section_max_tokens: Annotated[int, Field(gt=0)] = 4000
    # Пороги определены на косинусной близости L2-нормализованных векторов
    # по embedding_raw (SPEC раздел 3, замечание recs.md п.2).
    dedup_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.95
    dedup_threshold_fallback: Annotated[float, Field(gt=0.0, le=1.0)] = 0.97


class SearchConfig(_Section):
    """Онлайн-поиск: 4 списка -> weighted-RRF -> реранк -> boost -> MMR (SPEC 5.2)."""

    per_list_k: Annotated[int, Field(gt=0)] = 100
    rrf_k: Annotated[int, Field(gt=0)] = 60
    rrf_weight_hyde: Annotated[float, Field(gt=0.0, le=1.0)] = 0.7
    candidates: Annotated[int, Field(gt=0)] = 50
    top_k_final: Annotated[int, Field(gt=0)] = 8
    mmr_lambda: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    # None до калибровки на >=15 запросах без ответа (SPEC раздел 9). Отсечение
    # «не найдено» до калибровки не работает — это осознанное состояние, а не
    # забытый параметр: потребитель обязан проверить None и не отсекать ничего.
    rerank_threshold: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    rerank_input: Literal["text_raw", "text_enriched"] = "text_raw"
    boost_w_industry: Annotated[float, Field(ge=0.0)] = 0.05
    boost_w_won: Annotated[float, Field(ge=0.0)] = 0.05
    boost_w_fresh: Annotated[float, Field(ge=0.0)] = 0.03
    # Горизонт затухания свежести: max(0, 1 - возраст_лет / горизонт).
    boost_fresh_horizon_years: Annotated[float, Field(gt=0.0)] = 5.0
    section_exemplars: Annotated[int, Field(gt=0)] = 2
    # Фильтр публичности. На PoC только 'open': проекты anon исключаются
    # конфигурацией, разметка в данных не мутируется (SPEC раздел 12).
    allowed_publicity: list[Publicity] = ["open"]
    # None означает «взять активную версию из index_versions.status='active'».
    active_index_version: str | None = None


class GenerationConfig(_Section):
    """Сборка контекста и генерация раздела (SPEC 5.3, раздел 10)."""

    # SPEC называет 12000, но тройка (12000, x1.2, 3000) не помещается в окно
    # 16384 Ollama: 12000*1.2 + 3000 = 17400. Значение уменьшено до максимума,
    # который проходит кросс-проверку; расхождение со SPEC зафиксировано в README.
    context_budget_tokens: Annotated[int, Field(gt=0)] = 11000
    gen_max_tokens: Annotated[int, Field(gt=0)] = 3000
    # Запас на расхождение токенизаторов bge-m3 и GigaChat (SPEC раздел 1).
    context_tokenizer_margin: Annotated[float, Field(ge=1.0)] = 1.2
    classifier_conf_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7


class GraphConfig(_Section):
    """Граф связей (MVP+, SPEC 4.2 шаг 10)."""

    graph_edge_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.80
    graph_depth: Annotated[int, Field(ge=0)] = 1


class FeatureFlags(_Section):
    """Включение слоёв. Слой разбора отключается флагом по итогам этапа 3.5."""

    use_hyde: bool = True
    use_decomposition: bool = True
    use_graph_expansion: bool = False


class ExternalServiceConfig(_Section):
    """Внешний сервис эмбеддингов и реранкинга (SPEC 6.1–6.4).

    Диалект реранкера задаётся конфигурацией и не определяется автоматически:
    автоопределение по коду ответа маскирует смену контракта сервиса — вместо
    громкой ошибки система молча уйдёт на другой формат разбора.
    """

    # local — модели в процессе (секция `local`), external — этот HTTP-сервис.
    provider: Literal["external", "local"] = "external"
    base_url: str = "http://127.0.0.1:8080"
    embed_path: str = "/v1/embeddings"
    rerank_dialect: Literal["tei", "cohere"] = "tei"
    # None -> путь выводится из диалекта: TEI /rerank, Cohere/Jina/vLLM /v1/rerank.
    rerank_path: str | None = None
    health_path: str = "/v1/models"

    embed_model: str = "BAAI/bge-m3"
    # Ревизия входит в ключ кэша векторов: без неё кэш после обновления модели
    # молча отдаёт несовместимые векторы (SPEC 6.3).
    embed_model_revision: str = "unknown"
    embed_dim: Annotated[int, Field(gt=0)] = 1024
    # SPEC перечисляет EMBED_BATCH в строке «векторизация», но батчирование
    # выполняет модуль изоляции, и значение определяется пределом сервиса,
    # а не параметрами чанкинга — поэтому параметр живёт здесь.
    embed_batch: Annotated[int, Field(gt=0)] = 128
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # Сколько пар реранкер принимает за вызов — блокирующий вопрос SPEC 6.2;
    # до ответа исходим из «весь срез кандидатов одним запросом».
    rerank_batch: Annotated[int, Field(gt=0)] = 50
    # Усечение входа. Токенизатор bge-m3 в модуль изоляции не тянем, поэтому
    # предел задан в символах: chunk_hard_limit_embed (токены) x ~2.5 символа
    # на токен для русского XLM-R. Уточняется вместе с probe_api.
    max_input_chars: Annotated[int, Field(gt=0)] = 20000

    # Повторы: SPEC 6.3 «3 попытки, задержки 1/3/9 с». Трактовка зафиксирована:
    # retry_attempts — число ПОВТОРОВ после первой попытки, задержки берутся
    # из retry_delays поэлементно, поэтому всего запросов = 1 + retry_attempts.
    retry_attempts: Annotated[int, Field(ge=0)] = 3
    retry_delays: list[Annotated[float, Field(ge=0.0)]] = [1.0, 3.0, 9.0]
    # Ограничение на Retry-After: сервис может отдать десятки минут, ждать столько
    # внутри запроса пользователя бессмысленно — бюджет 60 с (SPEC раздел 11).
    retry_after_max: Annotated[float, Field(ge=0.0)] = 60.0

    timeout_connect: Annotated[float, Field(gt=0.0)] = 5.0
    timeout_read: Annotated[float, Field(gt=0.0)] = 60.0
    max_concurrency: Annotated[int, Field(gt=0)] = 4
    rate_limit_rps: Annotated[float, Field(gt=0.0)] = 20.0
    health_interval: Annotated[float, Field(gt=0.0)] = 30.0
    # Кэш векторов запросов; гипотезы HyDE не кэшируются (недетерминированы).
    query_cache_size: Annotated[int, Field(ge=0)] = 4096

    token: SecretStr | None = None

    @property
    def effective_rerank_path(self) -> str:
        """Путь реранкера с учётом диалекта."""
        if self.rerank_path is not None:
            return self.rerank_path
        return "/rerank" if self.rerank_dialect == "tei" else "/v1/rerank"


#: Размерность колонок vector(1024) в миграции 0001. Модель с другой размерностью
#: требует новой миграции и переиндексации, а не правки конфигурации.
SCHEMA_EMBED_DIM = 1024


class LocalModelsConfig(_Section):
    """Эмбеддер и реранкер в процессе приложения (external.provider = local).

    Профиль ноутбука: Apple Silicon, 16 ГБ общей памяти, ускоритель MPS.
    Модели скачиваются с Hugging Face при первом запуске и кэшируются в
    ~/.cache/huggingface; дальше сеть не нужна.
    """

    # 480M: 1024 измерения совпадают со схемой, ~1 ГБ памяти. 3B (2048 измерений)
    # не проходит в HNSW-индекс pgvector для типа vector (предел 2000) и съедает
    # ~6 ГБ, которых на 16 ГБ рядом с LLM нет.
    embed_model: str = "ai-sage/Giga-Embeddings-instruct-480M-0826"
    # Ревизия (commit на HF) входит в ключ кэша векторов запросов: после
    # обновления весов кэш обязан стать недействительным.
    embed_model_revision: str = "main"
    embed_dim: Annotated[int, Field(gt=0)] = 1024
    # Модель асимметрична: инструкция ставится только перед запросом, тексты
    # корпуса кодируются как есть (карточка модели).
    query_instruction: str = (
        "Instruct: Given a query, retrieve relevant passages\nQuery: "
    )
    # Предел входа в токенах токенизатора модели (у модели — 8192); длиннее —
    # усечение с конца. На 16 ГБ разумно снизить вместе с chunk_hard_limit_embed.
    embed_max_tokens: Annotated[int, Field(gt=0)] = 8192
    embed_batch: Annotated[int, Field(gt=0)] = 16

    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_tokens: Annotated[int, Field(gt=0)] = 1024
    rerank_batch: Annotated[int, Field(gt=0)] = 16

    # auto: mps, если доступен, иначе cpu.
    device: Literal["auto", "mps", "cpu", "cuda"] = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    # Giga-Embeddings — собственная архитектура (Qwen3 с двунаправленным
    # вниманием), без исполнения кода из репозитория модели не загружается.
    trust_remote_code: bool = True
    query_cache_size: Annotated[int, Field(ge=0)] = 4096


OllamaRole = Literal["query", "enrichment", "vision", "ocr"]


class OllamaConfig(_Section):
    """Локальная генерация (SPEC раздел 10, «Параметры Ollama обязательные»).

    Модели разведены по ролям. query — всё, что выполняется на запрос
    пользователя: классификация, HyDE, генерация раздела. enrichment —
    контекстные шапки чанков при индексации (SPEC 4.3): тысячи коротких
    вызовов, где решает скорость. Обе модели в 16 ГБ рядом с эмбеддером не
    помещаются; Ollama с OLLAMA_MAX_LOADED_MODELS=1 выгружает одну и
    загружает другую (~4 с), а индексация и запросы по времени не пересекаются.
    """

    base_url: str = "http://127.0.0.1:11434"
    # Qwen3.5 9B: плотная, Q4_K_M ~6.6 ГБ, ~23 ток/с на M5. В сравнении на
    # разделе заявки держит маркеры [N] и не выдумывает цифр; GigaChat Lightning
    # на той же задаче сочинил абзац фактов и сослался на несуществующий [4].
    model_query: str = "qwen3.5:9b"
    # GigaChat 3.1 Lightning: MoE 10B, 1.8B активных, Q4_K_M ~6.5 ГБ, ~75 ток/с.
    # Шапку чанка пишет точно и втрое быстрее. Тег — прямая загрузка GGUF с HF.
    model_enrichment: str = "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:Q4_K_M"
    # vision — разбор картинок слайдов при конвертации (docs/PARSING_PLAN.md, П2).
    # Версия instruct: «думающая» qwen3-vl:8b игнорирует think=false и тратит
    # весь лимит на рассуждения.
    model_vision: str = "qwen3-vl:8b-instruct"
    # ocr — текст со скриншотов слайдов в Word (parsing/ocr.py): дословно, без пересказа.
    # Qwen3-VL с промптом транскрипции точнее GLM-OCR на русском (см. parsing/ocr.py).
    model_ocr: str = "qwen3-vl:8b-instruct"
    # Qwen3.5 по умолчанию «думает»: скрытые рассуждения съедают num_predict и
    # бюджет латентности. Модели без режима рассуждений значение игнорируют.
    think: bool = False
    # Дефолт Ollama 2048 молча усёк бы контекст с разделами-образцами.
    num_ctx: Annotated[int, Field(gt=0)] = 16384
    # None -> подставляется gen_max_tokens при валидации.
    num_predict: Annotated[int | None, Field(gt=0)] = None
    # -1 — модель резидентна в памяти; перезагрузка весов на каждый запрос
    # съедает бюджет латентности целиком.
    keep_alive: int | str = -1
    timeout_connect: Annotated[float, Field(gt=0.0)] = 5.0
    # 3000 токенов на ~23 ток/с (Qwen3.5 9B на M5) — ~130 с; запас на загрузку модели.
    timeout_read: Annotated[float, Field(gt=0.0)] = 180.0

    def model_for(self, role: OllamaRole) -> str:
        return {"query": self.model_query, "enrichment": self.model_enrichment,
                "vision": self.model_vision, "ocr": self.model_ocr}[role]


class DatabaseConfig(_Section):
    """PostgreSQL. DSN — только из окружения (BZPO_DB__DSN)."""

    dsn: SecretStr | None = None
    statement_timeout_ms: Annotated[int, Field(gt=0)] = 30_000
    # None -> каталог bzpo/migrations внутри пакета.
    migrations_dir: Path | None = None


class Settings(BaseSettings):
    """Итоговая конфигурация. Собирается load_config(), валидируется при старте."""

    model_config = SettingsConfigDict(
        env_prefix="BZPO_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    # Путь к config.yaml объявлен полем, иначе BZPO_CONFIG_FILE упрётся в extra=forbid.
    config_file: Path | None = None
    app_env: Literal["dev", "stage", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    chunking: ChunkingConfig = ChunkingConfig()
    search: SearchConfig = SearchConfig()
    generation: GenerationConfig = GenerationConfig()
    graph: GraphConfig = GraphConfig()
    flags: FeatureFlags = FeatureFlags()
    external: ExternalServiceConfig = ExternalServiceConfig()
    local: LocalModelsConfig = LocalModelsConfig()
    ollama: OllamaConfig = OllamaConfig()
    db: DatabaseConfig = DatabaseConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Приоритет: аргументы конструктора -> окружение -> .env -> config.yaml -> дефолты.
        # Окружение выше файла: секреты и стендовые переопределения не должны
        # требовать правки версионируемого файла.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls, _resolve_yaml_path()),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _cross_checks(self) -> Settings:
        """Кросс-проверки между секциями: то, что нельзя выразить в одном поле."""
        problems: list[str] = []

        if not self.chunking.chunk_min < self.chunking.chunk_target <= self.chunking.chunk_max:
            problems.append(
                "чанкинг: требуется chunk_min < chunk_target <= chunk_max, получено "
                f"{self.chunking.chunk_min} / {self.chunking.chunk_target} / "
                f"{self.chunking.chunk_max}"
            )
        if self.chunking.section_max_tokens > self.chunking.chunk_hard_limit_embed:
            problems.append(
                f"section_max_tokens ({self.chunking.section_max_tokens}) превышает предел "
                f"входа эмбеддера chunk_hard_limit_embed "
                f"({self.chunking.chunk_hard_limit_embed}): секции будут усечены молча"
            )
        if self.chunking.dedup_threshold_fallback < self.chunking.dedup_threshold:
            problems.append(
                "dedup_threshold_fallback — запасной, более строгий порог: он не может быть "
                f"ниже основного ({self.chunking.dedup_threshold_fallback} < "
                f"{self.chunking.dedup_threshold})"
            )

        if not self.search.top_k_final <= self.search.candidates <= self.search.per_list_k:
            problems.append(
                "поиск: требуется top_k_final <= candidates <= per_list_k, получено "
                f"{self.search.top_k_final} / {self.search.candidates} / {self.search.per_list_k}"
            )
        if not self.search.allowed_publicity:
            problems.append("allowed_publicity пуст: поиск вернёт пустую выдачу на любой запрос")
        if len(set(self.search.allowed_publicity)) != len(self.search.allowed_publicity):
            problems.append(f"allowed_publicity содержит дубли: {self.search.allowed_publicity}")

        if len(self.external.retry_delays) != self.external.retry_attempts:
            problems.append(
                f"retry_delays ({len(self.external.retry_delays)} значений) не соответствует "
                f"retry_attempts ({self.external.retry_attempts}): задержка берётся на каждый "
                "повтор поэлементно"
            )

        active_dim = (
            self.local.embed_dim if self.external.provider == "local" else self.external.embed_dim
        )
        if active_dim != SCHEMA_EMBED_DIM:
            problems.append(
                f"размерность эмбеддера {active_dim} не совпадает со схемой БД "
                f"vector({SCHEMA_EMBED_DIM}): нужна миграция и переиндексация"
            )
        if (
            self.external.provider == "local"
            and self.chunking.chunk_hard_limit_embed > self.local.embed_max_tokens
        ):
            problems.append(
                f"chunk_hard_limit_embed ({self.chunking.chunk_hard_limit_embed}) больше "
                f"local.embed_max_tokens ({self.local.embed_max_tokens}): длинные секции "
                "будут усечены эмбеддером молча"
            )

        # Окно Ollama обязано вместить промпт (с запасом на токенизатор GigaChat)
        # и ответ целиком. Иначе llama.cpp усечёт начало промпта без сообщения.
        if self.ollama.num_predict is None:
            self.ollama.num_predict = self.generation.gen_max_tokens
        required = (
            self.generation.context_budget_tokens * self.generation.context_tokenizer_margin
            + self.generation.gen_max_tokens
        )
        if required > self.ollama.num_ctx:
            allowed = int(
                (self.ollama.num_ctx - self.generation.gen_max_tokens)
                / self.generation.context_tokenizer_margin
            )
            problems.append(
                "бюджет контекста не помещается в окно модели: "
                f"{self.generation.context_budget_tokens} x "
                f"{self.generation.context_tokenizer_margin} + "
                f"{self.generation.gen_max_tokens} = {required:.0f} > num_ctx "
                f"{self.ollama.num_ctx}; допустимый context_budget_tokens <= {allowed}"
            )
        if self.ollama.num_predict != self.generation.gen_max_tokens:
            problems.append(
                f"ollama.num_predict ({self.ollama.num_predict}) != gen_max_tokens "
                f"({self.generation.gen_max_tokens}): SPEC раздел 10 требует их совпадения"
            )

        if problems:
            raise ValueError("конфигурация невалидна:\n  - " + "\n  - ".join(problems))
        return self

    def public_dump(self) -> dict[str, Any]:
        """Конфигурация без секретов — для трассы, логов и config_hash."""
        return _public_payload(self)

    @property
    def config_hash(self) -> str:
        """sha256 конфигурации без секретов; пишется в каждую трассу (SPEC раздел 8)."""
        payload = json.dumps(
            self.public_dump(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def migrations_path(self) -> Path:
        """Каталог с нумерованными SQL-миграциями."""
        return self.db.migrations_dir or (Path(__file__).resolve().parent / "migrations")


# --------------------------------------------------------------------------- YAML


class _YamlSource(PydanticBaseSettingsSource):
    """Источник значений из config.yaml с запретом секретов в файле."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path | None) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Значения отдаются целиком в __call__, пофайловый доступ не нужен.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if self._path is None or not self._path.exists():
            # Отсутствующий config.yaml — легальная ситуация: дефолты валидны.
            return {}
        try:
            raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"не удалось прочитать конфигурацию {self._path}: {exc}") from exc
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ConfigError(
                f"{self._path}: ожидался словарь параметров, получено {type(raw).__name__}"
            )
        data = _lower_keys(raw)
        _forbid_secrets(data, self._path)
        return data


def _lower_keys(node: Any) -> Any:
    """Ключи YAML приводятся к нижнему регистру: в SPEC параметры записаны заглавными."""
    if isinstance(node, dict):
        return {str(k).lower(): _lower_keys(v) for k, v in node.items()}
    return node


def _forbid_secrets(node: Any, path: Path, trail: str = "") -> None:
    """Обход дерева конфигурации в поисках секретов, попавших в файл."""
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        where = f"{trail}.{key}" if trail else str(key)
        if key in FORBIDDEN_FILE_KEYS and value is not None:
            env_name = "BZPO_" + where.upper().replace(".", "__")
            raise ConfigError(
                f"{path}: секрет «{where}» не может храниться в конфигурационном файле — "
                f"файл версионируется в git. Задайте переменную окружения {env_name}"
            )
        _forbid_secrets(value, path, where)


# ------------------------------------------------------------------ секреты и хэш


def _is_secret_annotation(annotation: Any) -> bool:
    """Секретное ли поле — определяется по аннотации, включая `SecretStr | None`."""
    if annotation is SecretStr:
        return True
    return any(arg is SecretStr for arg in typing.get_args(annotation))


def _public_payload(model: BaseModel) -> dict[str, Any]:
    """Рекурсивный дамп без секретных полей.

    Секретные поля выбрасываются по аннотации, а не по значению: иначе хэш
    конфигурации менялся бы от самого факта наличия токена, и трассы стенда
    с токеном и без него не сравнивались бы.
    """
    out: dict[str, Any] = {}
    for name, field in type(model).model_fields.items():
        if _is_secret_annotation(field.annotation):
            continue
        value = getattr(model, name)
        out[name] = _public_payload(value) if isinstance(value, BaseModel) else value
    return out


# ------------------------------------------------------------------- загрузка


_UNSET = object()
_yaml_path_override: Any = _UNSET
_load_lock = threading.RLock()


def _resolve_yaml_path() -> Path | None:
    """Путь к config.yaml: явный аргумент load_config -> BZPO_CONFIG_FILE -> корень проекта."""
    if _yaml_path_override is not _UNSET:
        return typing.cast("Path | None", _yaml_path_override)
    from_env = os.environ.get("BZPO_CONFIG_FILE")
    if from_env:
        return Path(from_env)
    return DEFAULT_CONFIG_FILE


def load_config(config_path: str | Path | None = None, *, use_yaml: bool = True) -> Settings:
    """Собрать и провалидировать конфигурацию.

    `use_yaml=False` полностью отключает файловый источник — этим пользуются
    тесты, чтобы не зависеть от config.yaml рабочей копии.
    Нарушения кросс-проверок поднимаются одной ошибкой со всем перечнем сразу,
    а не по одному за прогон.
    """
    global _yaml_path_override
    with _load_lock:
        if not use_yaml:
            _yaml_path_override = None
        elif config_path is not None:
            _yaml_path_override = Path(config_path)
        else:
            _yaml_path_override = _UNSET
        try:
            return Settings()
        finally:
            _yaml_path_override = _UNSET


_settings: Settings | None = None


def get_settings() -> Settings:
    """Конфигурация процесса (ленивый синглтон)."""
    global _settings
    with _load_lock:
        if _settings is None:
            _settings = load_config()
        return _settings


def reset_settings_cache() -> None:
    """Сбросить синглтон. Нужен тестам и перечитыванию конфигурации в CLI."""
    global _settings
    with _load_lock:
        _settings = None
