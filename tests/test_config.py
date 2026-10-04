"""Тесты конфигурации (PLAN 0.5.1, SPEC раздел 10).

Проверяется четыре свойства: дефолты валидны, невалидная конфигурация роняет
старт, config_hash стабилен и не содержит секретов, окружение переопределяет файл.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bzpo.config import PROJECT_ROOT, ConfigError, Settings, load_config, reset_settings_cache

# --------------------------------------------------------------- дефолты


def test_defaults_are_valid() -> None:
    config = load_config(use_yaml=False)

    assert config.chunking.chunk_target == 500
    assert config.search.per_list_k == 100
    assert config.search.candidates == 50
    assert config.search.top_k_final == 8
    assert config.search.rrf_k == 60
    assert config.search.rrf_weight_hyde == 0.7
    assert config.search.mmr_lambda == 0.7
    # Порог отсечения «не найдено» калибруется на этапе 3.6 — до тех пор None.
    assert config.search.rerank_threshold is None
    assert config.search.allowed_publicity == ["open"]
    assert config.external.retry_delays == [1.0, 3.0, 9.0]
    assert config.external.timeout_connect == 5.0
    assert config.external.timeout_read == 60.0
    assert config.external.max_concurrency == 4
    assert config.external.health_interval == 30.0
    assert config.ollama.num_ctx == 16384
    assert config.ollama.keep_alive == -1
    # num_predict не задан в файле и подставляется из gen_max_tokens (SPEC 10).
    assert config.ollama.num_predict == config.generation.gen_max_tokens == 3000


def test_repo_config_yaml_is_valid() -> None:
    """Образец config.yaml в корне обязан загружаться: он же и есть стенд."""
    config = load_config(PROJECT_ROOT / "config.yaml")
    assert config.external.embed_batch == 128
    assert config.external.rerank_dialect in {"tei", "cohere"}


def test_rerank_path_follows_dialect() -> None:
    tei = load_config(use_yaml=False)
    assert tei.external.effective_rerank_path == "/rerank"

    cohere = Settings(external={"rerank_dialect": "cohere"})
    assert cohere.external.effective_rerank_path == "/v1/rerank"


# ------------------------------------------------------ невалидные конфигурации


def test_swapped_chunk_bounds_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Перепутанные chunk_min/chunk_max должны ронять старт, а не «работать»."""
    monkeypatch.setenv("BZPO_CHUNKING__CHUNK_MIN", "1000")
    monkeypatch.setenv("BZPO_CHUNKING__CHUNK_MAX", "100")

    with pytest.raises(ValidationError) as excinfo:
        load_config(use_yaml=False)
    assert "chunk_min < chunk_target <= chunk_max" in str(excinfo.value)


def test_context_budget_overflow_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """12000 x 1.2 + 3000 = 17400 > 16384 — значение SPEC не помещается в окно.

    Тест фиксирует расхождение: конфигурация с CONTEXT_BUDGET_TOKENS=12000 при
    num_ctx=16384 отвергается, потому что иначе llama.cpp усечёт промпт молча.
    """
    monkeypatch.setenv("BZPO_GENERATION__CONTEXT_BUDGET_TOKENS", "12000")

    with pytest.raises(ValidationError) as excinfo:
        load_config(use_yaml=False)
    message = str(excinfo.value)
    assert "не помещается в окно модели" in message
    assert "num_ctx" in message


def test_search_depth_order_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """top_k_final <= candidates <= per_list_k: иначе выдача режется раньше времени."""
    monkeypatch.setenv("BZPO_SEARCH__CANDIDATES", "5")

    with pytest.raises(ValidationError) as excinfo:
        load_config(use_yaml=False)
    assert "top_k_final <= candidates <= per_list_k" in str(excinfo.value)


def test_retry_delays_must_match_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BZPO_EXTERNAL__RETRY_DELAYS", '["1.0", "3.0"]')

    with pytest.raises(ValidationError) as excinfo:
        load_config(use_yaml=False)
    assert "retry_delays" in str(excinfo.value)


def test_empty_allowed_publicity_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BZPO_SEARCH__ALLOWED_PUBLICITY", "[]")

    with pytest.raises(ValidationError) as excinfo:
        load_config(use_yaml=False)
    assert "allowed_publicity" in str(excinfo.value)


def test_unknown_key_in_yaml_fails(tmp_path: Path) -> None:
    """Опечатка в имени параметра — ошибка, а не молчаливо проигнорированная строка."""
    path = tmp_path / "config.yaml"
    path.write_text("search:\n  top_k_finall: 4\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_broken_yaml_fails(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("search: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "не удалось прочитать конфигурацию" in str(excinfo.value)


def test_secret_in_yaml_is_rejected(tmp_path: Path) -> None:
    """Токен в версионируемом файле — ошибка загрузки: из git его уже не убрать."""
    path = tmp_path / "config.yaml"
    path.write_text('external:\n  token: "s3cret"\n', encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    message = str(excinfo.value)
    assert "BZPO_EXTERNAL__TOKEN" in message
    assert "s3cret" not in message


# ------------------------------------------------------------ окружение и хэш


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BZPO_SEARCH__TOP_K_FINAL", "5")
    monkeypatch.setenv("BZPO_FLAGS__USE_HYDE", "false")

    config = load_config(use_yaml=False)
    assert config.search.top_k_final == 5
    assert config.flags.use_hyde is False


def test_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("search:\n  top_k_final: 6\n", encoding="utf-8")

    assert load_config(path).search.top_k_final == 6

    monkeypatch.setenv("BZPO_SEARCH__TOP_K_FINAL", "7")
    assert load_config(path).search.top_k_final == 7


def test_uppercase_keys_in_yaml_are_accepted(tmp_path: Path) -> None:
    """В SPEC параметры записаны заглавными; регистр ключей файла не важен."""
    path = tmp_path / "config.yaml"
    path.write_text("SEARCH:\n  TOP_K_FINAL: 4\n", encoding="utf-8")

    assert load_config(path).search.top_k_final == 4


def test_config_hash_is_stable_and_secret_free(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = load_config(use_yaml=False).config_hash
    assert baseline == load_config(use_yaml=False).config_hash

    monkeypatch.setenv("BZPO_EXTERNAL__TOKEN", "very-secret-token")
    monkeypatch.setenv("BZPO_DB__DSN", "postgresql://user:pass@host/db")
    with_secrets = load_config(use_yaml=False)

    # Ротация токена не должна менять хэш конфигурации: иначе трассы разных
    # дней несравнимы без единой содержательной причины.
    assert with_secrets.config_hash == baseline

    dumped = json.dumps(with_secrets.public_dump(), ensure_ascii=False, default=str)
    assert "very-secret-token" not in dumped
    assert "postgresql://" not in dumped
    assert "token" not in with_secrets.public_dump()["external"]
    assert "dsn" not in with_secrets.public_dump()["db"]


def test_config_hash_changes_with_meaningful_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = load_config(use_yaml=False).config_hash
    monkeypatch.setenv("BZPO_SEARCH__MMR_LAMBDA", "0.5")
    assert load_config(use_yaml=False).config_hash != baseline


def test_settings_singleton_resets() -> None:
    reset_settings_cache()
    from bzpo.config import get_settings

    first = get_settings()
    assert get_settings() is first
    reset_settings_cache()
    assert get_settings() is not first


def test_embed_dim_must_match_schema() -> None:
    """Giga-Embeddings-3B (2048) без новой миграции не подключить."""
    with pytest.raises(ValueError) as excinfo:
        Settings(external={"provider": "local"}, local={"embed_dim": 2048})
    assert "vector(1024)" in str(excinfo.value)
    # Размерность неактивного провайдера не проверяется.
    Settings(external={"provider": "external"}, local={"embed_dim": 2048})


def test_local_embed_limit_covers_chunking() -> None:
    with pytest.raises(ValueError) as excinfo:
        Settings(
            external={"provider": "local"},
            chunking={"chunk_hard_limit_embed": 8192, "section_max_tokens": 4000},
            local={"embed_max_tokens": 4096},
        )
    assert "embed_max_tokens" in str(excinfo.value)


def test_repo_config_is_laptop_profile() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    assert config.external.provider == "local"
    assert config.local.query_instruction.endswith("\nQuery: ")
