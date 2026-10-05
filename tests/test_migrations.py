"""Тесты миграции 0001 и раннера без живой базы (SPEC раздел 3, PLAN 0.5.2).

Применение на чистой PostgreSQL проверяется вручную на стенде (критерий этапа
0.5). Здесь проверяется то, что можно проверить без базы и что ломается чаще
всего: состав и ПОРЯДОК таблиц. Порядок обязателен — DDL исполняется сверху
вниз, и ссылка на ещё не созданную таблицу роняет всю миграцию (recs.md п.1).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bzpo.storage.migrate import (
    MigrationError,
    discover_migrations,
    pending_migrations,
)

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "bzpo" / "migrations"
INITIAL = MIGRATIONS_DIR / "0001_initial.sql"

EXPECTED_ORDER = [
    "index_versions",
    "clients",
    "projects",
    "registry_uploads",
    "documents",
    "sections",
    "chunks",
    "artifacts",
    "chunk_edges",
]

CREATE_TABLE_RE = re.compile(r"^CREATE TABLE (?:IF NOT EXISTS )?(\w+)", re.MULTILINE)


@pytest.fixture(scope="module")
def sql() -> str:
    return INITIAL.read_text(encoding="utf-8")


def test_initial_migration_exists() -> None:
    assert INITIAL.is_file()


def test_vector_extension_is_created_first(sql: str) -> None:
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert sql.index("CREATE EXTENSION") < sql.index("CREATE TABLE")


def test_all_nine_tables_in_required_order(sql: str) -> None:
    assert CREATE_TABLE_RE.findall(sql) == EXPECTED_ORDER


def test_foreign_keys_reference_already_created_tables(sql: str) -> None:
    """Ни одна таблица не ссылается на ту, что создаётся ниже по скрипту."""
    statements = sql.split("CREATE TABLE ")
    created: set[str] = set()
    for statement in statements[1:]:
        name = statement.split("(", 1)[0].strip().split()[0]
        for referenced in re.findall(r"REFERENCES (\w+)", statement):
            assert referenced in created or referenced == name, (
                f"таблица {name} ссылается на {referenced}, которая создаётся позже"
            )
        created.add(name)


def test_critical_columns_are_present(sql: str) -> None:
    """Колонки, потеря которых ломает целые механизмы SPEC."""
    # Два вектора: обогащённый для поиска, сырой для дедупликации (recs.md п.2).
    assert "embedding       vector(1024) NOT NULL" in sql
    assert "embedding_raw   vector(1024) NOT NULL" in sql
    # FTS по text_raw генерируемой колонкой (SPEC 4.2 шаг 9).
    assert "to_tsvector('russian'::regconfig, text_raw)" in sql
    # Фильтр публичности опирается на NOT NULL project_id (recs.md п.10).
    assert "project_id      uuid NOT NULL REFERENCES projects(project_id)" in sql
    assert "publicity      text NOT NULL CHECK (publicity IN ('open','anon','closed'))" in sql
    # Версионирование индекса и дедупликация.
    assert "index_version   text NOT NULL REFERENCES index_versions(version)" in sql
    assert "is_canonical    boolean DEFAULT true" in sql


def test_no_ann_index_is_created(sql: str) -> None:
    """ANN-индекс не создаётся сознательно: точный перебор по 20 тыс. векторов."""
    assert "ivfflat" not in sql.lower()
    assert "hnsw" not in sql.lower()


# ------------------------------------------------------------------ раннер


def test_discover_reads_and_orders_migrations() -> None:
    migrations = discover_migrations(MIGRATIONS_DIR)
    assert [migration.number for migration in migrations] == [1, 2]
    assert str(migrations[0]) == "0001_initial"
    assert len(migrations[0].checksum) == 64


def test_discover_rejects_bad_file_name(tmp_path: Path) -> None:
    (tmp_path / "initial.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(MigrationError, match="NNNN_имя.sql"):
        discover_migrations(tmp_path)


def test_discover_rejects_duplicate_numbers(tmp_path: Path) -> None:
    (tmp_path / "0001_a.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "0001_b.sql").write_text("SELECT 2;", encoding="utf-8")
    with pytest.raises(MigrationError, match="дважды"):
        discover_migrations(tmp_path)


def test_missing_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="каталог миграций"):
        discover_migrations(tmp_path / "нет-такого")


def test_pending_skips_applied_and_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "0001_a.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "0002_b.sql").write_text("SELECT 2;", encoding="utf-8")
    migrations = discover_migrations(tmp_path)

    assert [item.number for item in pending_migrations(migrations, {})] == [1, 2]

    applied = {migrations[0].number: migrations[0].checksum}
    assert [item.number for item in pending_migrations(migrations, applied)] == [2]

    everything = {item.number: item.checksum for item in migrations}
    assert pending_migrations(migrations, everything) == []


def test_edited_applied_migration_is_rejected(tmp_path: Path) -> None:
    """Правка применённой миграции разводит схему стенда и git молча."""
    (tmp_path / "0001_a.sql").write_text("SELECT 1;", encoding="utf-8")
    migrations = discover_migrations(tmp_path)

    with pytest.raises(MigrationError, match="изменена после применения"):
        pending_migrations(migrations, {1: "0" * 64})
