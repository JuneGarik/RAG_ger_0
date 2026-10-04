"""Раннер SQL-миграций.

РЕШЕНИЕ (PLAN 0.5.2, фиксируется и не меняется): собственный раннер поверх
нумерованных .sql-файлов, НЕ Alembic. Обоснование:

* схема пишется руками из SPEC раздела 3 и содержит вещи, которые Alembic
  всё равно не сгенерирует и не сравнит: расширение vector, тип vector(1024),
  GENERATED-колонку tsvector, частичные CHECK-ограничения;
* ORM в проекте нет и не планируется (SQL живёт только в storage/), значит
  автогенерация миграций по моделям — главная ценность Alembic — не работает;
* весь жизненный цикл схемы на PoC — 3–5 миграций вперёд, откат делается
  версией индекса (index_versions), а не откатом DDL.

Цена решения: нет downgrade и нет автогенерации. Обе потери осознанные.

psycopg импортируется лениво внутри функций: модуль обязан импортироваться
на машине без драйвера (тесты границ и разбор файлов миграций от базы
не зависят), а `pip install bzpo[db]` нужен только для реального применения.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

#: 0001_initial.sql -> номер 1, имя initial
_MIGRATION_RE = re.compile(r"^(?P<number>\d{4})_(?P<name>[A-Za-z0-9_\-]+)\.sql$")

SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  number      int PRIMARY KEY,
  name        text NOT NULL,
  checksum    text NOT NULL,
  applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(RuntimeError):
    """Проблема с набором миграций: дубли номеров, изменённый уже применённый файл."""


@dataclass(frozen=True, slots=True)
class Migration:
    """Одна миграция: номер, имя, текст, контрольная сумма."""

    number: int
    name: str
    path: Path
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        return f"{self.number:04d}_{self.name}"


def discover_migrations(directory: Path | str | None = None) -> list[Migration]:
    """Прочитать миграции из каталога, отсортировав по номеру.

    Чистая функция: файлы читаются, база не нужна. Дубликат номера — ошибка,
    а не «применим оба в алфавитном порядке»: порядок DDL в SPEC обязателен.
    """
    base = Path(directory) if directory is not None else DEFAULT_MIGRATIONS_DIR
    if not base.is_dir():
        raise MigrationError(f"каталог миграций не найден: {base}")

    found: dict[int, Migration] = {}
    for path in sorted(base.glob("*.sql")):
        match = _MIGRATION_RE.match(path.name)
        if match is None:
            raise MigrationError(
                f"файл {path.name} не соответствует шаблону NNNN_имя.sql — "
                "порядок применения миграций определяется номером и должен быть однозначным"
            )
        number = int(match.group("number"))
        if number in found:
            raise MigrationError(
                f"номер миграции {number:04d} встречается дважды: "
                f"{found[number].path.name} и {path.name}"
            )
        found[number] = Migration(
            number=number,
            name=match.group("name"),
            path=path,
            sql=path.read_text(encoding="utf-8"),
        )
    return [found[number] for number in sorted(found)]


def applied_migrations(conn: Any) -> dict[int, str]:
    """Уже применённые миграции: номер -> контрольная сумма."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_MIGRATIONS_DDL)
        cur.execute("SELECT number, checksum FROM schema_migrations")
        return {int(row[0]): str(row[1]) for row in cur.fetchall()}


def pending_migrations(
    migrations: list[Migration], applied: dict[int, str]
) -> list[Migration]:
    """Отобрать неприменённые, попутно проверив, что применённые не менялись.

    Правка уже применённого файла — молчаливое расхождение схемы стенда и git,
    поэтому она останавливает раннер. Ручные правки схемы запрещены (SPEC 3).
    """
    pending: list[Migration] = []
    for migration in migrations:
        known = applied.get(migration.number)
        if known is None:
            pending.append(migration)
        elif known != migration.checksum:
            raise MigrationError(
                f"миграция {migration} изменена после применения "
                f"(в базе {known[:12]}, в файле {migration.checksum[:12]}). "
                "Схему правят новой миграцией, а не редактированием старой"
            )
    return pending


def migrate(dsn: str, directory: Path | str | None = None) -> list[Migration]:
    """Применить все неприменённые миграции. Возвращает применённые.

    Идемпотентность: повторный запуск не делает ничего. Транзакция — на
    миграцию: упавшая миграция откатывается целиком, ранее применённые
    остаются, следующие не выполняются.
    """
    import psycopg  # локальный импорт: драйвер нужен только здесь

    migrations = discover_migrations(directory)
    applied_list: list[Migration] = []

    with psycopg.connect(dsn, autocommit=True) as conn:
        already = applied_migrations(conn)
        pending = pending_migrations(migrations, already)
        if not pending:
            logger.info("миграции: нечего применять, схема на версии %s", max(already, default=0))
            return []

        for migration in pending:
            logger.info("миграции: применяю %s", migration)
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(migration.sql)
                    cur.execute(
                        "INSERT INTO schema_migrations (number, name, checksum) "
                        "VALUES (%s, %s, %s)",
                        (migration.number, migration.name, migration.checksum),
                    )
            applied_list.append(migration)

    logger.info("миграции: применено %d", len(applied_list))
    return applied_list


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m bzpo.storage.migrate [--dsn ...] [--dir ...]."""
    import argparse

    from bzpo.config import get_settings

    parser = argparse.ArgumentParser(description="Применение SQL-миграций БЗПО")
    parser.add_argument("--dsn", default=None, help="строка подключения; по умолчанию BZPO_DB__DSN")
    parser.add_argument("--dir", default=None, help="каталог миграций")
    args = parser.parse_args(argv)

    settings = get_settings()
    dsn = args.dsn
    if dsn is None:
        if settings.db.dsn is None:
            parser.error("DSN не задан: укажите --dsn или переменную окружения BZPO_DB__DSN")
        dsn = settings.db.dsn.get_secret_value()

    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
    applied = migrate(dsn, args.dir or settings.migrations_path)
    for migration in applied:
        print(f"применено: {migration}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
