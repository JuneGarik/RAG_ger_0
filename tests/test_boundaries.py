"""Тесты границ модулей (SPEC 10а, PLAN 0.5.6).

Правила: обращения к базе — только из `bzpo/storage/`, к сети (внешний сервис,
Ollama) — только из `bzpo/providers/`. Проверка идёт по разбору AST, а не
регулярным выражением по тексту: `# import psycopg` в комментарии не должен
валить сборку, а `from psycopg import sql` внутри функции — должен.

Каждый сканер проверяется дважды: на реальных исходниках (нарушений нет) и на
специально подложенном нарушении во временном каталоге — иначе тест, который
всегда зелёный, неотличим от сломанного.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "bzpo"

#: Модуль (или его префикс) -> каталоги, где он разрешён.
IMPORT_RULES: dict[str, tuple[str, ...]] = {
    "psycopg": ("bzpo/storage",),
    "psycopg2": ("bzpo/storage",),
    "asyncpg": ("bzpo/storage",),
    "sqlalchemy": ("bzpo/storage",),
    "httpx": ("bzpo/providers",),
    "requests": ("bzpo/providers",),
    "aiohttp": ("bzpo/providers",),
    "urllib.request": ("bzpo/providers",),
    "http.client": ("bzpo/providers",),
    "socket": ("bzpo/providers",),
}

#: Литералы URL допустимы там, где ходят в сеть, и в дефолтах конфигурации.
URL_ALLOWED = ("bzpo/providers", "bzpo/config.py")
URL_PATTERN = re.compile(r"https?://")


@dataclass(frozen=True)
class Violation:
    """Нарушение границы: файл, что именно и на какой строке."""

    path: str
    what: str
    line: int

    def __str__(self) -> str:
        return f"{self.path}:{self.line} — {self.what}"


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in (root / PACKAGE).rglob("*.py"))


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _allowed(relative_path: str, allowed: tuple[str, ...]) -> bool:
    return any(
        relative_path == prefix or relative_path.startswith(prefix + "/") for prefix in allowed
    )


def _imported_modules(tree: ast.AST) -> list[tuple[str, int]]:
    """Все импортируемые модули с номерами строк, включая импорты внутри функций."""
    modules: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.append((node.module, node.lineno))
    return modules


def scan_imports(root: Path) -> list[Violation]:
    """Найти импорты запрещённых модулей вне разрешённых каталогов."""
    violations: list[Violation] = []
    for path in _python_files(root):
        relative_path = _relative(path, root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module, line in _imported_modules(tree):
            for forbidden, allowed in IMPORT_RULES.items():
                if module == forbidden or module.startswith(forbidden + "."):
                    if not _allowed(relative_path, allowed):
                        violations.append(
                            Violation(
                                relative_path,
                                f"импорт «{module}» разрешён только в {', '.join(allowed)}",
                                line,
                            )
                        )
    return violations


def scan_url_literals(root: Path) -> list[Violation]:
    """Найти зашитые URL вне модуля изоляции и конфигурации."""
    violations: list[Violation] = []
    for path in _python_files(root):
        relative_path = _relative(path, root)
        if _allowed(relative_path, URL_ALLOWED):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if URL_PATTERN.search(line):
                violations.append(
                    Violation(relative_path, "URL вне providers/ и конфигурации", number)
                )
    return violations


# ------------------------------------------------------ проверка исходников


def test_database_and_network_imports_stay_in_their_modules() -> None:
    violations = scan_imports(PROJECT_ROOT)
    assert not violations, "нарушены границы модулей:\n" + "\n".join(map(str, violations))


def test_no_hardcoded_urls_outside_providers() -> None:
    violations = scan_url_literals(PROJECT_ROOT)
    assert not violations, "URL вне модуля изоляции:\n" + "\n".join(map(str, violations))


def test_providers_do_use_httpx() -> None:
    """Страховка от обратной ошибки: если сканер перестанет видеть импорты,
    первый тест станет зелёным по недоразумению."""
    modules = [
        module
        for path in _python_files(PROJECT_ROOT)
        for module, _ in _imported_modules(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert "httpx" in modules


def test_storage_imports_psycopg_lazily() -> None:
    """psycopg не установлен в окружении разработки — модуль обязан импортироваться."""
    import importlib

    module = importlib.import_module("bzpo.storage.migrate")
    assert hasattr(module, "migrate")


# ------------------------------------- проверка самого сканера на нарушении


def _plant(tmp_path: Path, relative_path: str, source: str) -> Path:
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    return target


def test_scanner_catches_planted_import_violation(tmp_path: Path) -> None:
    _plant(tmp_path, "bzpo/search/rerank_client.py", "import httpx\n\nX = httpx\n")
    _plant(
        tmp_path,
        "bzpo/generation/context.py",
        "def load():\n    from psycopg import connect\n    return connect\n",
    )
    _plant(tmp_path, "bzpo/providers/ok.py", "import httpx\n")
    _plant(tmp_path, "bzpo/storage/ok.py", "def go():\n    import psycopg\n    return psycopg\n")

    violations = scan_imports(tmp_path)
    offenders = {violation.path for violation in violations}

    assert offenders == {"bzpo/search/rerank_client.py", "bzpo/generation/context.py"}
    assert any("psycopg" in violation.what for violation in violations)


def test_scanner_catches_planted_url_literal(tmp_path: Path) -> None:
    _plant(tmp_path, "bzpo/eval/run.py", 'URL = "http://embeddings.internal/v1/embeddings"\n')
    _plant(tmp_path, "bzpo/providers/ok.py", 'URL = "http://embeddings.internal"\n')
    _plant(tmp_path, "bzpo/config.py", 'BASE = "http://127.0.0.1:8080"\n')

    violations = scan_url_literals(tmp_path)
    assert [violation.path for violation in violations] == ["bzpo/eval/run.py"]


def test_comments_do_not_count_as_imports(tmp_path: Path) -> None:
    _plant(tmp_path, "bzpo/search/notes.py", "# import psycopg — так делать нельзя\nX = 1\n")
    assert scan_imports(tmp_path) == []
