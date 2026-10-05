"""Что изменилось с последней индексации и что с этим сделать.

Индекс устаревает тремя путями: правится md (вручную или повторной
конвертацией), правится строка реестра (заказчик, публичность, название попадают
в шапки фрагментов и фильтры), документ убирают из реестра. Плюс оригинал в
data/sources может оказаться новее своего md — тогда нужна повторная конвертация.

Сравнение идёт по отпечаткам, а не по датам: сохранение файла без правок
переиндексацию не вызывает. Решает код; интерфейс показывает список и одну
кнопку «Обработать изменения».
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Что сделать с документом. Порядок — по важности для пользователя.
NEW = "new"  # проверен в реестре, ещё не в поиске
MD_CHANGED = "md_changed"  # md изменён после индексации
REGISTRY_CHANGED = "registry_changed"  # строка реестра изменилась после индексации
REMOVED = "removed"  # в поиске, но в реестре больше нет
SOURCE_NEWER = "source_newer"  # оригинал новее md — переконвертировать

MESSAGES = {
    NEW: "проверен в реестре, ещё не в поиске — проиндексировать",
    MD_CHANGED: "md изменён после индексации — переиндексировать",
    REGISTRY_CHANGED: "строка реестра изменилась — переиндексировать",
    REMOVED: "нет в реестре — убрать из поиска",
    SOURCE_NEWER: "оригинал новее md — сконвертировать заново на вкладке «Конвертация»",
}
# Эти изменения снимает индексация; SOURCE_NEWER — только повторная конвертация.
FIXED_BY_INDEXING = (NEW, MD_CHANGED, REGISTRY_CHANGED, REMOVED)


@dataclass(frozen=True, slots=True)
class Change:
    key: str  # путь от data/, NFC
    kind: str

    @property
    def message(self) -> str:
        return MESSAGES[self.kind]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry_fingerprint(project: Any, document: Any) -> str:
    """Отпечаток строк реестра, от которых зависят фрагменты документа и фильтры."""
    payload = {
        "project": asdict(project),
        "document": {k: v for k, v in asdict(document).items() if k != "file"},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str,
                                     ensure_ascii=False).encode()).hexdigest()


def detect(
    report: Any, indexed: dict[str, Any], data_root: Path, *, sources: Path | None = None
) -> list[Change]:
    """Сравнить реестр и файлы с тем, что в базе.

    indexed — документы из базы по ключу (путь от data/): объекты с полями
    file_hash, registry_hash, chunks.
    """
    root = data_root.resolve()
    projects = {p.code: p for p in report.projects}
    changes: list[Change] = []
    keys = set()
    for document in report.documents:
        key = _key(document.file.resolve().relative_to(root).as_posix())
        keys.add(key)
        stored = indexed.get(key)
        if stored is None or not stored.chunks:
            changes.append(Change(key, NEW))
        elif not document.file.exists():
            continue  # отсутствующий файл покажет диагностика реестра
        elif stored.file_hash != file_hash(document.file):
            changes.append(Change(key, MD_CHANGED))
        elif stored.registry_hash != registry_fingerprint(projects[document.project_code],
                                                          document):
            changes.append(Change(key, REGISTRY_CHANGED))
    changes += [Change(key, REMOVED) for key, stored in indexed.items()
                if key not in keys and stored.chunks]
    if sources is not None:
        changes += [Change(key, SOURCE_NEWER) for key in _sources_newer(data_root, sources)]
    return changes


_SOURCE = re.compile(r'^source_file:\s*"?(.+?)"?\s*$', re.M)


def _sources_newer(data_root: Path, sources: Path) -> list[str]:
    """md, чей оригинал в data/sources изменён позже самого md."""
    result = []
    for md in sorted((data_root / "samples").glob("*.md")):
        head = md.read_text(encoding="utf-8", errors="replace")[:2000]
        match = _SOURCE.search(head)
        if not match:
            continue
        original = sources / match.group(1)
        if original.exists() and original.stat().st_mtime > md.stat().st_mtime + 1:
            result.append(_key(md.resolve().relative_to(data_root.resolve()).as_posix()))
    return result


def _key(path: str) -> str:
    return unicodedata.normalize("NFC", path)
