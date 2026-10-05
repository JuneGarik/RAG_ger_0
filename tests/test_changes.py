from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from bzpo.ingest import changes
from bzpo.registry.loader import LoadReport, RegistryDocument, RegistryProject


@dataclass
class Stored:
    file_hash: str | None
    registry_hash: str | None
    chunks: int


def setup(tmp_path: Path) -> tuple[LoadReport, RegistryProject, RegistryDocument]:
    (tmp_path / "samples").mkdir()
    md = tmp_path / "samples" / "отчёт.md"
    md.write_text('---\nsource_file: "отчёт.pptx"\n---\n# Слайд', encoding="utf-8")
    project = RegistryProject("P-1", "Проект", "Заказчик", None, None, "open", None, "won",
                              None, None, None)
    document = RegistryDocument("P-1", md, "Отчёт", "report", None)
    return LoadReport("r.xlsx", [project], [document]), project, document


def test_states(tmp_path: Path) -> None:
    report, project, document = setup(tmp_path)
    key = "samples/отчёт.md"
    actual = Stored(changes.file_hash(document.file),
                    changes.registry_fingerprint(project, document), 5)

    assert changes.detect(report, {}, tmp_path) == [changes.Change(key, changes.NEW)]
    assert changes.detect(report, {key: actual}, tmp_path) == []

    stale_md = Stored("другой", actual.registry_hash, 5)
    assert changes.detect(report, {key: stale_md}, tmp_path)[0].kind == changes.MD_CHANGED

    stale_registry = Stored(actual.file_hash, "прежний", 5)
    assert changes.detect(report, {key: stale_registry}, tmp_path)[0].kind == \
        changes.REGISTRY_CHANGED

    gone = changes.detect(report, {key: actual, "samples/старый.md": actual}, tmp_path)
    assert gone == [changes.Change("samples/старый.md", changes.REMOVED)]


def test_registry_change_changes_fingerprint(tmp_path: Path) -> None:
    _, project, document = setup(tmp_path)
    before = changes.registry_fingerprint(project, document)
    renamed = RegistryProject(**{**{f: getattr(project, f) for f in project.__slots__},
                                 "client": "Новый заказчик"})
    assert changes.registry_fingerprint(renamed, document) != before


def test_source_newer_than_md(tmp_path: Path) -> None:
    report, _, document = setup(tmp_path)
    sources = tmp_path / "sources"
    sources.mkdir()
    original = sources / "отчёт.pptx"
    original.write_bytes(b"pptx")
    md_time = document.file.stat().st_mtime
    os.utime(original, (md_time + 60, md_time + 60))
    found = changes.detect(report, {}, tmp_path, sources=sources)
    assert changes.Change("samples/отчёт.md", changes.SOURCE_NEWER) in found
