"""Индексатор: изоляция ошибок конвертации, проверка перед индексацией, версия индекса."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from bzpo.config import load_config
from bzpo.ingest import indexer
from bzpo.parsing.check import DeckCheck, SlideCheck
from bzpo.parsing.convert import Conversion
from bzpo.parsing.detect import Detection

SLIDE_MD = "<!-- slide: 1 -->\n## [Слайд 1] Заголовок\n**Раздел:** Р\n\nТекст слайда.\n"


class FakeConn:
    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


@dataclass
class Doc:
    project_code: str
    file: Path
    title: str = "Документ"
    doc_type: str = "report"
    doc_date: date | None = None


@dataclass
class Project:
    code: str = "P1"
    publicity: str = "open"
    client: str = "Клиент"
    anon_label: str | None = None


@dataclass
class Report:
    projects: list[Project]
    documents: list[Doc]


@pytest.fixture
def fake_repo(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"indexed": [], "activated": None}
    monkeypatch.setattr(indexer.repo, "ensure_index_version", lambda conn, **kw: None)
    monkeypatch.setattr(indexer.repo, "upsert_document",
                        lambda conn, **kw: (kw["file_key"], (None, None)))
    monkeypatch.setattr(indexer.repo, "mark_indexed", lambda conn, *a: None)
    monkeypatch.setattr(indexer.repo, "document_indexed", lambda conn, doc_id, v: False)
    monkeypatch.setattr(indexer.repo, "replace_chunks",
                        lambda conn, **kw: calls["indexed"].append(kw["doc_id"]))
    monkeypatch.setattr(indexer.repo, "set_document_status", lambda conn, *a, **kw: None)
    monkeypatch.setattr(indexer.repo, "activate_index_version",
                        lambda conn, version: calls.__setitem__("activated", version))
    return calls


def test_bad_file_and_failed_check_do_not_stop_or_enter_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_repo: dict[str, Any]
) -> None:
    files = {name: tmp_path / name for name in ("broken.pptx", "suspicious.pptx", "good.md")}
    for path in files.values():
        path.write_bytes(b"x")

    failed_check = DeckCheck([SlideCheck(1, paragraphs=10, covered=5, unsupported=["52"])])

    def fake_convert(path: Path) -> Conversion:
        if path.name == "broken.pptx":
            raise ValueError("picture contains no image")
        if path.name == "suspicious.pptx":
            return Conversion(path, Detection("pptx", "тест"), SLIDE_MD, check=failed_check)
        return Conversion(path, Detection("md_slides", "тест"), SLIDE_MD)

    monkeypatch.setattr(indexer, "convert", fake_convert)
    report = Report([Project()], [Doc("P1", p) for p in files.values()])
    stats = indexer.index_documents(
        FakeConn(), load_config(use_yaml=False), report, FakeEmbedder(), lambda t: len(t.split()),
        data_root_key=lambda path: path.name,
    )
    assert "picture contains no image" in stats.failed["broken.pptx"]
    assert "проверка конвертации не пройдена" in stats.failed["suspicious.pptx"]
    assert stats.indexed == ["good.md"]
    assert fake_repo["indexed"] == ["good.md"]
    assert fake_repo["activated"] is None  # с ошибками версия не активируется


def test_index_version_depends_on_document_chunking() -> None:
    settings = load_config(use_yaml=False)
    before = indexer.index_version_name(settings)
    settings.chunking.chunk_target = settings.chunking.chunk_target + 100
    assert indexer.index_version_name(settings) != before
    assert indexer.chunk_params(settings)["document_target"] == settings.chunking.chunk_target
