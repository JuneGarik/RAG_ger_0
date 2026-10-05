"""Диагностика реестра и запись решений человека (форма проверки в интерфейсе)."""

from __future__ import annotations

from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from bzpo.registry.loader import diagnose  # noqa: E402
from bzpo.registry.workbook import (  # noqa: E402
    RegistryError,
    project_row,
    review_project,
)

PROJECT_HEADER = ["Код проекта", "Клиент", "Обзелич.название клиента", "Название",
                  "Уровень публичности", "Результат тендера", "Статус записи"]
DOC_HEADER = ["Код проекта", "Файл", "Тип документа", "Название"]


def registry(tmp_path: Path) -> Path:
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "kp.md").write_text("x", encoding="utf-8")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Проекты"
    ws.append(PROJECT_HEADER)
    ws.append(["ИТК-2024-DRAFT", "ООО «ИТК»", "ИТ-заказчик", "Выбор платформы", None,
               None, "черновик"])
    ds = wb.create_sheet("Документы")
    ds.append(DOC_HEADER)
    ds.append(["ИТК-2024-DRAFT", "samples/kp.md", "kp", "КП"])
    path = tmp_path / "registry.xlsx"
    wb.save(path)
    return path


def test_diagnose_lists_what_blocks_indexing(tmp_path: Path) -> None:
    problems = diagnose(registry(tmp_path), tmp_path)["ИТК-2024-DRAFT"]
    joined = " | ".join(problems)
    assert "Уровень публичности" in joined
    assert "временный код" in joined
    assert "Результат тендера" in joined
    assert "проверено" in joined


def test_review_renames_code_on_both_sheets_and_clears_problems(tmp_path: Path) -> None:
    path = registry(tmp_path)
    review_project(path, "ИТК-2024-DRAFT", new_code="ИТК-2024-01", publicity="anon",
                   anon_label=None, doc_type="kp", tender_result="lost", verified=True)
    row = project_row(path, "ИТК-2024-01")
    assert row["Уровень публичности"] == "anon"
    assert row["Обзелич.название клиента"] == "ИТ-заказчик"  # пустое значение не стирает
    assert row["_documents"][0]["Код проекта"] == "ИТК-2024-01"
    assert diagnose(path, tmp_path) == {"ИТК-2024-01": []}


def test_review_unknown_project_refused(tmp_path: Path) -> None:
    path = registry(tmp_path)
    with pytest.raises(RegistryError, match="нет в реестре"):
        review_project(path, "НЕТ-ТАКОГО", new_code="X", publicity="open", anon_label=None,
                       doc_type=None, tender_result=None, verified=True)
