"""Загрузчик реестра: только проверенные строки, проверки SPEC 4.5, отчёт об отклонённых."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

openpyxl = pytest.importorskip("openpyxl")

from bzpo.registry.loader import load_registry  # noqa: E402

PROJECT_HEADER = ["Год", "Код проекта", "Основная отрасль", "Клиент", "Обзелич.название клиента",
                  "Название", "Дата начала", "Дата окончания", "Уровень публичности",
                  "Клиент (по договору)", "Результат тендера", "Статус записи"]
DOC_HEADER = ["Код проекта", "Файл", "Исходный файл", "Тип документа", "Название",
              "Дата документа", "Слайдов", "Комментарии"]


def project(code: str, **overrides: Any) -> list[Any]:
    row = {
        "Год": 2025, "Код проекта": code, "Основная отрасль": "Недвижимость",
        "Клиент": "ГК Ромашка", "Обзелич.название клиента": "крупный девелопер",
        "Название": f"Проект {code}", "Дата начала": date(2025, 3, 31),
        "Дата окончания": date(2025, 7, 18), "Уровень публичности": "open",
        "Клиент (по договору)": "ООО «Ромашка»", "Результат тендера": None,
        "Статус записи": "проверено",
    }
    row.update(overrides)
    return [row[h] for h in PROJECT_HEADER]


def document(code: str, file: str, doc_type: str = "report") -> list[Any]:
    return [code, file, None, doc_type, f"Документ {file}", date(2025, 7, 18), 3, None]


def registry(tmp_path: Path, projects: list[list[Any]], docs: list[list[Any]]) -> Path:
    (tmp_path / "samples").mkdir(exist_ok=True)
    for row in docs:
        (tmp_path / row[1]).write_text("x", encoding="utf-8")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Проекты"
    ws.append(PROJECT_HEADER)
    for row in projects:
        ws.append(row)
    ds = wb.create_sheet("Документы")
    ds.append(DOC_HEADER)
    for row in docs:
        ds.append(row)
    path = tmp_path / "registry.xlsx"
    wb.save(path)
    return path


def test_verified_report_project_loaded_with_implied_won(tmp_path: Path) -> None:
    path = registry(tmp_path, [project("P1")], [document("P1", "samples/a.md")])
    report = load_registry(path, tmp_path)
    assert report.rejected == []
    [loaded] = report.projects
    assert loaded.tender_result == "won"  # есть отчёт -> выигран, даже если ячейка пуста
    assert loaded.client_contract == "ООО «Ромашка»"
    [doc] = report.documents
    assert doc.file == (tmp_path / "samples/a.md").resolve()
    assert doc.doc_type == "report"


def test_drafts_skipped_and_their_documents_not_loaded(tmp_path: Path) -> None:
    path = registry(
        tmp_path,
        [project("P1", **{"Статус записи": "черновик"})],
        [document("P1", "samples/a.md")],
    )
    report = load_registry(path, tmp_path)
    assert report.projects == [] and report.documents == []
    assert report.skipped_drafts == ["P1"]


@pytest.mark.parametrize(
    ("overrides", "doc_type", "problem"),
    [
        ({"Уровень публичности": None}, "report", "Уровень публичности"),
        ({"Уровень публичности": "secret"}, "report", "ожидается"),
        ({"Уровень публичности": "anon", "Обзелич.название клиента": None}, "report", "anon"),
        ({"Результат тендера": "lost"}, "report", "есть отчёт или акт"),
        ({"Результат тендера": None}, "kp", "только с КП"),
        ({"Код проекта": "РОМАШКА-2025-DRAFT"}, "report", "временный код"),
        ({"Дата окончания": date(2025, 1, 1)}, "report", "раньше"),
    ],
)
def test_invalid_project_rejected_with_reason(
    tmp_path: Path, overrides: dict[str, Any], doc_type: str, problem: str
) -> None:
    code = overrides.get("Код проекта", "P1")
    path = registry(tmp_path, [project("P1", **overrides)],
                    [document(code, "samples/a.md", doc_type)])
    report = load_registry(path, tmp_path)
    assert report.projects == []
    [rejected] = report.rejected
    assert any(problem in p for p in rejected["problems"]), rejected


def test_kp_only_project_with_result_loaded(tmp_path: Path) -> None:
    path = registry(tmp_path, [project("P1", **{"Результат тендера": "lost"})],
                    [document("P1", "samples/kp.md", "kp")])
    [loaded] = load_registry(path, tmp_path).projects
    assert loaded.tender_result == "lost"


def test_missing_file_and_bad_type_rejected(tmp_path: Path) -> None:
    path = registry(
        tmp_path, [project("P1")],
        [document("P1", "samples/a.md"), document("P1", "samples/b.md", "отчёт")],
    )
    (tmp_path / "samples/a.md").unlink()
    report = load_registry(path, tmp_path)
    problems = [p for r in report.rejected for p in r["problems"]]
    assert any("файла нет" in p for p in problems)
    assert any("Тип документа" in p for p in problems)
    assert report.documents == []
