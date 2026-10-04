"""Загрузчик реестра Excel (SPEC 4.5): проверенные строки -> проекты и документы.

Берутся только строки со статусом «проверено»: черновики модели в систему не
попадают. Каждая отклонённая строка попадает в отчёт с причиной — отчёт
пишется в `registry_uploads.report`, молча строки не теряются.

Проверки (SPEC 4.5): обязательные поля; справочные значения по закрытым
спискам; ссылки на файлы существуют; документ ссылается на загруженный
проект; результат тендера согласован с типами документов (есть отчёт или
акт — только `won`).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from bzpo.registry.draft import DOC_TYPES, tender_result_for
from bzpo.registry.workbook import DOCUMENTS, PROJECTS, STATUS_COLUMN, STATUS_VERIFIED

PUBLICITY = ("open", "anon", "closed")
TENDER_RESULTS = ("won", "lost", "not_submitted")


@dataclass(frozen=True, slots=True)
class RegistryProject:
    code: str
    name: str
    client: str
    client_contract: str | None
    industry: str | None
    publicity: str
    anon_label: str | None
    tender_result: str | None
    date_start: date | None
    date_end: date | None
    contract_no: str | None


@dataclass(frozen=True, slots=True)
class RegistryDocument:
    project_code: str
    file: Path  # абсолютный путь
    title: str
    doc_type: str
    doc_date: date | None


@dataclass(slots=True)
class LoadReport:
    source_file: str
    projects: list[RegistryProject] = field(default_factory=list)
    documents: list[RegistryDocument] = field(default_factory=list)
    skipped_drafts: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.projects) + len(self.documents) + len(self.rejected)

    def as_json(self) -> dict[str, Any]:
        return {
            "projects": [p.code for p in self.projects],
            "documents": [str(d.file) for d in self.documents],
            "skipped_drafts": self.skipped_drafts,
            "rejected": self.rejected,
        }


def load_registry(path: Path, data_root: Path) -> LoadReport:
    from openpyxl import load_workbook

    workbook = load_workbook(path, data_only=True, read_only=True)
    report = LoadReport(source_file=str(path))
    project_rows = _rows(workbook[PROJECTS])
    document_rows = _rows(workbook[DOCUMENTS])

    docs_by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_no, row in document_rows:
        docs_by_code[_text(row.get("Код проекта")) or ""].append({"_row": row_no, **row})

    for row_no, row in project_rows:
        code = _text(row.get("Код проекта"))
        if _text(row.get(STATUS_COLUMN)) != STATUS_VERIFIED:
            report.skipped_drafts.append(code or f"строка {row_no}")
            continue
        problems, project = _project(row, docs_by_code.get(code or "", []))
        if problems:
            report.rejected.append({"sheet": PROJECTS, "row": row_no, "code": code,
                                    "problems": problems})
        else:
            report.projects.append(project)

    loaded = {p.code for p in report.projects}
    for code, rows in docs_by_code.items():
        if code not in loaded:
            continue
        for row in rows:
            problems, document = _document(row, code, data_root)
            if problems:
                report.rejected.append({"sheet": DOCUMENTS, "row": row["_row"], "code": code,
                                        "problems": problems})
            else:
                report.documents.append(document)
    return report


# --------------------------------------------------------------- детали


def _project(
    row: dict[str, Any], docs: list[dict[str, Any]]
) -> tuple[list[str], RegistryProject | None]:
    problems: list[str] = []
    code = _text(row.get("Код проекта"))
    name = _text(row.get("Название"))
    client = _text(row.get("Клиент"))
    publicity = _text(row.get("Уровень публичности"))
    anon_label = _text(row.get("Обзелич.название клиента"))
    tender = _text(row.get("Результат тендера"))

    for label, value in (("Код проекта", code), ("Название", name), ("Клиент", client),
                         ("Уровень публичности", publicity)):
        if not value:
            problems.append(f"не заполнено «{label}»")
    if code and code.endswith("-DRAFT"):
        problems.append("временный код черновика: замените на код из учёта")
    if publicity and publicity not in PUBLICITY:
        problems.append(f"«Уровень публичности» = {publicity!r}, ожидается {PUBLICITY}")
    if publicity == "anon" and not anon_label:
        problems.append("anon без «Обзелич.название клиента»")
    if tender and tender not in TENDER_RESULTS:
        problems.append(f"«Результат тендера» = {tender!r}, ожидается {TENDER_RESULTS}")

    doc_types = [_text(d.get("Тип документа")) or "" for d in docs]
    implied = tender_result_for(doc_types)
    if implied and tender and tender != implied:
        problems.append(f"в проекте есть отчёт или акт, а результат тендера {tender!r}")
    if not docs:
        problems.append("нет ни одного документа на листе «Документы»")
    if not implied and docs and not tender:
        problems.append("проект только с КП: укажите «Результат тендера»")

    start, end = _date(row.get("Дата начала")), _date(row.get("Дата окончания"))
    if start and end and end < start:
        problems.append("«Дата окончания» раньше «Дата начала»")
    if problems:
        return problems, None
    return [], RegistryProject(
        code=code or "", name=name or "", client=client or "",
        client_contract=_text(row.get("Клиент (по договору)")),
        industry=_text(row.get("Основная отрасль")) or _text(row.get("Отрасль")),
        publicity=publicity or "", anon_label=anon_label,
        tender_result=tender or implied, date_start=start, date_end=end,
        contract_no=None,
    )


def _document(
    row: dict[str, Any], code: str, data_root: Path
) -> tuple[list[str], RegistryDocument | None]:
    problems: list[str] = []
    file = _text(row.get("Файл"))
    doc_type = _text(row.get("Тип документа"))
    title = _text(row.get("Название"))
    path = (data_root / file).resolve() if file else None
    if not file:
        problems.append("не заполнено «Файл»")
    elif not path or not path.is_file():
        problems.append(f"файла нет: {data_root / file}")
    if doc_type not in DOC_TYPES:
        problems.append(f"«Тип документа» = {doc_type!r}, ожидается {DOC_TYPES}")
    if not title:
        problems.append("не заполнено «Название» (оно идёт в список «Источники»)")
    if problems or path is None:
        return problems, None
    return [], RegistryDocument(project_code=code, file=path, title=title or "",
                                doc_type=doc_type or "", doc_date=_date(row.get("Дата документа")))


def _rows(sheet: Any) -> list[tuple[int, dict[str, Any]]]:
    rows = sheet.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    result = []
    for number, values in enumerate(rows, start=2):
        if all(v in (None, "") for v in values):
            continue
        result.append((number, dict(zip(header, values, strict=False))))
    return result


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        for parse in (date.fromisoformat, lambda s: datetime.strptime(s, "%d.%m.%Y").date()):
            try:
                return parse(value.strip())
            except ValueError:
                continue
    return None
