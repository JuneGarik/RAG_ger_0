"""CLI реестра.

    python -m bzpo.registry draft data/samples/отчёт.md [--registry data/registry.xlsx]

Черновик строки по новому документу: GigaChat (роль enrichment) заполняет
поля, проверка сверяет их со слайдами, результат дописывается в реестр со
статусом «черновик». Закройте реестр в Excel перед запуском: файл
перезаписывается целиком.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

from bzpo.config import load_config
from bzpo.ingest.document import chunk_document
from bzpo.ingest.slides import chunk_slides, is_slide_markdown
from bzpo.parsing.convert import convert
from bzpo.providers.ollama import OllamaClient
from bzpo.registry.draft import (
    apply_team_table,
    build_prompt,
    parse,
    select_context,
    tender_result_for,
    verify,
)
from bzpo.registry.workbook import (
    DocumentRow,
    RegistryError,
    append_draft,
    ensure_not_registered,
    file_key,
)

# Бюджет считается грубо (3 символа на токен русского текста): окно 16k,
# ответ ~1000 токенов, промпт с полями ~1000 — 9000 на документ с запасом.
CONTEXT_BUDGET = 9000


def _count(text: str) -> int:
    return len(text) // 3


def draft(md_path: Path, registry: Path, data_root: Path) -> int:
    """Черновик строки реестра по файлу любого поддерживаемого формата.

    Формат определяется по содержимому: pptx и docx конвертируются на лету,
    md берётся как есть. Для документа без слайдов «слайдами» для модели
    служат фрагменты разделов.
    """
    conversion = convert(md_path)
    if conversion.markdown is None:
        print(f"{md_path}: {conversion.error}")
        return 2
    text = conversion.markdown
    print(f"формат: {conversion.detection.method}")
    file = file_key(md_path.resolve().relative_to(data_root.resolve()).as_posix())
    try:
        ensure_not_registered(registry, file)
    except RegistryError as exc:
        print(f"пропущено: {exc}")
        return 1
    config = load_config()
    if is_slide_markdown(text):
        deck = chunk_slides(text, _count, hard_limit=config.chunking.chunk_hard_limit_embed)
    else:
        deck = chunk_document(text, _count, target=config.chunking.chunk_target,
                              minimum=config.chunking.chunk_min,
                              hard_limit=config.chunking.chunk_hard_limit_embed)
    context = select_context(deck, _count, CONTEXT_BUDGET)
    print(f"слайды для модели: {sorted(context.slides)}; генерация займёт 1–2 минуты…")

    started = time.perf_counter()
    with OllamaClient(config.ollama, role="enrichment") as client:
        result = client.generate(build_prompt(context), json_format=True, num_predict=2500)
    fields = apply_team_table(verify(parse(result.text), context), context)
    print(f"готово за {time.perf_counter() - started:.0f} с")

    doc_type = fields["Тип документа"].value
    source = deck.meta.get("source_file")
    source_path = md_path.parent / source if source else None
    report_date = deck.meta.get("report_date")
    document = DocumentRow(
        file=file,
        source_file=(
            source_path.resolve().relative_to(data_root.resolve()).as_posix()
            if source_path and source_path.exists() else None
        ),
        doc_type=str(doc_type) if doc_type else None,
        title=str(deck.meta.get("doc_title") or fields["Название"].value or md_path.stem),
        doc_date=date.fromisoformat(str(report_date)) if report_date else None,
        slides=len(deck.chunks),
    )
    try:
        row, code = append_draft(
            registry, fields, document,
            tender_result=tender_result_for([document.doc_type or ""]),
            model=config.ollama.model_enrichment.split("/")[-1],
        )
    except RegistryError as exc:
        print(f"не записано: {exc}")
        return 1

    unverified = [f for f in fields.values() if f.value is not None and not f.verified]
    empty = [f.name for f in fields.values() if f.value is None]
    print(f"\nзаписано: {registry}, лист «Проекты», строка {row}, код {code}")
    print(f"жёлтых (проверить): {sum(f.value is not None for f in fields.values())}, "
          f"оранжевых (не подтверждено): {len(unverified)}, пустых: {len(empty)}")
    for f in unverified:
        print(f"  ⚠ {f.name}: {'; '.join(f.problems)}")
    print("Проверьте строку и поставьте в «Статус записи» значение «проверено».")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bzpo.registry")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draft", help="черновик строки реестра по md-документу")
    d.add_argument("md", type=Path)
    d.add_argument("--registry", type=Path, default=Path("data/registry.xlsx"))
    d.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    return draft(args.md, args.registry, args.data_root)


if __name__ == "__main__":
    sys.exit(main())
