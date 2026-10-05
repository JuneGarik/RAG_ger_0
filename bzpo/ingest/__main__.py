"""Загрузка реестра и индексация документов.

    python -m bzpo.ingest [--registry data/registry.xlsx] [--data-root data]

Шаги: проверенные строки реестра -> clients/projects (+ отчёт загрузки) ->
документы -> чанки с векторами в новой версии индекса -> версия активируется,
если ни один документ не упал. DSN — из BZPO_DB__DSN.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bzpo.config import load_config
from bzpo.ingest.indexer import index_documents
from bzpo.providers.local import LocalEmbedder
from bzpo.registry.loader import load_registry
from bzpo.registry.workbook import file_key
from bzpo.storage import repo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bzpo.ingest")
    parser.add_argument("--registry", type=Path, default=Path("data/registry.xlsx"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--file", action="append", default=None,
                        help="индексировать только этот документ (путь от data/, например "
                             "samples/отчёт.md); можно несколько раз")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = load_config()
    if settings.db.dsn is None:
        print("не задан BZPO_DB__DSN (например, postgresql://localhost/bzpo)")
        return 2
    if settings.external.provider != "local":
        print("индексация из CLI пока только с external.provider: local")
        return 2

    report = load_registry(args.registry, args.data_root)
    print(f"реестр: проектов {len(report.projects)}, документов {len(report.documents)}, "
          f"черновиков пропущено {len(report.skipped_drafts)}, отклонено {len(report.rejected)}")
    for item in report.rejected:
        print(f"  ✗ {item['sheet']}, строка {item['row']} ({item['code']}): "
              f"{'; '.join(item['problems'])}")

    root = args.data_root.resolve()
    registered = [file_key(d.file.resolve().relative_to(root).as_posix())
                  for d in report.documents]
    if args.file:
        # Реестр (проекты, публичность) загружается целиком, индексируются выбранные.
        wanted = {file_key(f) for f in args.file}
        chosen = [d for d in report.documents
                  if file_key(d.file.resolve().relative_to(root).as_posix()) in wanted]
        missing = wanted - {file_key(d.file.resolve().relative_to(root).as_posix())
                            for d in chosen}
        for key in sorted(missing):
            print(f"  ✗ {key}: нет среди проверенных документов реестра")
        report.documents = chosen
        print(f"выбрано для индексации: {len(chosen)}")
    embedder = LocalEmbedder(settings.local)
    with repo.connect(settings.db.dsn.get_secret_value(), settings.db.statement_timeout_ms) as conn:
        upload_id = repo.save_registry(conn, report)
        print(f"загрузка реестра {upload_id} записана")
        stats = index_documents(
            conn, settings, report, embedder, embedder.count_tokens,
            data_root_key=lambda path: file_key(path.resolve().relative_to(root).as_posix()),
        )
        # Полный прогон приводит поиск в соответствие с реестром: документ, убранный
        # из реестра (или вернувшийся в черновик), из поиска исчезает. Выборочный
        # прогон (--file) других документов не трогает.
        removed = [] if args.file else repo.remove_documents_except(conn, registered)
        conn.commit()

    print(f"\nверсия индекса: {stats.version}")
    print(f"проиндексировано: {len(stats.indexed)} ({stats.chunks} чанков), "
          f"без изменений: {len(stats.skipped)}, ошибок: {len(stats.failed)}, "
          f"{stats.seconds:.0f} с")
    for key in removed:
        print(f"  − {key}: нет в реестре — убран из поиска")
    for key, error in stats.failed.items():
        print(f"  ✗ {key}: {error}")
    if stats.failed:
        print("версия не активирована: исправьте ошибки и запустите снова")
    return 1 if stats.failed else 0


if __name__ == "__main__":
    sys.exit(main())
