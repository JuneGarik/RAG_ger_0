"""Генерация из командной строки.

    python -m bzpo.generation approach "Диагностика цифровой зрелости" \
        --area "жилой девелопмент" --situation "нет стратегии ЦТ" [--quick] [--type report]

Результат печатается и сохраняется в data/drafts/<время>_approach.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bzpo.config import load_config
from bzpo.generation.approach import ApproachRequest
from bzpo.generation.run import generate_approach, save_draft
from bzpo.providers.local import LocalEmbedder, LocalReranker
from bzpo.storage import repo


def approach(args: argparse.Namespace) -> int:
    settings = load_config()
    if settings.db.dsn is None:
        print("не задан BZPO_DB__DSN (например, postgresql://localhost/bzpo)")
        return 2
    request = ApproachRequest(args.goal, args.area, args.situation)
    embedder, reranker = LocalEmbedder(settings.local), LocalReranker(settings.local)
    scope = repo.SearchScope(doc_types=args.type or None)
    with repo.connect(settings.db.dsn.get_secret_value(), settings.db.statement_timeout_ms) as conn:
        run = generate_approach(
            conn, settings, embedder, reranker, request,
            mode="quick" if args.quick else "thorough", scope=scope,
            progress=lambda message: print(f"  {message}"),
        )
    print("\n" + run.markdown)
    print(f"\nпоиск {run.search_seconds:.0f} с, генерация {run.generation_seconds:.0f} с, "
          f"{run.completion_tokens} ток."
          + (" — ОБРЕЗАНО по лимиту" if run.truncated else ""))
    if run.section is None:
        return 1
    print(f"сохранено: {save_draft(run, Path(args.out))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bzpo.generation")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("approach", help="описание подхода к новому проекту")
    a.add_argument("goal", help="цель нового проекта")
    a.add_argument("--area", help="область / отрасль")
    a.add_argument("--situation", help="ситуация заказчика, ограничения")
    a.add_argument("--quick", action="store_true", help="быстрый режим: меньше контекст и ответ")
    a.add_argument("--type", action="append", choices=["report", "kp", "act", "presentation"],
                   help="искать только в документах этого типа (можно несколько)")
    a.add_argument("--out", default="data/drafts")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    return approach(args)


if __name__ == "__main__":
    sys.exit(main())
