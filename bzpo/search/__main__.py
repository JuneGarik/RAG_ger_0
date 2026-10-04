"""Поиск из командной строки.

    python -m bzpo.search "какие барьеры цифровой трансформации у девелопера" [-k 5]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from bzpo.config import load_config
from bzpo.providers.local import LocalEmbedder, LocalReranker
from bzpo.search.query import search
from bzpo.storage import repo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bzpo.search")
    parser.add_argument("query", nargs="+")
    parser.add_argument("-k", type=int, default=None, help="сколько фрагментов показать")
    parser.add_argument("--full", action="store_true", help="печатать текст фрагментов целиком")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    settings = load_config()
    if settings.db.dsn is None:
        print("не задан BZPO_DB__DSN (например, postgresql://localhost/bzpo)")
        return 2
    query = " ".join(args.query)
    embedder, reranker = LocalEmbedder(settings.local), LocalReranker(settings.local)

    started = time.perf_counter()
    with repo.connect(settings.db.dsn.get_secret_value(), settings.db.statement_timeout_ms) as conn:
        result = search(conn, settings, embedder, reranker, query, top_k=args.k)
    elapsed = time.perf_counter() - started

    print(f"запрос: {query}\nлеммы FTS: {', '.join(result.terms) or '—'}")
    if result.not_found:
        print("ничего не найдено")
        return 0
    for n, hit in enumerate(result.hits, 1):
        c = hit.chunk
        lists = ", ".join(f"{name.split(':')[0]}#{rank}" for name, rank in hit.lists.items())
        print(f"\n[{n}] {c.doc_title} — слайд {c.page} ({hit.customer})")
        print(f"    раздел: {c.section_title}")
        print(f"    реранкер {hit.rerank_score:.3f} | итог {hit.score:.3f} "
              f"| буст {'+'.join(hit.boost) or '—'} | списки: {lists}")
        text = c.text_raw if args.full else c.text_raw.split("\n", 1)[0][:200]
        print(f"    {text}")
    print(f"\n{elapsed:.1f} с (включая загрузку моделей)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
