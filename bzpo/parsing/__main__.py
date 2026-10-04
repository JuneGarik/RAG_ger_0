"""Конвертация презентаций в md и проверка md против pptx.

    python -m bzpo.parsing convert data/samples/отчёт.pptx [--out data/converted]
    python -m bzpo.parsing check data/samples/отчёт.pptx data/samples/отчёт.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bzpo.parsing.check import COVERAGE_MIN, DeckCheck, check
from bzpo.parsing.pptx import extract, to_markdown


def report(result: DeckCheck) -> None:
    print(f"полнота текста: {result.coverage:.1%} (нужно ≥ {COVERAGE_MIN:.0%}); "
          f"чисел без опоры в XML: {result.unsupported}")
    for problem in result.problems:
        print(f"  ✗ {problem}")
    for slide in sorted(result.slides, key=lambda s: s.coverage)[:5]:
        if slide.coverage < 1.0:
            print(f"  слайд {slide.slide}: полнота {slide.coverage:.0%}")
    for slide in result.slides:
        if slide.unsupported:
            print(f"  слайд {slide.slide}: числа без опоры {', '.join(slide.unsupported[:6])}")
    print("ПРОЙДЕНО" if result.passed else "НЕ ПРОЙДЕНО")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bzpo.parsing")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert", help="pptx -> md (без модели) с проверкой")
    c.add_argument("pptx", type=Path)
    c.add_argument("--out", type=Path, default=Path("data/converted"))
    k = sub.add_parser("check", help="проверить готовый md против pptx")
    k.add_argument("pptx", type=Path)
    k.add_argument("md", type=Path)
    args = parser.parse_args(argv)

    if args.command == "convert":
        deck = extract(args.pptx)
        markdown = to_markdown(deck)
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / f"{args.pptx.stem}.md"
        target.write_text(markdown, encoding="utf-8")
        print(f"{target}: {len(deck.slides)} слайдов, разбор изображений нужен на "
              f"{sum(s.needs_vlm for s in deck.slides)}")
        result = check(args.pptx, markdown)
    else:
        result = check(args.pptx, args.md.read_text(encoding="utf-8"))
    report(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
