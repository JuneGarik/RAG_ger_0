"""Конвертация в md с автоопределением формата и проверка md против pptx.

    python -m bzpo.parsing convert data/samples/файл.pptx|.docx|.md [--out data/converted]
    python -m bzpo.parsing convert data/samples              # все исходники папки
    python -m bzpo.parsing convert отчёт.pptx --vlm          # + разбор изображений моделью
    python -m bzpo.parsing check data/samples/отчёт.pptx data/samples/отчёт.md

Формат определяется по содержимому файла (bzpo/parsing/detect.py), способ
разбора выбирается автоматически.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bzpo.parsing.check import COVERAGE_MIN, DeckCheck, check
from bzpo.parsing.convert import convert, sources


def report(result: DeckCheck) -> None:
    print(f"  полнота текста: {result.coverage:.1%} (нужно ≥ {COVERAGE_MIN:.0%}); "
          f"чисел без опоры в XML: {result.unsupported}")
    for problem in result.problems:
        print(f"  ✗ {problem}")
    for slide in result.slides:
        if slide.unsupported:
            print(f"  слайд {slide.slide}: числа без опоры {', '.join(slide.unsupported[:6])}")
    print("  проверка: " + ("ПРОЙДЕНА" if result.passed else "НЕ ПРОЙДЕНА"))


def convert_one(path: Path, out: Path, vlm: bool = False) -> bool:
    result = convert(
        path, vlm=vlm,
        progress=lambda done, total, slide: print(f"  модель: {done}/{total} (слайд {slide})",
                                                   flush=True),
    )
    print(f"{path.name}: {result.detection.method}")
    if result.markdown is None:
        print(f"  ✗ {result.error}")
        return False
    if result.detection.format in ("pptx", "docx"):
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"{path.stem}.md"
        target.write_text(result.markdown, encoding="utf-8")
        print(f"  -> {target} ({result.notes}; разбор изображений нужен: {result.needs_vlm})")
    else:
        print(f"  уже md: {result.notes}")
    for issue in result.vlm_issues:
        print(f"  ⚠ {issue}")
    if result.check is not None:
        report(result.check)
    return result.ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bzpo.parsing")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("convert", help="файл или папка -> md, формат определяется сам")
    c.add_argument("path", type=Path)
    c.add_argument("--out", type=Path, default=Path("data/converted"))
    c.add_argument("--vlm", action="store_true",
                   help="с разбором изображений моделью (pptx; ~20 с на слайд, нужен LibreOffice)")
    k = sub.add_parser("check", help="проверить готовый md против pptx")
    k.add_argument("pptx", type=Path)
    k.add_argument("md", type=Path)
    args = parser.parse_args(argv)

    if args.command == "check":
        result = check(args.pptx, args.md.read_text(encoding="utf-8"))
        report(result)
        return 0 if result.passed else 1
    paths = sources(args.path) if args.path.is_dir() else [args.path]
    results = [convert_one(path, args.out, args.vlm) for path in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
