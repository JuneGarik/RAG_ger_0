"""A/B: точность чтения широкой таблицы Qwen — как есть и после линеаризации.

Слайд 34 отчёта ДЕВ (тепловая карта 8 критериев × 12 функций). Модель отвечает
на вопросы о конкретных ячейках, ответы сверяются с таблицей.
Запуск: python scripts/wide_table_ab.py [число_повторов]
"""

import json
import os
import random
import re
import sys
from pathlib import Path

from bzpo.config import load_config
from bzpo.generation.tables import linearize_wide_tables
from bzpo.ingest.slides import chunk_slides
from bzpo.providers.ollama import OllamaClient

ROOT = Path(__file__).resolve().parent.parent
# md отчёта с тепловой картой на слайде 34; имя файла заказчика в репозиторий не пишется.
SAMPLE = Path(os.environ.get("BZPO_AB_SAMPLE", ROOT / "data/samples/report_dev.md"))
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
deck = chunk_slides(
    SAMPLE.read_text(encoding="utf-8"),
    lambda s: len(s) // 4,
    hard_limit=8192,
)
slide = deck.chunks[33].text_raw

rows = [
    [c.strip() for c in line.strip().strip("|").split("|")]
    for line in slide.split("\n")
    if line.startswith("|") and not line.startswith("|---")
]
header, body = rows[0], rows[1:]
cells = [(r[0].strip("*"), header[j], r[j]) for r in body for j in range(1, len(header))]

ollama = OllamaClient(load_config().ollama, role="query")
variants = {"таблица как есть": slide, "линеаризация": linearize_wide_tables(slide)}
score = {name: [0, 0] for name in variants}
for repeat in range(REPEATS):
    asked = random.Random(repeat).sample(cells, 12)
    questions = "\n".join(f"{n}. «{row}» × «{col}»" for n, (row, col, _) in enumerate(asked, 1))
    for name, context in variants.items():
        prompt = (
            f"Фрагмент отчёта:\n{context}\n\nДля каждой пары «критерий × функция» выпиши "
            "значение из таблицы ровно как в источнике. Верни JSON: "
            '{"1": "значение", "2": "значение", ...}\n\n' + questions
        )
        answer = json.loads(ollama.generate(prompt, json_format=True, num_predict=400).text)
        for n, (row, col, truth) in enumerate(asked, 1):
            got = re.sub(r"\s", "", str(answer.get(str(n), "")))
            ok = got == re.sub(r"\s", "", truth)
            score[name][0] += ok
            score[name][1] += 1
            if not ok:
                print(f"  [{name}] {row} × {col}: ответ {got!r}, в таблице {truth!r}")

for name, (ok, total) in score.items():
    print(f"{name}: верно {ok}/{total} ({ok / total:.0%})")
