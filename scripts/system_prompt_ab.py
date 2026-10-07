"""Черновик раздела заявки на реальных слайдах: полная цепочка этапа 4 в миниатюре.

Подписи фрагментов + линеаризация широких таблиц + системный промпт, затем
проверка чисел ответа против процитированных фрагментов.
Запуск: python scripts/system_prompt_ab.py [номера слайдов через запятую]
"""

import os
import sys
from pathlib import Path

from bzpo.config import load_config
from bzpo.generation.tables import linearize_wide_tables
from bzpo.generation.validate import check_numbers
from bzpo.ingest.slides import chunk_slides
from bzpo.providers.ollama import OllamaClient

ROOT = Path(__file__).resolve().parent.parent
# md отчёта с тепловой картой на слайде 34; имя файла заказчика в репозиторий не пишется.
SAMPLE = Path(os.environ.get("BZPO_AB_SAMPLE", ROOT / "data/samples/report_dev.md"))
SYSTEM = (ROOT / "bzpo/generation/prompts/section_system.txt").read_text(encoding="utf-8")
PICK = [int(n) for n in sys.argv[1].split(",")] if len(sys.argv) > 1 else [33, 34, 86, 87]
CUSTOMER = "ГК ДЕВ"

deck = chunk_slides(
    SAMPLE.read_text(encoding="utf-8"),
    lambda s: len(s) // 4,
    hard_limit=8192,
)
by_slide = {c.slide: c for c in deck.chunks}
fragments = {
    i: linearize_wide_tables(by_slide[n].text_raw) for i, n in enumerate(PICK, 1)
}
context = "\n\n".join(
    f"[{i}] (Источник: отчёт исполнителя по проекту для {CUSTOMER}, 2025, слайд {n})\n"
    f"{fragments[i]}"
    for i, n in enumerate(PICK, 1)
)
task = (
    "Напиши раздел заявки «Опыт диагностики цифровой зрелости в девелопменте», "
    f"не более 220 слов. Пиши как опыт исполнителя в проекте для {CUSTOMER}.\n\n"
    "Контекст:\n" + context
)

result = OllamaClient(load_config().ollama, role="query").generate(
    task, system=SYSTEM, num_predict=1200
)
print(f"{result.text.strip()}\n\nслов: {len(result.text.split())}")
issues = check_numbers(result.text, fragments)
print(f"неподтверждённых чисел: {len(issues)}")
for issue in issues:
    print(f"  - {issue.reason}: «{issue.sentence[:120]}»")
