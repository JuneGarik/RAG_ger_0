"""Презентация решения для управляющего комитета: python scripts/build_deck.py

Собирает data/БЗПО_презентация_УК.pptx (data/ вне git: в презентации замеры по
документам заказчиков). Стиль — как у отчётов
проекта (ДЕВ, «ОЦО Ритейл»): белый фон, заголовок-вывод с засечками, надпись
раздела над ним, бирюзовые акценты, серая панель комментариев, источник внизу.
Цифры на слайдах — замеры из тестирования (см. слайд «Метрики»); где цифра —
допущение, это написано на слайде.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parents[1] / "data" / "БЗПО_презентация_УК.pptx"

TEAL = RGBColor(0x0E, 0xBE, 0xAD)
TEAL_DARK = RGBColor(0x00, 0x8C, 0x82)
TEAL_LIGHT = RGBColor(0xE5, 0xFA, 0xFB)
NAVY = RGBColor(0x1A, 0x3C, 0x5B)
TEXT = RGBColor(0x22, 0x2B, 0x33)
MUTED = RGBColor(0x7F, 0x86, 0x8D)
PANEL = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
AMBER = RGBColor(0xC9, 0x7A, 0x00)

TITLE_FONT = "Cambria"
BODY_FONT = "Arial"
W, H = Inches(13.333), Inches(7.5)
LEFT = Inches(0.6)
CONTENT_W = W - 2 * LEFT

prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]
page = 0


# --------------------------------------------------------------- примитивы


def text(slide, x, y, w, h, runs, *, size=14, color=TEXT, bold=False, font=BODY_FONT,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=0, name=None):
    """Текстовый блок. runs — строка, список абзацев или абзацы из [(текст, опции)]."""
    box = slide.shapes.add_textbox(x, y, w, h)
    if name:
        box.name = name
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    paragraphs = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paragraphs):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        pieces = para if isinstance(para, list) else [(para, {})]
        for piece, opts in pieces:
            r = p.add_run()
            r.text = piece
            f = r.font
            f.name = opts.get("font", font)
            f.size = Pt(opts.get("size", size))
            f.bold = opts.get("bold", bold)
            f.italic = opts.get("italic", False)
            f.color.rgb = opts.get("color", color)
    return box


def bullets(slide, x, y, w, h, items, *, size=14, color=TEXT, spacing=6):
    """Маркированный список: «•» как маркер абзаца, а не символ в тексте."""
    from pptx.oxml.ns import qn

    box = text(slide, x, y, w, h, items, size=size, color=color, spacing=spacing)
    for p in box.text_frame.paragraphs:
        p_pr = p._p.get_or_add_pPr()
        p_pr.set("marL", str(Emu(Inches(0.22))))
        p_pr.set("indent", str(-Emu(Inches(0.22))))
        bu = p_pr.makeelement(qn("a:buChar"), {"char": "•"})
        font = p_pr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
        p_pr.append(font)
        p_pr.append(bu)
    return box


def rect(slide, x, y, w, h, fill, *, line=None, rounded=False, name=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if rounded:
        shape.adjustments[0] = 0.08
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    _no_style(shape)
    if name:
        shape.name = name
    return shape


def _no_style(shape):
    """Без теней и эффектов темы — как в отчётах (плоские плашки)."""
    shape.shadow.inherit = False
    style = shape._element.find("{http://schemas.openxmlformats.org/presentationml/2006/main}"
                                "style")
    if style is not None:
        shape._element.remove(style)


def rule(slide, x, y, w, color=TEXT, weight=1.5):
    line = slide.shapes.add_connector(1, x, y, x + w, y)
    line.line.color.rgb = color
    line.line.width = Pt(weight)
    return line


def new_slide(section, title, source=None, notes=None):
    """Каркас слайда как в отчётах: раздел, заголовок-вывод, источник, номер."""
    global page
    page += 1
    slide = prs.slides.add_slide(BLANK)
    text(slide, LEFT, Inches(0.28), Inches(9), Inches(0.3), section, size=11, color=MUTED,
         font=TITLE_FONT, name="Раздел")
    text(slide, LEFT, Inches(0.6), CONTENT_W, Inches(0.95), title, size=24, color=TEXT,
         font=TITLE_FONT, name="Заголовок")
    if source:
        text(slide, LEFT, Inches(7.02), Inches(11), Inches(0.3), f"Источник: {source}",
             size=9, color=MUTED, name="Источник")
    badge = rect(slide, W - Inches(0.75), Inches(6.98), Inches(0.32), Inches(0.32), TEAL,
                 name="Номер")
    frame = badge.text_frame
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(page)
    r.font.size, r.font.color.rgb, r.font.name = Pt(10), WHITE, BODY_FONT
    if notes:
        slide.notes_slide.notes_text_frame.text = notes
    return slide


def header(slide, x, y, w, label, color=TEXT):
    """Подзаголовок блока с линией, как «Финансовые показатели…» в отчётах."""
    text(slide, x, y, w, Inches(0.32), label, size=13, bold=True, color=color,
         font=TITLE_FONT)
    rule(slide, x, y + Inches(0.38), w, color=color, weight=1.25)


def panel(slide, x, y, w, h, label, items, *, size=13):
    """Серая панель комментариев справа."""
    rect(slide, x, y, w, h, PANEL)
    text(slide, x + Inches(0.2), y + Inches(0.18), w - Inches(0.4), Inches(0.3), label,
         size=13, bold=True, font=TITLE_FONT)
    rule(slide, x + Inches(0.2), y + Inches(0.55), w - Inches(0.4), weight=1.25)
    bullets(slide, x + Inches(0.2), y + Inches(0.75), w - Inches(0.4), h - Inches(0.9),
            items, size=size, spacing=8)


def table(slide, x, y, w, rows, widths, *, size=11, header_fill=TEAL, row_h=0.36,
          first_bold=True):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w,
                                   Inches(row_h * len(rows)))
    tbl = shape.table
    total = sum(widths)
    for i, width in enumerate(widths):
        tbl.columns[i].width = Emu(int(w * width / total))
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(row_h)
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else (
                WHITE if r % 2 else TEAL_LIGHT)
            frame = cell.text_frame
            frame.word_wrap = True
            p = frame.paragraphs[0]
            run = p.add_run()
            run.text = value
            run.font.name = BODY_FONT
            run.font.size = Pt(size)
            run.font.bold = r == 0 or (first_bold and c == 0)
            run.font.color.rgb = WHITE if r == 0 else TEXT
    return shape


def stat(slide, x, y, w, value, label, color=TEAL_DARK):
    text(slide, x, y, w, Inches(0.7), value, size=34, bold=True, color=color,
         font=TITLE_FONT)
    text(slide, x, y + Inches(0.72), w, Inches(0.7), label, size=12, color=TEXT)


def chip(slide, x, y, w, h, title, body, fill=TEAL_LIGHT, title_color=TEAL_DARK, size=12):
    rect(slide, x, y, w, h, fill, rounded=True)
    text(slide, x + Inches(0.18), y + Inches(0.14), w - Inches(0.36), Inches(0.35), title,
         size=14, bold=True, color=title_color, font=TITLE_FONT)
    bullets(slide, x + Inches(0.18), y + Inches(0.55), w - Inches(0.36), h - Inches(0.65),
            body, size=size, spacing=5)


# --------------------------------------------------------------- слайды

# 1. Титул
page += 1
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
text(s, LEFT, Inches(2.55), Inches(11), Inches(1.6),
     "БЗПО — база знаний проектного опыта", size=40, color=WHITE, font=TITLE_FONT,
     bold=True)
text(s, LEFT, Inches(3.55), Inches(11), Inches(1.2),
     "Черновики подхода к проекту и разделов заявки — из материалов прошлых проектов, "
     "со ссылками на слайды-источники и проверкой каждого утверждения",
     size=18, color=RGBColor(0xCF, 0xE8, 0xEC))
text(s, LEFT, Inches(6.4), Inches(11), Inches(0.4),
     "Управляющий комитет · октябрь 2026 · PoC на локальной инфраструктуре",
     size=12, color=RGBColor(0xA9, 0xC4, 0xD6))
s.notes_slide.notes_text_frame.text = (
    "5 минут: 5 слайдов по требованиям УК (эффект, ПО, готовность, цифры, контур человека). "
    "Дальше — приложение для технической команды, на защите его не показываем.")

# 2. Для кого и эффект
s = new_slide(
    "1. Для кого решение и какой полезный эффект",
    "Руководитель проекта получает черновик подхода к новому проекту за 2–3 минуты, "
    "собранный из реально применённых подходов прошлых проектов",
    source="SPEC БЗПО v1.1 (пользователи, сценарии); тестирование PoC, октябрь 2026",
    notes=("Ключевое: знания компании лежат в сотнях слайдов отчётов и КП, найти нужный "
           "прошлый подход вручную долго. Система находит слайды по смыслу, предлагает "
           "подход со ссылками и честно показывает, что не подтверждено."))
header(s, LEFT, Inches(1.75), Inches(7.6), "Пользователи и сценарии")
chip(s, LEFT, Inches(2.3), Inches(3.65), Inches(1.85), "Кто пользуется", [
    "12 руководителей проектов и 6 директоров, перспектива — до 100",
    "Подготовка тендерных заявок и отчётов",
])
chip(s, LEFT + Inches(3.95), Inches(2.3), Inches(3.65), Inches(1.85), "Что делает", [
    "Подход к новому проекту по 6 граням",
    "Поиск по слайдам прошлых отчётов и КП",
    "Загрузка pptx / Word → база знаний",
])
header(s, LEFT, Inches(4.5), Inches(7.6), "Полезный эффект")
bullets(s, LEFT, Inches(5.05), Inches(7.6), Inches(1.6), [
    "Опыт компании не теряется в архиве: ищется по смыслу, а не по имени файла",
    "Каждое утверждение — со ссылкой на слайд-источник; непроверенное помечено",
    "Достоверность по каждой грани подхода в % — видно, где опора слабая",
], size=14)
panel(s, Inches(8.6), Inches(1.75), Inches(4.13), Inches(4.95), "Пример запроса", [
    "Цель: «Стратегия цифровой трансформации», отрасль — производство",
    "Ответ: подход с опорой на прошлые проекты, этапы, риски, контрольные точки",
    "Достоверность граней в тесте: методология 99%, этапы 91%, риски 86%, результат 80%",
    "Всё работает локально: данные заказчиков не покидают контур",
])

# 3. ПО и системные требования
s = new_slide(
    "2. ПО и системные требования",
    "Всё решение работает локально на ноутбуке 16 ГБ; для команды нужен один сервер "
    "с GPU 24 ГБ — внешних облачных сервисов нет",
    source="docs/LAPTOP.md, config.yaml, SPEC БЗПО раздел 1; замеры на MacBook M5 16 ГБ",
    notes=("Важно для ИБ: ни один документ не уходит наружу. Модели скачаны один раз; "
           "далее работа без интернета (HF_HUB_OFFLINE=1)."))
table(s, LEFT, Inches(1.75), Inches(8.0), [
    ["Компонент", "Что используется", "Память"],
    ["Эмбеддинги", "Giga-Embeddings-instruct 480M (Сбер)", "~1 ГБ"],
    ["Реранкер", "bge-reranker-v2-m3 (BAAI)", "~1,2 ГБ"],
    ["Генерация", "Qwen3.5 9B (Ollama)", "~6,6 ГБ"],
    ["Черновик реестра", "GigaChat 3.1 Lightning (Ollama)", "~6,5 ГБ"],
    ["Картинки, OCR", "Qwen3-VL 8B instruct (Ollama)", "~6 ГБ"],
    ["Хранилище", "PostgreSQL 18 + pgvector 0.8", "~0,5 ГБ"],
    ["Рендер слайдов", "LibreOffice 26.8", "—"],
    ["Интерфейс", "Python 3.13, Streamlit", "—"],
], [1.6, 3.6, 1.0], size=11, row_h=0.42)
panel(s, Inches(8.95), Inches(1.75), Inches(3.78), Inches(4.95), "Требования", [
    "Ноутбук (PoC): Apple Silicon, 16 ГБ — в памяти одна LLM за раз",
    "Сервер (команда): GPU 24 ГБ, 32+ ГБ RAM, 100 ГБ диска",
    "Диск под модели ~25 ГБ",
    "Лицензии: открытые модели и ПО (MIT, Apache 2.0, PostgreSQL)",
    "Интернет — только для разовой загрузки моделей",
])

# 4. Готовность к проду
s = new_slide(
    "3. Готовность к «проду»",
    "Сквозной сценарий работает на реальных документах; до пилота для команды — около "
    "8–10 недель доработок и решение УК по двум вопросам",
    source="docs/PARSING_PLAN.md, README.md; сроки — оценка команды разработки",
    notes=("Сроки — оценка, не обязательство. Два решения от УК: можно ли использовать "
           "внешний API для качественной конвертации и выделяется ли сервер с GPU."))
header(s, LEFT, Inches(1.75), Inches(4.1), "Готово")
bullets(s, LEFT, Inches(2.3), Inches(4.1), Inches(4.5), [
    "Загрузка pptx / Word / md с автоопределением формата",
    "Конвертация с проверкой против исходника",
    "Реестр проектов: черновик от модели, проверка человеком",
    "Поиск: вектор + полнотекст + реранкер",
    "Генерация подхода с проверкой и % по граням",
    "Веб-интерфейс, 202 автотеста",
], size=13)
header(s, Inches(5.0), Inches(1.75), Inches(7.73), "Осталось доработать (оценка сроков)")
table(s, Inches(5.0), Inches(2.3), Inches(7.73), [
    ["Работа", "Срок"],
    ["Эталонный набор вопросов и метрики качества", "2 нед."],
    ["Разбор запроса, гипотезы HyDE, порог «не найдено»", "1–2 нед."],
    ["Разделы-образцы, база подтверждений (акты, письма)", "2–3 нед."],
    ["Сервер: многопользовательский доступ, роли, бэкапы", "2–3 нед."],
    ["Наполнение корпуса (~25 документов), проверка реестра", "параллельно"],
], [5.2, 1.2], size=11, row_h=0.45)
rect(s, Inches(5.0), Inches(5.2), Inches(7.73), Inches(1.5), PANEL)
text(s, Inches(5.2), Inches(5.32), Inches(7.4), Inches(0.3), "Нужны решения УК",
     size=13, bold=True, font=TITLE_FONT, color=AMBER)
bullets(s, Inches(5.2), Inches(5.7), Inches(7.4), Inches(1.0), [
    "Сервер с GPU 24 ГБ для командного доступа",
    "Можно ли конвертировать документы через внешний API — качество выше, "
    "но нужна оценка конфиденциальности",
], size=12, spacing=4)

# 5. Эффект в цифрах
s = new_slide(
    "4. Эффект в цифрах «до / после»",
    "Время «после» замерено на реальных документах; «до» и деньги посчитаем на пилоте — "
    "сейчас нет хронометража ручной работы",
    source="замеры PoC на MacBook M5 16 ГБ; «до» — допущение для иллюстрации, требует "
           "хронометража",
    notes=("Честно говорим: «после» — замер, «до» — допущение. Деньги зависят от числа "
           "заявок и ставки часа, а главный эффект (выигранные тендеры) нельзя приписать "
           "системе без нескольких месяцев наблюдения."))
header(s, LEFT, Inches(1.75), Inches(7.6), "Замерено «после»")
stat(s, LEFT, Inches(2.3), Inches(2.4), "2–3 мин", "черновик подхода по 6 граням со ссылками")
stat(s, LEFT + Inches(2.6), Inches(2.3), Inches(2.4), "~10 с", "поиск по базе слайдов")
stat(s, LEFT + Inches(5.2), Inches(2.3), Inches(2.4), "41 с", "индексация отчёта на 147 слайдов")
table(s, LEFT, Inches(4.0), Inches(7.6), [
    ["Задача", "До (допущение)", "После (замер)"],
    ["Найти опыт по теме в архиве отчётов", "2–4 ч", "~10 с + чтение"],
    ["Черновик раздела «Подход»", "4–8 ч", "2–3 мин + проверка ~1 ч"],
    ["Внести отчёт в базу знаний", "—", "1–25 мин, без участия"],
], [3.3, 1.8, 2.5], size=11, row_h=0.42)
panel(s, Inches(8.6), Inches(1.75), Inches(4.13), Inches(4.95), "Почему деньги — позже", [
    "Нет замера ручной работы «до» — нужен хронометраж на 10 заявках",
    "Экономия = часы × число заявок × ставка; число заявок в год — уточнить",
    "Главный эффект — выигранные тендеры — проявится через месяцы",
    "Пилот: замерить время 5 РП на 10 задачах с системой и без",
], size=12)

# 6. Модель / детерминизм / человек
s = new_slide(
    "5. Что детерминировано, что решает модель, где человек",
    "Всё, что есть в файле, извлекает и проверяет код; модель пишет текст и толкует "
    "картинки; решения о публичности и итоговый текст — за человеком",
    source="bzpo/parsing, bzpo/generation, bzpo/registry; тестирование PoC",
    notes=("Принцип, подтверждённый тестами: где код может определить точно — определяет код. "
           "Модель на тепловой карте угадала 2 ячейки из 12, код — 96 из 96."))
col_w = Inches(3.9)
chip(s, LEFT, Inches(1.75), col_w, Inches(2.75), "Код (детерминировано)", [
    "Текст, таблицы, данные диаграмм из файла",
    "Тепловые карты, шкалы, подписи диаграмм",
    "Проверка чисел и полноты против исходника",
    "Поиск, фильтр публичности, ссылки на источники",
], size=12)
chip(s, LEFT + col_w + Inches(0.17), Inches(1.75), col_w, Inches(2.75), "Модель", [
    "Смысловой поиск и ранжирование",
    "Черновик подхода (Qwen3.5 9B)",
    "Черновик строки реестра (GigaChat)",
    "Текст со скриншотов, толкование картинок",
], fill=RGBColor(0xEA, 0xF1, 0xF8), title_color=NAVY, size=12)
chip(s, LEFT + 2 * (col_w + Inches(0.17)), Inches(1.75), col_w, Inches(2.75),
     "Человек в контуре", [
         "Проверяет строку реестра: код, публичность, тендер",
         "Разбирает блок «Не подтверждено»",
         "Проверяет текст, распознанный с картинок",
         "Принимает и правит итоговый черновик",
     ], fill=RGBColor(0xFD, 0xF3, 0xE3), title_color=AMBER, size=12)
header(s, LEFT, Inches(4.85), CONTENT_W, "Проверено на кейсах")
text(s, LEFT, Inches(5.4), CONTENT_W, Inches(0.8), [[
    ("Отчёт ДЕВ ", {"bold": True}), ("(pptx, 147 слайдов) · ", {}),
    ("КП «ОЦО Ритейл» ", {"bold": True}), ("(pptx, 13 слайдов) · ", {}),
    ("Отчёт НГК ", {"bold": True}), ("(Word, 68 рисунков-скриншотов) · ", {}),
    ("генерация подхода ", {"bold": True}), ("для девелопмента, недвижимости, производства",
                                              {}),
]], size=13)

# --------------------------------------------------------------- приложение

page += 1
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
text(s, LEFT, Inches(2.8), Inches(11), Inches(0.9), "Приложение", size=40, bold=True,
     color=WHITE, font=TITLE_FONT)
text(s, LEFT, Inches(3.7), Inches(11), Inches(1.2),
     "Техническая часть для команды: стек, устройство репозитория, конвейер, модели, "
     "метрики тестирования, запуск и ограничения", size=18,
     color=RGBColor(0xCF, 0xE8, 0xEC))

# A1. Стек
s = new_slide(
    "Приложение · Стек",
    "Стек — Python, PostgreSQL и локальные модели; одна команда запускает всё решение",
    source="pyproject.toml, config.yaml, start.sh")
table(s, LEFT, Inches(1.75), CONTENT_W, [
    ["Слой", "Технологии", "Где в коде"],
    ["Язык, окружение", "Python 3.13, conda/venv, pydantic-settings, ruff, pytest",
     "pyproject.toml, config.yaml"],
    ["Хранилище", "PostgreSQL 18, pgvector 0.8 (векторы 1024), полнотекст russian",
     "bzpo/storage, bzpo/migrations"],
    ["Модели в процессе", "sentence-transformers, torch (MPS): эмбеддер и реранкер",
     "bzpo/providers/local.py"],
    ["LLM-сервис", "Ollama 0.35: Qwen3.5 9B, GigaChat 3.1 Lightning, Qwen3-VL 8B",
     "bzpo/providers/ollama.py"],
    ["Разбор документов", "python-pptx, python-docx, LibreOffice, pypdfium2",
     "bzpo/parsing"],
    ["Реестр", "openpyxl (Excel), валидация по SPEC 4.5", "bzpo/registry"],
    ["Интерфейс", "Streamlit, запуск ./start.sh (поднимает Postgres и Ollama)",
     "bzpo/ui/app.py"],
], [1.6, 4.4, 2.2], size=12, row_h=0.55)

# A2. Карта репозитория
s = new_slide(
    "Приложение · Репозиторий",
    "Код разложен по модулям конвейера; база — только в storage, сеть — только в providers",
    source="git, ветка main; строки — подсчёт по *.py; правило границ проверяет "
           "tests/test_boundaries.py")
table(s, LEFT, Inches(1.7), CONTENT_W, [
    ["Папка", "Назначение", "Строк"],
    ["bzpo/parsing", "pptx/docx → md: извлечение, тепловые карты, шкалы, диаграммы, рендер, "
                     "модель, OCR, проверка против исходника", "1 846"],
    ["bzpo/providers", "Эмбеддер, реранкер, клиент Ollama; внешний сервис (режим external)",
     "1 357"],
    ["bzpo/registry", "Реестр Excel: черновик от GigaChat, загрузчик, диагностика, проверка",
     "1 065"],
    ["bzpo/ui", "Веб-интерфейс: подход, поиск, конвертация, документы, черновики", "814"],
    ["bzpo/generation", "Подход по граням: контекст, промпты, проверка, достоверность", "798"],
    ["bzpo/ingest", "Нарезка слайдов и документов, индексация с версиями", "612"],
    ["bzpo/storage", "Весь SQL: реестр, документы, чанки, поиск, миграции", "577"],
    ["bzpo/search", "RRF, реранкер, boost, MMR, фильтр публичности", "564"],
    ["tests · scripts · docs", "202 теста · A/B моделей, сборка презентации · SPEC, план, "
                               "инструкции", "3 398"],
    ["data/ (не в git)", "samples — md и реестр; sources — оригиналы; cache — рендер, "
                         "ответы моделей; drafts — черновики", "—"],
], [1.7, 6.0, 0.7], size=10, row_h=0.45)

# A3. Конвейер
s = new_slide(
    "Приложение · Конвейер",
    "Документ проходит семь шагов; на каждом стоит проверка, а не доверие модели",
    source="bzpo/parsing, bzpo/registry, bzpo/ingest, bzpo/search, bzpo/generation")
steps = [
    ("1", "Загрузка", "формат по содержимому: pptx, docx, md"),
    ("2", "Конвертация", "XML → md; картинки — модель; проверка против исходника"),
    ("3", "Реестр", "GigaChat заполняет, человек проверяет"),
    ("4", "Индексация", "слайд = фрагмент; 2 вектора; версии индекса"),
    ("5", "Поиск", "вектор + полнотекст → RRF → реранкер → boost → MMR"),
    ("6", "Генерация", "6 граней подхода → контекст → Qwen3.5"),
    ("7", "Проверка", "числа, ссылки, перенос, % по граням"),
]
box_w, gap = Inches(1.6), Inches(0.12)
for i, (num, title, body) in enumerate(steps):
    x = LEFT + i * (box_w + gap)
    rect(s, x, Inches(1.9), box_w, Inches(2.5), TEAL_LIGHT if i % 2 == 0 else PANEL,
         rounded=True)
    circle = s.shapes.add_shape(MSO_SHAPE.OVAL, x + Inches(0.15), Inches(2.05),
                                Inches(0.45), Inches(0.45))
    circle.fill.solid()
    circle.fill.fore_color.rgb = TEAL
    circle.line.fill.background()
    _no_style(circle)
    frame = circle.text_frame
    frame.margin_left = frame.margin_right = 0
    p = frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = num
    run.font.size, run.font.bold, run.font.color.rgb = Pt(14), True, WHITE
    text(s, x + Inches(0.15), Inches(2.62), box_w - Inches(0.3), Inches(0.35), title,
         size=13, bold=True, font=TITLE_FONT)
    text(s, x + Inches(0.15), Inches(3.0), box_w - Inches(0.3), Inches(1.35), body, size=11)
panel(s, LEFT, Inches(4.7), CONTENT_W, Inches(2.1), "Ключевые решения", [
    "Слайд не режется: предел фрагмента = крупнейший слайд; таблица не теряет шапку",
    "Заполнение «посыла» и «визуала» моделью — для поиска, но не для цитирования",
    "Широкие таблицы подаются модели построчно: точность чтения ячеек 81% → 100%",
], size=12)

# A4. Модели и роли
s = new_slide(
    "Приложение · Модели",
    "Каждой задаче — своя модель: выбор сделан по сравнению на документах проекта, "
    "а не по бенчмаркам",
    source="config.yaml (ollama.model_*), scripts/compare_llm.py, scripts/wide_table_ab.py")
table(s, LEFT, Inches(1.75), CONTENT_W, [
    ["Роль", "Модель", "Почему она", "Скорость на M5"],
    ["Поиск: векторы", "Giga-Embeddings 480M", "Русский, 1024 изм. = схема БД без миграции",
     "147 слайдов ~20 с"],
    ["Ранжирование", "bge-reranker-v2-m3", "Точная оценка пары «вопрос–слайд»",
     "50 кандидатов ~3 с"],
    ["Подход, запросы", "Qwen3.5 9B", "Держит ссылки, не выдумывает факты", "~23 ток/с"],
    ["Черновик реестра", "GigaChat 3.1 Lightning", "Быстрый, русский; поля проверяет код",
     "~75 ток/с"],
    ["Картинки и OCR", "Qwen3-VL 8B instruct", "Дословный текст без ошибок, в отличие от "
                                               "GLM-OCR", "~20–30 с на слайд"],
], [1.5, 2.0, 3.6, 1.6], size=11, row_h=0.6)
panel(s, LEFT, Inches(5.55), CONTENT_W, Inches(1.25), "Отклонено по тестам", [
    "GigaChat для генерации: выдумал абзац фактов и ссылку на несуществующий источник",
], size=12)

# A5. Метрики
s = new_slide(
    "Приложение · Метрики",
    "Что удалось замерить за время тестирования PoC",
    source="замеры на MacBook M5 16 ГБ, отчёты ДЕВ, «ОЦО Ритейл», НГК; скрипты scripts/")
table(s, LEFT, Inches(1.65), CONTENT_W, [
    ["Что измеряли", "Результат"],
    ["Конвертация отчёта ДЕВ кодом: полнота текста / числа без опоры в XML",
     "99,8% / 0 (md, сделанный вручную через Claude: 96,9% / 40)"],
    ["Тепловая карта ДЕВ (96 ячеек): код по легенде против мультимодальной модели",
     "96 из 96 против 2 из 12"],
    ["Подписи диаграмм think-cell кодом / шкалы-оценки «Ритейла»", "27 из 29 / 16 из 16"],
    ["Чтение широких таблиц моделью: как есть → построчно (36 вопросов)", "81% → 100%"],
    ["Поиск: нужный слайд в топ-3 на контрольных вопросах", "3 из 3"],
    ["Индексация 147 слайдов / повторный прогон без изменений", "41 с / 0 с"],
    ["Генерация подхода: тщательно / быстро", "2,5–3 мин / ~2 мин"],
    ["Разбор картинок моделью: окно 24k → 8k", "~80 → ~30 с на слайд"],
    ["Текст со скриншотов Word (отчёт НГК): Qwen3-VL против GLM-OCR",
     "без ошибок против 5+ ошибок на слайд; ~20 с"],
    ["Черновик реестра GigaChat / роли команды по таблице", "30–90 с / 9 из 9"],
    ["Автотесты / находки внешнего код-ревью закрыты", "202 / 7 из 7"],
], [5.0, 3.6], size=11, row_h=0.41)

# A6. Запуск и эксплуатация
s = new_slide(
    "Приложение · Запуск",
    "Для работы достаточно одной команды; операции блокируют друг друга, чтобы не "
    "перегружать 16 ГБ",
    source="README.md, docs/LAPTOP.md, start.sh")
header(s, LEFT, Inches(1.75), Inches(6.0), "Команды")
text(s, LEFT, Inches(2.3), Inches(6.0), Inches(3.6), [
    [("./start.sh", {"bold": True})],
    [("веб-интерфейс http://localhost:8501", {"color": MUTED, "size": 11})],
    [("python -m bzpo.parsing convert data/sources/x.pptx --vlm", {"bold": True})],
    [("конвертация в md с разбором картинок", {"color": MUTED, "size": 11})],
    [("python -m bzpo.registry draft data/samples/x.md", {"bold": True})],
    [("черновик строки реестра", {"color": MUTED, "size": 11})],
    [("python -m bzpo.ingest", {"bold": True})],
    [("загрузка реестра и индексация", {"color": MUTED, "size": 11})],
    [("python -m pytest tests -q", {"bold": True})],
    [("202 теста, без базы и сети", {"color": MUTED, "size": 11})],
], size=12, font="Courier New", spacing=3)
panel(s, Inches(7.0), Inches(1.75), Inches(5.73), Inches(4.1), "Что знать новому человеку", [
    "Секреты — только в окружении: BZPO_DB__DSN, токены",
    "В data/samples — только md и реестр; оригиналы — в data/sources",
    "В индекс попадает только строка реестра со статусом «проверено»",
    "Смена модели эмбеддера → новая версия индекса, старая работает до конца сборки",
    "Одна долгая операция за раз: генерация, конвертация, индексация",
    "Ответы моделей кэшируются в data/cache — повтор быстрый",
], size=12)

# A7. Ограничения
s = new_slide(
    "Приложение · Ограничения",
    "Известные ограничения PoC и что с ними делать",
    source="docs/PARSING_PLAN.md, README.md")
table(s, LEFT, Inches(1.75), CONTENT_W, [
    ["Ограничение", "Влияние", "Что делаем"],
    ["Толкование картинок моделью 8B поверхностное, бывают ошибки",
     "Посыл слайда может быть неточен", "Не цитируется; модель мощнее или API — решение УК"],
    ["Текст со скриншотов в Word не с чем сверить", "Возможны ошибки OCR",
     "Помечен «проверить»; выборочная проверка"],
    ["Генерация ~2,5–3 мин на ноутбуке", "Выше бюджета SPEC (60 с)", "Сервер с GPU"],
    ["Нет эталонного набора вопросов", "Качество поиска — на глаз",
     "Этап 3: набор и метрики"],
    ["PDF не поддержан", "Часть архива не загрузить", "Отдельный этап по SPEC"],
    ["Один пользователь за раз на ноутбуке", "Не для команды", "Серверное развёртывание"],
], [3.6, 2.4, 3.2], size=11, row_h=0.6)

prs.save(OUT)
print(OUT)
