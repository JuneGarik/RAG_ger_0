from __future__ import annotations

from bzpo.storage.repo import ChunkHit


def hit(text: str, section: str | None, page: int | None = None) -> ChunkHit:
    return ChunkHit("id", text, text, [], page, section, "Отчёт", None, None, None,
                    "open", None, None)


def test_slide_locator() -> None:
    assert hit("…", "01. Раздел", page=14).locator == "слайд 14"


def test_document_locator_names_figure_and_shortens_heading() -> None:
    section = ("Этап 2. Организационное и методическое сопровождение при проведении анализа "
               "цифровой трансформации нефтегазовой отрасли")
    text = "Абзац\n<!-- figure: 2; kind: slide_screenshot -->\n**Рисунок 2 …**"
    assert hit(text, section).locator == "раздел «Этап 2», рисунок 2 (скриншот слайда)"


def test_document_locator_without_figures() -> None:
    long = "Предложения по содержательной составляющей и оформлению презентационных материалов"
    assert hit("текст", long).locator == (
        "раздел «Предложения по содержательной составляющей и оформлению…»")
    assert hit("текст", None).locator == "без раздела"
