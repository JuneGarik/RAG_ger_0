"""Рендер документа в картинки страниц: LibreOffice -> PDF -> PNG (П2).

LibreOffice работает без окон (`--headless`) и одинаково на ноутбуке и на
сервере. Результат кэшируется по хэшу файла: повторная конвертация того же
документа рендер не повторяет. 147 слайдов ДЕВ — около 15 с.

PDF -> PNG — pypdfium2 (лицензия BSD/Apache, ставится через pip без
системных библиотек).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

SOFFICE_CANDIDATES = (
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "soffice",
    "libreoffice",
)
SCALE = 1.5  # 1440×810 для 16:9 — мелкий текст читается, картинка ~0,3 МБ
TIMEOUT = 600


class RenderError(RuntimeError):
    """LibreOffice не найден или не смог отрисовать документ."""


def soffice() -> str | None:
    for candidate in SOFFICE_CANDIDATES:
        if Path(candidate).is_file() or shutil.which(candidate):
            return candidate if Path(candidate).is_file() else shutil.which(candidate)
    return None


def render_pages(source: Path, cache_dir: Path) -> list[Path]:
    """PNG каждой страницы (слайда) в порядке документа; из кэша, если уже отрисован."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = cache_dir / digest
    pages = sorted(target.glob("page_*.png"))
    if pages and (target / "done").exists():
        return pages

    binary = soffice()
    if binary is None:
        raise RenderError("LibreOffice не найден: установите его с libreoffice.org")
    target.mkdir(parents=True, exist_ok=True)
    pdf = target / "document.pdf"
    workdir = target / "pdf"
    workdir.mkdir(exist_ok=True)
    # Отдельный профиль: не мешает открытому LibreOffice пользователя.
    profile = (target / "profile").resolve().as_uri()
    try:
        subprocess.run(
            [binary, "--headless", "--norestore", f"-env:UserInstallation={profile}",
             "--convert-to", "pdf", "--outdir", str(workdir), str(source)],
            check=True, capture_output=True, timeout=TIMEOUT,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RenderError(f"LibreOffice не отрисовал {source.name}: {exc}") from exc
    produced = next(workdir.glob("*.pdf"), None)
    if produced is None:
        raise RenderError(f"LibreOffice не создал PDF для {source.name}")
    produced.replace(pdf)

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf))
    pages = []
    for index in range(len(document)):
        path = target / f"page_{index + 1:04d}.png"
        document[index].render(scale=SCALE).to_pil().save(path)
        pages.append(path)
    (target / "done").write_text(source.name, encoding="utf-8")
    return pages
