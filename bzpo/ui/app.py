"""Веб-интерфейс БЗПО (Streamlit).

    ./start.sh                      # или: streamlit run bzpo/ui/app.py

Вкладки: генерация подхода, поиск, документы (реестр и индексация), черновики.
Модели эмбеддинга и реранкинга загружаются один раз на процесс
(st.cache_resource), поэтому повторные запросы идут без загрузки весов.
Длинные операции реестра и индексации запускаются теми же командами, что и из
терминала (subprocess), — у интерфейса нет своей версии этой логики.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from bzpo.config import load_config
from bzpo.generation.approach import (
    FACET_LABELS,
    ApproachRequest,
    FacetScore,
    facet_queries,
    retrieval_confidence,
)
from bzpo.generation.run import MODES as MODES_PARAMS
from bzpo.generation.run import generate_approach, save_draft, search_facets
from bzpo.ingest.slides import _split_frontmatter
from bzpo.parsing.check import COVERAGE_MIN, DeckCheck
from bzpo.parsing.convert import convert
from bzpo.parsing.render import RenderError
from bzpo.providers.local import LocalEmbedder, LocalReranker
from bzpo.providers.ollama import OllamaClient
from bzpo.registry.loader import diagnose, load_registry
from bzpo.registry.workbook import (
    DocumentRow,
    RegistryError,
    add_document,
    file_key,
    is_registry,
    project_row,
    registered_files,
    review_project,
)
from bzpo.search.query import SearchResult, search
from bzpo.storage import repo

ROOT = Path(__file__).resolve().parents[2]
# BZPO_DATA_DIR — подменить папку данных (тесты интерфейса на копии реестра).
DATA = Path(os.environ.get("BZPO_DATA_DIR", ROOT / "data"))
SAMPLES = DATA / "samples"
DRAFTS = DATA / "drafts"
SOURCES = DATA / "sources"  # оригиналы pptx/docx; в data/samples — только md и xlsx
REGISTRY = DATA / "registry.xlsx"
# Команды получают пути явно: папка данных страницы может быть подменена.
INGEST = ["bzpo.ingest", "--registry", str(REGISTRY), "--data-root", str(DATA)]
DRAFT = ["bzpo.registry", "draft", "--registry", str(REGISTRY), "--data-root", str(DATA)]
DOC_TYPES = {"Отчёт": "report", "КП": "kp", "Презентация": "presentation", "Акт": "act"}
MODES = {"Тщательно (~3 мин)": "thorough", "Быстро (~2 мин)": "quick"}

st.set_page_config(page_title="БЗПО", page_icon="📚", layout="wide")


# ------------------------------------------------------------ ресурсы


@st.cache_resource(show_spinner="Загрузка конфигурации…")
def settings():
    return load_config(ROOT / "config.yaml")


@st.cache_resource(show_spinner="Загрузка эмбеддера и реранкера (один раз, ~10 с)…")
def models():
    cfg = settings().local
    embedder, reranker = LocalEmbedder(cfg), LocalReranker(cfg)
    embedder.embed_query("прогрев")
    reranker.rerank("прогрев", ["прогрев"])
    return embedder, reranker


def connection():
    cfg = settings()
    if cfg.db.dsn is None:
        st.error("Не задан BZPO_DB__DSN. Запускайте интерфейс через ./start.sh.")
        st.stop()
    return repo.connect(cfg.db.dsn.get_secret_value(), cfg.db.statement_timeout_ms)


def run_command(args: list[str], title: str) -> bool:
    """Запустить команду проекта, показать вывод и сохранить его до следующей перерисовки."""
    with st.status(title, expanded=True) as status:
        result = subprocess.run(
            [sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True,
        )
        lines = [line for line in (result.stdout + result.stderr).splitlines()
                 if "httpx" not in line and "Loading" not in line]
        output = "\n".join(lines[-40:]) or "(нет вывода)"
        st.code(output, language=None)
        ok = result.returncode == 0
        status.update(label=f"{title}: {'готово' if ok else 'ошибка'}",
                      state="complete" if ok else "error")
    st.session_state["command_log"] = (title, ok, output)
    return ok


def show_command_log() -> None:
    """Итог последней команды: после неё страница перерисовывается, вывод не теряется."""
    log = st.session_state.get("command_log")
    if log:
        title, ok, output = log
        with st.expander(f"{'✅' if ok else '❌'} {title}: результат последнего запуска"):
            st.code(output, language=None)


# ------------------------------------------------------------ операции
#
# Долгие операции (генерация, конвертация, черновик реестра, индексация) идут по
# одной на весь процесс: пока одна выполняется, кнопки остальных заблокированы во
# всех вкладках браузера, а рядом с кнопками виден статус. Нажатие кнопки только
# ставит операцию в очередь (колбэк, до перерисовки — поэтому кнопки успевают
# стать неактивными), а выполняется она там, где её кнопка.

OPERATION_STALE = 3 * 60 * 60  # блокировка старше 3 ч — след упавшего процесса, снимается


@st.cache_resource
def _operation() -> dict[str, Any]:
    return {"kind": None, "label": None, "started": 0.0}


def running() -> dict[str, Any] | None:
    op = _operation()
    if op["kind"] and time.time() - op["started"] > OPERATION_STALE:
        op.update(kind=None, label=None)
    return op if op["kind"] else None


def request(kind: str, label: str, **payload: Any) -> None:
    """Колбэк кнопки: занять операцию и запомнить, что выполнить."""
    op = _operation()
    if running():
        return
    op.update(kind=kind, label=label, started=time.time())
    st.session_state["pending"] = {"kind": kind, **payload}


def pending(kind: str) -> dict[str, Any] | None:
    """Операция этого вида, запрошенная в этой вкладке и ещё не выполненная."""
    job = st.session_state.get("pending")
    return job if job and job["kind"] == kind else None


def finish() -> None:
    _operation().update(kind=None, label=None)
    st.session_state.pop("pending", None)


def execute(kind: str, action: Any) -> None:
    """Выполнить запрошенную операцию здесь, затем снять блокировку и перерисовать."""
    job = pending(kind)
    if job is None:
        return
    label = _operation()["label"]
    try:
        action(job)
    except Exception as exc:  # noqa: BLE001 - пользователю сообщение, а не стектрейс
        st.session_state["operation_error"] = (label, f"{type(exc).__name__}: {exc}")
    finally:
        finish()
    st.rerun()


def show_operation_error() -> None:
    """Ошибка последней операции — понятным сообщением (SPEC 10а: без стектрейсов)."""
    error = st.session_state.pop("operation_error", None)
    if error:
        label, detail = error
        st.error(f"«{label}» не завершилась. Частично готовое сохранено в кэше — "
                 "повторный запуск продолжит с того же места.")
        with st.expander("Подробности для разработчика"):
            st.code(detail, language=None)


def status_line() -> None:
    """Рядом с кнопками: какая операция идёт и сколько уже длится; ошибка прошлой."""
    show_operation_error()
    op = running()
    if op:
        st.info(f"⏳ Идёт: **{op['label']}** — {time.time() - op['started']:.0f} с. "
                "Остальные действия заблокированы до её завершения.")


def locked() -> bool:
    return running() is not None


# ------------------------------------------------------------ боковая панель


def sidebar() -> tuple[str, repo.SearchScope]:
    cfg = settings()
    st.sidebar.title("📚 БЗПО")

    st.sidebar.subheader("Состояние")
    documents: list[repo.DocumentInfo] = []
    try:
        with connection() as conn:
            version = repo.active_index_version(conn)
            documents = repo.list_documents(conn, version)
        chunks = sum(d.chunks for d in documents)
        st.sidebar.success(f"База: {len(documents)} док., {chunks} фрагментов")
        if version is None:
            st.sidebar.warning("Нет активного индекса — вкладка «Документы» → «Проиндексировать»")
    except Exception as exc:  # noqa: BLE001 - показать пользователю, а не упасть
        st.sidebar.error(f"PostgreSQL недоступен: запустите Postgres.app\n\n{exc}")

    with OllamaClient(cfg.ollama, role="query") as client:
        installed = client.installed_models()
    if installed is None:
        st.sidebar.error("Ollama не отвечает: запустите приложение Ollama")
    else:
        missing = [m for m in (cfg.ollama.model_query, cfg.ollama.model_enrichment)
                   if not any(name == m or name == f"{m}:latest" for name in installed)]
        if missing:
            st.sidebar.warning("Не скачаны модели: " + ", ".join(missing))
        else:
            st.sidebar.success(f"Ollama: {cfg.ollama.model_query}")

    op = running()
    if op:
        st.sidebar.warning(f"⏳ Идёт: {op['label']} ({time.time() - op['started']:.0f} с)")
    st.sidebar.subheader("Параметры")
    mode = MODES[st.sidebar.radio("Режим генерации", list(MODES), index=0)]
    types = st.sidebar.multiselect(
        "Тип материалов", list(DOC_TYPES), placeholder="все типы",
        help="Искать только в отчётах, только в КП и т. д. Пусто — во всех.",
    )
    by_title = {d.title: d.doc_id for d in documents}
    chosen = st.sidebar.multiselect(
        "Документы", list(by_title), placeholder="все документы",
        help="Ограничить поиск конкретными документами. Пусто — по всей базе.",
    )
    scope = repo.SearchScope(
        doc_types=[DOC_TYPES[t] for t in types] or None,
        doc_ids=[by_title[t] for t in chosen] or None,
    )
    return mode, scope


# ------------------------------------------------------------ вкладки


def _relevance(score: float) -> str:
    if score >= 0.8:
        return "🟢 высокая"
    if score >= 0.5:
        return "🟡 средняя"
    return "⚪ низкая"


def _confidence_badge(value: int | None) -> str:
    if value is None:
        return "—"
    icon = "🟢" if value >= 75 else "🟡" if value >= 50 else "🔴"
    return f"{icon} {value}%"


def render_facet(
    facet: str, result: SearchResult, query: str, score: FacetScore | None = None
) -> None:
    """Раскрывающийся блок грани: что ищем, что нашли и насколько этому верить.

    До генерации показывается предварительная достоверность (только опора
    поиска), после — итоговая: опора поиска × доля утверждений без проблем.
    """
    name, what = FACET_LABELS[facet]
    found = "не найдено" if result.not_found else f"{len(result.hits)} слайд(ов)"
    if score is None:
        badge = f"предв. достоверность {_confidence_badge(retrieval_confidence(result.hits))}"
    elif score.confidence is None:
        badge = "в тексте не использована"
    else:
        badge = f"достоверность {_confidence_badge(score.confidence)}"
    with st.expander(f"{name} — {found} · {badge}"):
        st.markdown(f"**Что ищем:** {what}.")
        st.caption(f"Запрос к базе: «{query}»")
        if score is not None and score.claims:
            st.caption(f"Опора поиска {score.retrieval or 0:.2f} · утверждений в тексте "
                       f"{score.claims}, с проблемами {score.issues}")
        if not result.hits:
            st.write("В базе нет подходящих фрагментов: грань будет отмечена как пробел.")
        for hit in result.hits:
            c = hit.chunk
            title = c.text_raw.split("\n", 1)[0]
            st.markdown(f"- **{c.locator.capitalize()}** — {title}  \n"
                        f"  {_relevance(hit.rerank_score)} ({hit.rerank_score:.2f}) · "
                        f"{c.doc_title} · раздел «{c.section_title}»")


def _start_generation() -> None:
    """Колбэк кнопки: запрос фиксируется здесь, до перерисовки страницы."""
    ss = st.session_state
    goal = ss.get("goal", "").strip()
    if not goal:
        ss["goal_missing"] = True
        return
    ss["goal_missing"] = False
    ss["approach_run"] = None
    request("generate", "Генерация подхода", request=ApproachRequest(
        goal, ss.get("area", "").strip() or None, ss.get("situation", "").strip() or None
    ))


def _run_generation(job: dict[str, Any], mode: str, scope: repo.SearchScope) -> None:
    ss = st.session_state
    request = job["request"]
    queries = facet_queries(request)
    embedder, reranker = models()
    params = MODES_PARAMS[mode]
    limit = params.num_predict or settings().generation.gen_max_tokens

    st.subheader("Ход работы")
    stage = st.progress(0.0, text="Этап 1 из 3 · поиск по шести граням подхода")
    facet_box = st.container()
    with connection() as conn:
        done: list[str] = []

        def on_facet(facet: str, result: SearchResult) -> None:
            done.append(facet)
            with facet_box:
                render_facet(facet, result, queries[facet])
            stage.progress(0.3 * len(done) / len(queries),
                           text=f"Этап 1 из 3 · поиск: {len(done)} из {len(queries)} граней")

        search_started = time.perf_counter()
        facets = search_facets(conn, settings(), embedder, reranker, request, mode=mode,
                               scope=scope, progress=on_facet)
        search_seconds = time.perf_counter() - search_started

        stage.progress(0.35, text="Этап 2 из 3 · модель пишет черновик")
        live_caption = st.empty()
        live_text = st.empty()
        state = {"text": "", "tokens": 0, "started": time.perf_counter(), "shown": 0.0}

        def on_token(piece: str) -> None:
            state["text"] += piece
            state["tokens"] += 1
            now = time.perf_counter()
            if now - state["shown"] < 0.25:  # не перерисовывать страницу на каждый токен
                return
            state["shown"] = now
            elapsed = now - state["started"]
            speed = state["tokens"] / elapsed if elapsed else 0.0
            share = min(state["tokens"] / limit, 1.0)
            stage.progress(0.35 + 0.6 * share,
                           text=f"Этап 2 из 3 · пишет: {state['tokens']} токенов, "
                                f"{speed:.0f} ток/с, {elapsed:.0f} с")
            live_caption.caption("Текст появляется по мере генерации; ссылки и проверка — "
                                 "после завершения.")
            live_text.markdown(state["text"] + " ▌")

        run = generate_approach(conn, settings(), embedder, reranker, request, mode=mode,
                                scope=scope, facets=facets, on_token=on_token)
    stage.progress(1.0, text="Этап 3 из 3 · проверка завершена")
    run.search_seconds = search_seconds
    ss["approach_run"] = run
    ss["approach_queries"] = queries
    if run.section is not None:
        ss["approach_path"] = save_draft(run, DRAFTS)


def tab_approach(mode: str, scope: repo.SearchScope) -> None:
    st.header("Подход к новому проекту")
    st.caption("Поиск по 6 граням подхода в прошлых проектах → черновик с источниками. "
               "Всё, что модель не подтвердила фрагментами, попадает в блок «Не подтверждено».")
    busy = locked()
    with st.form("approach_form"):
        # Поля не блокируются: смена disabled пересоздаёт виджет и стирает ввод.
        st.text_input("Цель нового проекта *", key="goal",
                      placeholder="Разработка стратегии цифровой трансформации")
        st.text_input("Область / отрасль", key="area",
                      placeholder="коммерческая недвижимость")
        st.text_area("Ситуация заказчика", key="situation", height=100,
                     placeholder="нет ИТ-стратегии, учёт в Excel и 1С, "
                                 "руководство сомневается в окупаемости")
        st.form_submit_button("Сгенерировать", type="primary", disabled=busy,
                              on_click=_start_generation)
    status_line()
    if st.session_state.get("goal_missing"):
        st.warning("Укажите цель проекта.")
    execute("generate", lambda job: _run_generation(job, mode, scope))

    run = st.session_state.get("approach_run")
    if run is None:
        return
    st.success(f"Готово: поиск {run.search_seconds:.0f} с, генерация "
               f"{run.generation_seconds:.0f} с, {run.completion_tokens or 0} токенов")
    queries = st.session_state.get("approach_queries", {})
    scores = run.section.scores() if run.section is not None else {}
    st.subheader("Достоверность по граням")
    st.caption("Эвристика, а не вероятность: средняя оценка реранкера фрагментов грани × "
               "доля утверждений грани, прошедших проверки (числа, ссылки, источник, перенос).")
    columns = st.columns(len(FACET_LABELS))
    for column, facet in zip(columns, FACET_LABELS, strict=True):
        score = scores.get(facet)
        column.metric(FACET_LABELS[facet][0],
                      _confidence_badge(score.confidence if score else None),
                      help=f"проблем: {score.issues}" if score else None)
    st.subheader("Что нашлось по граням подхода")
    for facet, result in run.facets.items():
        render_facet(facet, result, queries.get(facet, ""), scores.get(facet))
    if run.truncated:
        st.warning("Ответ обрезан по лимиту длины — последний раздел может быть неполным.")
    if run.section is not None and run.section.unverified:
        st.info(f"Не подтверждено: {len(run.section.unverified)} — см. блок в конце черновика.")
    st.subheader("Черновик")
    st.markdown(run.markdown)
    path = st.session_state.get("approach_path")
    if path:
        st.download_button("Скачать .md", Path(path).read_bytes(), file_name=Path(path).name,
                           key="download_approach")
        st.caption(f"Сохранено: {Path(path).relative_to(ROOT)}")


def tab_search(scope: repo.SearchScope) -> None:
    st.header("Поиск по базе")
    query = st.text_input("Вопрос", key="search_query",
                          placeholder="сколько интервью провели при оценке зрелости")
    k = st.slider("Сколько фрагментов", 1, 15, 5)
    if not query.strip():
        return
    embedder, reranker = models()
    with st.spinner("Поиск…"), connection() as conn:
        result = search(conn, settings(), embedder, reranker, query.strip(), top_k=k, scope=scope)
    st.caption(f"Леммы полнотекстового поиска: {', '.join(result.terms) or '—'}")
    if result.not_found:
        st.warning("Ничего не найдено.")
        return
    for n, hit in enumerate(result.hits, 1):
        c = hit.chunk
        title = c.text_raw.split("\n", 1)[0]
        with st.expander(f"[{n}] {c.locator} · {title[:110]}", expanded=n == 1):
            st.caption(f"{c.doc_title} · {hit.customer} · раздел: {c.section_title} · "
                       f"реранкер {hit.rerank_score:.3f} · "
                       f"списки: {', '.join(f'{k}#{v}' for k, v in hit.lists.items())}")
            st.markdown(c.text_raw)


def _save_upload(name: str, data: bytes, directory: Path) -> Path:
    """Сохранить загруженный файл, не перезаписывая существующий с другим содержимым."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / file_key(name)
    if target.exists() and target.read_bytes() != data:
        stem, suffix = target.stem, target.suffix
        n = 2
        while (directory / f"{stem}_{n}{suffix}").exists():
            n += 1
        target = directory / f"{stem}_{n}{suffix}"
    target.write_bytes(data)
    return target


PUBLICITY_LABELS = {
    "open": "open — заказчика можно называть",
    "anon": "anon — только обезличенное название",
    "closed": "closed — в ответах не использовать",
}
TENDER_LABELS = {"won": "выиграно", "lost": "проиграно", "not_submitted": "не подавалось"}


def review_form(busy: bool) -> None:
    """Проверка строки реестра на странице: решения, которые принимает человек.

    Модель заполняет описание проекта; код, публичность, тип документа,
    результат тендера и отметку «проверено» ставит человек — здесь, без Excel.
    """
    problems = diagnose(REGISTRY, DATA)
    waiting = [code for code, issues in problems.items() if issues]
    st.subheader("Проверка строки реестра")
    message = st.session_state.pop("review_message", None)
    if message:
        (st.success if message[0] == "ok" else st.warning)(message[1])
    if not waiting:
        st.caption("Все строки реестра проверены и готовы к индексации.")
        return
    code = st.selectbox("Проект", waiting, key="review_project")
    row = project_row(REGISTRY, code)
    st.markdown("**Что мешает индексации:**\n" + "\n".join(f"- {p}" for p in problems[code]))
    with st.expander("Что заполнила модель (полный просмотр и правка — в Excel)"):
        shown = ("Название", "Клиент", "Год", "Основная отрасль", "Описание проекта", "Подход",
                 "Key findings", "Курирующий партнер", "Менеджер проекта")
        st.markdown("\n".join(f"- **{k}:** {row.get(k)}" for k in shown if row.get(k)))
    documents = row.get("_documents", [])
    current_type = next((d.get("Тип документа") for d in documents if d.get("Тип документа")),
                        "report")
    with st.form("review"):
        suggested = code.removesuffix("-DRAFT") + ("-01" if code.endswith("-DRAFT") else "")
        new_code = st.text_input("Код проекта", value=suggested,
                                 help="Код из учёта компании; временный «…-DRAFT» не подходит.")
        publicity = st.radio("Уровень публичности", list(PUBLICITY_LABELS),
                             format_func=PUBLICITY_LABELS.get,
                             index=list(PUBLICITY_LABELS).index(row["Уровень публичности"])
                             if row.get("Уровень публичности") in PUBLICITY_LABELS else 0)
        anon_label = st.text_input("Обезличенное название заказчика",
                                   value=row.get("Обзелич.название клиента") or "",
                                   help="Обязательно для anon.")
        doc_type = st.selectbox("Тип документа", list(DOC_TYPE_LABELS),
                                format_func=DOC_TYPE_LABELS.get,
                                index=list(DOC_TYPE_LABELS).index(current_type)
                                if current_type in DOC_TYPE_LABELS else 0)
        tender = st.selectbox("Результат тендера (для КП)", ["—", *TENDER_LABELS],
                              format_func=lambda v: TENDER_LABELS.get(v, "—"),
                              index=(["—", *TENDER_LABELS].index(row["Результат тендера"])
                                     if row.get("Результат тендера") in TENDER_LABELS else 0),
                              help="Для отчёта и акта ставится «выиграно» автоматически.")
        verified = st.checkbox("Строка проверена — поставить «проверено»", value=True)
        submitted = st.form_submit_button("Сохранить в реестр", disabled=busy)
    if submitted:
        tender_value = "won" if doc_type in ("report", "act") else (
            None if tender == "—" else tender)
        try:
            review_project(REGISTRY, code, new_code=new_code.strip(), publicity=publicity,
                           anon_label=anon_label.strip() or None, doc_type=doc_type,
                           tender_result=tender_value, verified=verified)
        except RegistryError as exc:
            st.error(str(exc))
            return
        left = diagnose(REGISTRY, DATA).get(new_code.strip(), [])
        st.session_state["review_message"] = (
            ("warn", "Сохранено, но для индексации ещё нужно: " + "; ".join(left)) if left
            else ("ok", f"Строка {new_code.strip()} проверена — нажмите «Проиндексировать»."))
        st.rerun()


def tab_documents() -> None:
    st.header("Документы")
    st.caption("Здесь только то, что уже в формате системы: md (сконвертированные документы) "
               "и xlsx (реестр проектов). Презентации и Word загружаются на вкладке "
               "«Конвертация» — там они превращаются в md.")
    with connection() as conn:
        documents = repo.list_documents(conn, repo.active_index_version(conn))
    in_db = {file_key(d.file_path): d for d in documents}

    uploaded = st.file_uploader("Загрузить md или xlsx", type=["md", "xlsx"],
                                accept_multiple_files=True, key="documents_upload")
    for file in uploaded or []:
        data = file.getvalue()
        if file.name.lower().endswith(".xlsx"):
            temporary = _save_upload(file.name, data, DATA / "cache" / "uploads")
            if is_registry(temporary):
                if REGISTRY.exists():
                    backup = DATA / f"registry_backup_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
                    REGISTRY.replace(backup)
                    st.info(f"Прежний реестр сохранён как {backup.name}")
                temporary.replace(REGISTRY)
                st.success("Реестр проектов заменён загруженным файлом.")
            else:
                saved = _save_upload(file.name, data, SAMPLES)
                temporary.unlink(missing_ok=True)
                st.success(f"Таблица сохранена: {saved.name} (в поиск не индексируется)")
        else:
            saved = _save_upload(file.name, data, SAMPLES)
            st.success(f"Сохранён {saved.relative_to(ROOT)}")
    registry = registered_files(REGISTRY) if REGISTRY.exists() else {}
    problems = diagnose(REGISTRY, DATA) if REGISTRY.exists() else {}

    st.subheader("Сохранённые документы")
    rows = []
    for path in sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in (".md", ".xlsx")):
        if path.suffix.lower() == ".xlsx":
            rows.append({"Файл": path.name, "Проект": "—", "Состояние": "📊 таблица"})
            continue
        key = file_key(path.relative_to(DATA).as_posix())
        code, status = registry.get(key, (None, None))
        doc = in_db.get(key)
        if doc and doc.chunks:
            state = f"✅ в поиске ({doc.chunks} фрагм.)"
        elif code and problems.get(code):
            state = f"⛔ не готов к индексации: {len(problems[code])} — см. «Проверка строки»"
        elif status == "проверено":
            state = "🟡 проверен в реестре — нужно проиндексировать"
        else:
            state = "🆕 нет в реестре"
        rows.append({"Файл": path.name, "Проект": code or "—", "Состояние": state})
    if not rows:
        st.info("Документов пока нет — загрузите презентацию или Word на вкладке «Конвертация».")
    else:
        st.dataframe(rows, width="stretch", hide_index=True)

    new = [r["Файл"] for r in rows if r["Проект"] == "—" and r["Файл"].endswith(".md")]
    busy = locked()
    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.selectbox("Новый документ", new, disabled=not new or busy,
                              placeholder="нет новых документов", index=None)
        st.button("Черновик в реестр (GigaChat, 1–2 мин)", disabled=not target or busy,
                  on_click=request, args=("draft", "Черновик строки реестра"),
                  kwargs={"file": target})
    with col2:
        st.write("")
        st.write("")
        if st.button("Открыть реестр в Excel", disabled=busy):
            subprocess.run(["open", str(REGISTRY)], check=False)
            st.caption("Проверьте жёлтые и оранжевые ячейки, сохраните и закройте файл. "
                       "Основные решения можно принять и ниже, в форме проверки.")
    with col3:
        st.write("")
        st.write("")
        st.button("Проиндексировать", type="primary", disabled=busy,
                  on_click=request, args=("index", "Индексация"))
    status_line()
    show_command_log()
    execute("draft", lambda job: run_command(
        [*DRAFT, str(SAMPLES / job["file"])], "Черновик строки реестра"))
    execute("index", lambda job: run_command(INGEST, "Загрузка реестра и индексация"))

    if REGISTRY.exists():
        review_form(busy)

    if documents:
        st.subheader("В поиске")
        st.dataframe(
            [{"Документ": d.title, "Тип": d.doc_type, "Заказчик": d.customer,
              "Фрагментов": d.chunks, "Статус": d.status} for d in documents],
            width="stretch", hide_index=True,
        )




def show_check(result: DeckCheck, markdown: str, needs_vlm: int) -> None:
    """Отчёт проверки md против pptx: итог, проблемные слайды, их md."""
    passed = "✅ пройдена" if result.passed else "❌ не пройдена"
    st.markdown(f"**Проверка {passed}** (полнота ≥ {COVERAGE_MIN:.0%}, ноль чисел без опоры)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Полнота текста", f"{result.coverage:.1%}")
    c2.metric("Чисел без опоры в XML", result.unsupported)
    c3.metric("Нужен разбор изображений", f"{needs_vlm} слайд.",
               help="Картинки с содержимым, значения цветом, подписи диаграмм think-cell. "
                    "Логотипы и фоновые рисунки шаблона не считаются.")
    for problem in result.problems:
        st.error(problem)
    problem_slides = [s for s in result.slides if s.unsupported or s.coverage < 0.9]
    if not problem_slides:
        return
    st.caption("Слайды, которые стоит посмотреть: низкая полнота или числа, которых нет в XML "
               "(у md, сделанного моделью, это могут быть её добавления — ссылки, пересказ).")
    blocks = dict(re.findall(r"<!-- slide: (\d+) -->(.*?)(?=<!-- slide: \d+ -->|\Z)",
                             markdown, re.S))
    for slide in problem_slides[:30]:
        label = f"Слайд {slide.slide}: полнота {slide.coverage:.0%}"
        if slide.unsupported:
            label += f", без опоры: {', '.join(slide.unsupported[:4])}"
        with st.expander(label):
            if slide.missing_paragraphs:
                st.markdown("**Нет в md:**\n" + "\n".join(
                    f"- {p[:200]}" for p in slide.missing_paragraphs[:8]))
            st.code(blocks.get(str(slide.slide), "(нет в md)").strip()[:3000], language="markdown")


DOC_TYPE_LABELS = {"report": "Отчёт", "kp": "КП", "act": "Акт", "presentation": "Презентация",
                   "contract": "Договор", "letter": "Письмо"}
NEW_PROJECT = "Новый проект — черновик строки реестра (GigaChat)"


def _verified_projects() -> dict[str, str]:
    """Проверенные проекты реестра: подпись в списке -> код."""
    if not REGISTRY.exists():
        return {}
    report = load_registry(REGISTRY, DATA)
    return {f"{p.code} — {p.name[:70]}": p.code for p in report.projects}


def tab_conversion() -> None:
    st.header("Загрузка документа")
    st.caption("Загрузите презентацию, Word или md. Формат определится сам; документ "
               "превратится в md (текст, таблицы, данные и подписи диаграмм, тепловые карты — "
               "из файла; картинки разбирает модель) и сохранится в data/samples. Оригинал "
               "хранится в data/sources.")
    upload = st.file_uploader("Документ", type=["pptx", "docx", "md"], key="convert_upload")
    projects = _verified_projects()
    project = st.selectbox("Проект", [NEW_PROJECT, *projects],
                           help="Существующий проверенный проект — документ сразу добавится "
                                "в реестр. Новый — GigaChat заполнит черновик строки, его нужно "
                                "проверить в Excel.")
    doc_type = st.selectbox("Тип документа", list(DOC_TYPE_LABELS),
                            format_func=DOC_TYPE_LABELS.get, disabled=project == NEW_PROJECT,
                            help="Для нового проекта тип определит GigaChat.")
    index_now = st.checkbox("Сразу проиндексировать", value=True)

    busy = locked()
    st.button("Конвертировать", type="primary", disabled=upload is None or busy,
              on_click=_start_conversion, args=(project, projects, doc_type, index_now))
    status_line()
    show_command_log()
    execute("convert", lambda job: _run_conversion(job["name"], job["data"], project, projects,
                                                    doc_type, index_now))

    state = st.session_state.get("conversion")
    if state:
        _show_conversion(*state)


def _start_conversion(project: str, projects: dict[str, str], doc_type: str,
                      index_now: bool) -> None:
    upload = st.session_state.get("convert_upload")
    if upload is None:
        return
    request("convert", f"Конвертация «{upload.name}»", name=upload.name, data=upload.getvalue())


def _run_conversion(name: str, data: bytes, project: str, projects: dict[str, str],
                    doc_type: str, index_now: bool) -> None:
    is_md = name.lower().endswith(".md")
    source = _save_upload(name, data, SAMPLES if is_md else SOURCES)
    target = SAMPLES / f"{source.stem}.md"
    if not is_md and target.exists():
        st.error(f"В data/samples уже есть {target.name}. Удалите его или переименуйте "
                 "загружаемый файл, чтобы не держать в поиске две версии документа.")
        return

    bar = st.progress(0.0, text="Разбор файла…")

    def on_slide(done: int, total: int, slide: int) -> None:
        what = "рисунок" if name.lower().endswith(".docx") else "слайд"
        bar.progress(done / total, text=f"Модель разбирает картинки: {what} {slide} "
                                         f"({done} из {total})")

    try:
        result = convert(source, vlm=True, cache_dir=DATA / "cache", progress=on_slide)
    except RenderError as exc:
        st.warning(f"Картинки не разобраны ({exc}); документ сконвертирован без модели.")
        result = convert(source)
    bar.progress(1.0, text="Готово")
    if result.markdown is None:
        st.error(f"Не удалось сконвертировать: {result.error}")
        return
    if result.check is not None and not result.check.passed:
        st.error("Проверка против исходного файла не пройдена — md не сохранён, чтобы неверные "
                 "данные не попали в поиск.")
        st.session_state["conversion"] = (result, None, [])
        return
    if not is_md:
        target.write_text(result.markdown, encoding="utf-8")

    steps: list[str] = [f"md сохранён: {target.relative_to(ROOT)}"]
    relative = file_key(target.relative_to(DATA).as_posix())
    if project == NEW_PROJECT:
        run_command([*DRAFT, str(target)], "Черновик строки реестра")
        steps.append("черновик строки реестра создан — проверьте его в Excel (вкладка "
                     "«Документы» → «Открыть реестр»); в поиск документ попадёт после отметки "
                     "«проверено» и индексации")
    else:
        meta, _ = _split_frontmatter(result.markdown)
        try:
            add_document(REGISTRY, projects[project], DocumentRow(
                file=relative,
                source_file=None if is_md else file_key(source.relative_to(DATA).as_posix()),
                doc_type=doc_type, title=str(meta.get("doc_title") or target.stem),
                doc_date=None, slides=meta.get("slides_total"),
            ))
            steps.append(f"документ добавлен к проекту {projects[project]}")
            if index_now:
                run_command(INGEST, "Индексация")
                steps.append("индексация запущена — результат выше")
        except RegistryError as exc:
            steps.append(f"в реестр не добавлен: {exc}")
    st.session_state["conversion"] = (result, str(target), steps)


def _show_conversion(result: Any, target: str | None, steps: list[str]) -> None:
    st.markdown(f"**Определён как:** {result.detection.format} ({result.detection.reason}). "
                f"**Разбор:** {result.detection.method}. {result.notes}")
    for step in steps:
        st.markdown(f"- {step}")
    if result.vlm_issues:
        with st.expander(f"Замечания к ответам модели: {len(result.vlm_issues)}"):
            st.markdown("\n".join(f"- {issue}" for issue in result.vlm_issues))
    if result.check is not None:
        show_check(result.check, result.markdown or "", result.needs_vlm)
    if result.markdown:
        st.download_button("Скачать md", result.markdown.encode("utf-8"),
                           file_name=Path(target).name if target else "document.md",
                           key="download_converted")
        with st.expander("Весь md"):
            st.code(result.markdown[:20000], language="markdown")


def tab_drafts() -> None:
    st.header("Черновики")
    files = sorted(DRAFTS.glob("*.md"), reverse=True) if DRAFTS.exists() else []
    if not files:
        st.info("Черновиков пока нет — сгенерируйте подход на первой вкладке.")
        return
    chosen = st.selectbox("Черновик", files, format_func=lambda p: p.name)
    text = chosen.read_text(encoding="utf-8")
    header = text.split("\n", 1)[0]
    if header.startswith("<!--"):
        st.caption(header.strip("<!-> "))
    st.download_button("Скачать .md", text.encode("utf-8"), file_name=chosen.name,
                       key="download_draft")
    st.markdown(text)


def main() -> None:
    mode, scope = sidebar()
    approach, search_tab, conversion, documents, drafts = st.tabs(
        ["🧭 Подход", "🔎 Поиск", "🔄 Конвертация", "📁 Документы", "📝 Черновики"]
    )
    with approach:
        tab_approach(mode, scope)
    with search_tab:
        tab_search(scope)
    with conversion:
        tab_conversion()
    with documents:
        tab_documents()
    with drafts:
        tab_drafts()


main()
