"""Веб-интерфейс БЗПО (Streamlit).

    ./start.sh                      # или: streamlit run bzpo/ui/app.py

Вкладки: генерация подхода, поиск, документы (реестр и индексация), черновики.
Модели эмбеддинга и реранкинга загружаются один раз на процесс
(st.cache_resource), поэтому повторные запросы идут без загрузки весов.
Длинные операции реестра и индексации запускаются теми же командами, что и из
терминала (subprocess), — у интерфейса нет своей версии этой логики.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st

from bzpo.config import load_config
from bzpo.generation.approach import ApproachRequest
from bzpo.generation.run import generate_approach, save_draft
from bzpo.providers.local import LocalEmbedder, LocalReranker
from bzpo.providers.ollama import OllamaClient
from bzpo.registry.workbook import file_key, registered_files
from bzpo.search.query import search
from bzpo.storage import repo

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SAMPLES = DATA / "samples"
DRAFTS = DATA / "drafts"
REGISTRY = DATA / "registry.xlsx"
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


def run_command(args: list[str], title: str) -> None:
    """Запустить команду проекта и показать её вывод."""
    with st.status(title, expanded=True) as status:
        result = subprocess.run(
            [sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True,
        )
        lines = [line for line in (result.stdout + result.stderr).splitlines()
                 if "httpx" not in line and "Loading" not in line]
        st.code("\n".join(lines[-40:]) or "(нет вывода)", language=None)
        status.update(label=f"{title}: {'готово' if result.returncode == 0 else 'ошибка'}",
                      state="complete" if result.returncode == 0 else "error")


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


def tab_approach(mode: str, scope: repo.SearchScope) -> None:
    st.header("Подход к новому проекту")
    st.caption("Поиск по 6 граням подхода в прошлых проектах → черновик с источниками. "
               "Всё, что модель не подтвердила фрагментами, попадает в блок «Не подтверждено».")
    with st.form("approach"):
        goal = st.text_input("Цель нового проекта *",
                             placeholder="Разработка стратегии цифровой трансформации")
        area = st.text_input("Область / отрасль", placeholder="коммерческая недвижимость")
        situation = st.text_area(
            "Ситуация заказчика", height=100,
            placeholder="нет ИТ-стратегии, учёт в Excel и 1С, "
                        "руководство сомневается в окупаемости",
        )
        submitted = st.form_submit_button("Сгенерировать", type="primary")
    if submitted:
        if not goal.strip():
            st.warning("Укажите цель проекта.")
            return
        request = ApproachRequest(goal.strip(), area.strip() or None, situation.strip() or None)
        embedder, reranker = models()
        with st.status("Генерация…", expanded=True) as status, connection() as conn:
            run = generate_approach(
                conn, settings(), embedder, reranker, request, mode=mode, scope=scope,
                progress=lambda message: status.write(message),
            )
            status.update(label=f"Готово: поиск {run.search_seconds:.0f} с, "
                                f"генерация {run.generation_seconds:.0f} с", state="complete")
        st.session_state["approach_run"] = run
        if run.section is not None:
            st.session_state["approach_path"] = save_draft(run, DRAFTS)

    run = st.session_state.get("approach_run")
    if run is None:
        return
    if run.truncated:
        st.warning("Ответ обрезан по лимиту длины — попробуйте режим «Тщательно».")
    if run.section is not None and run.section.unverified:
        st.info(f"Не подтверждено: {len(run.section.unverified)} — см. блок в конце черновика.")
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
        with st.expander(f"[{n}] слайд {c.page} · {title[:110]}", expanded=n == 1):
            st.caption(f"{c.doc_title} · {hit.customer} · раздел: {c.section_title} · "
                       f"реранкер {hit.rerank_score:.3f} · "
                       f"списки: {', '.join(f'{k}#{v}' for k, v in hit.lists.items())}")
            st.markdown(c.text_raw)


def tab_documents() -> None:
    st.header("Документы")
    with connection() as conn:
        documents = repo.list_documents(conn, repo.active_index_version(conn))
    in_db = {file_key(d.file_path): d for d in documents}
    registry = registered_files(REGISTRY) if REGISTRY.exists() else {}

    uploaded = st.file_uploader("Добавить файлы в data/samples (md и исходный pptx)",
                                type=["md", "pptx"], accept_multiple_files=True)
    for file in uploaded or []:
        target = SAMPLES / file.name
        if not target.exists():
            target.write_bytes(file.getvalue())
            st.success(f"Сохранён {target.relative_to(ROOT)}")

    st.subheader("Файлы в data/samples")
    rows = []
    for md in sorted(SAMPLES.glob("*.md")):
        key = file_key(md.relative_to(DATA).as_posix())
        code, status = registry.get(key, (None, None))
        doc = in_db.get(key)
        if doc and doc.chunks:
            state = f"✅ в поиске ({doc.chunks} фрагм.)"
        elif status == "проверено":
            state = "🟡 проверен в реестре — нужно проиндексировать"
        elif code:
            state = "✏️ черновик в реестре — проверьте строку в Excel"
        else:
            state = "🆕 нет в реестре"
        rows.append({"Файл": md.name, "Проект": code or "—", "Состояние": state})
    st.dataframe(rows, width="stretch", hide_index=True)

    new = [r["Файл"] for r in rows if r["Проект"] == "—"]
    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.selectbox("Новый файл", new, disabled=not new,
                              placeholder="нет новых файлов", index=None)
        if st.button("Черновик в реестр (GigaChat, 1–2 мин)", disabled=not target):
            run_command(["bzpo.registry", "draft", f"data/samples/{target}"],
                        "Черновик строки реестра")
    with col2:
        st.write("")
        st.write("")
        if st.button("Открыть реестр в Excel"):
            subprocess.run(["open", str(REGISTRY)], check=False)
            st.caption("Проверьте жёлтые и оранжевые ячейки и поставьте «проверено».")
    with col3:
        st.write("")
        st.write("")
        if st.button("Проиндексировать", type="primary"):
            run_command(["bzpo.ingest"], "Загрузка реестра и индексация")
            st.cache_resource.clear()  # модели перезагрузятся при следующем запросе

    if documents:
        st.subheader("В базе")
        st.dataframe(
            [{"Документ": d.title, "Тип": d.doc_type, "Заказчик": d.customer,
              "Фрагментов": d.chunks, "Статус": d.status} for d in documents],
            width="stretch", hide_index=True,
        )


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
    approach, search_tab, documents, drafts = st.tabs(
        ["🧭 Подход", "🔎 Поиск", "📁 Документы", "📝 Черновики"]
    )
    with approach:
        tab_approach(mode, scope)
    with search_tab:
        tab_search(scope)
    with documents:
        tab_documents()
    with drafts:
        tab_drafts()


main()
