"""Веб-интерфейс БЗПО (Streamlit).

    ./start.sh                      # или: streamlit run bzpo/ui/app.py

Вкладки: генерация подхода, поиск, документы (реестр и индексация), черновики.
Модели эмбеддинга и реранкинга загружаются один раз на процесс
(st.cache_resource), поэтому повторные запросы идут без загрузки весов.
Длинные операции реестра и индексации запускаются теми же командами, что и из
терминала (subprocess), — у интерфейса нет своей версии этой логики.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

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
from bzpo.parsing.check import COVERAGE_MIN, DeckCheck, check
from bzpo.parsing.convert import convert, sources
from bzpo.parsing.detect import detect
from bzpo.providers.local import LocalEmbedder, LocalReranker
from bzpo.providers.ollama import OllamaClient
from bzpo.registry.workbook import file_key, registered_files
from bzpo.search.query import SearchResult, search
from bzpo.storage import repo

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SAMPLES = DATA / "samples"
DRAFTS = DATA / "drafts"
DRAFTS_CONVERTED = DATA / "converted"
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
    ss["pending_request"] = ApproachRequest(
        goal, ss.get("area", "").strip() or None, ss.get("situation", "").strip() or None
    )
    ss["busy"] = True
    ss["approach_run"] = None


def _run_generation(mode: str, scope: repo.SearchScope) -> None:
    ss = st.session_state
    request = ss["pending_request"]
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
    busy = st.session_state.get("busy", False)
    with st.form("approach_form"):
        # Поля не блокируются: смена disabled пересоздаёт виджет и стирает ввод.
        st.text_input("Цель нового проекта *", key="goal",
                      placeholder="Разработка стратегии цифровой трансформации")
        st.text_input("Область / отрасль", key="area",
                      placeholder="коммерческая недвижимость")
        st.text_area("Ситуация заказчика", key="situation", height=100,
                     placeholder="нет ИТ-стратегии, учёт в Excel и 1С, "
                                 "руководство сомневается в окупаемости")
        st.form_submit_button("⏳ Генерация идёт…" if busy else "Сгенерировать",
                              type="primary", disabled=busy, on_click=_start_generation)
    if st.session_state.get("goal_missing"):
        st.warning("Укажите цель проекта.")

    if busy:
        try:
            _run_generation(mode, scope)
        finally:
            st.session_state["busy"] = False
        st.rerun()

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


def tab_documents() -> None:
    st.header("Документы")
    with connection() as conn:
        documents = repo.list_documents(conn, repo.active_index_version(conn))
    in_db = {file_key(d.file_path): d for d in documents}
    registry = registered_files(REGISTRY) if REGISTRY.exists() else {}

    uploaded = st.file_uploader("Добавить файлы в data/samples (pptx, docx, md)",
                                type=["pptx", "docx", "md"], accept_multiple_files=True)
    for file in uploaded or []:
        target = SAMPLES / file.name
        if not target.exists():
            target.write_bytes(file.getvalue())
            st.success(f"Сохранён {target.relative_to(ROOT)}")

    st.subheader("Файлы в data/samples")
    rows = []
    # pptx/docx, у которого есть готовый md, представлен этим md — не дублируем.
    files = [f for f in sources(SAMPLES)
             if not (f.suffix.lower() in (".pptx", ".docx") and _md_for(f))]
    for md in files:
        detection = detect(md)
        if not detection.supported:
            rows.append({"Файл": md.name, "Формат": detection.format, "Проект": "—",
                         "Состояние": f"⛔ {detection.method}"})
            continue
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
        rows.append({"Файл": md.name, "Формат": detection.format, "Проект": code or "—",
                     "Состояние": state})
    st.dataframe(rows, width="stretch", hide_index=True)

    new = [r["Файл"] for r in rows if r["Проект"] == "—" and not r["Состояние"].startswith("⛔")]
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


def _name_key(path: Path) -> str:
    """Ключ для сопоставления pptx и md: «отчет v1.2» и «отчет_v1_2» — один документ."""
    return re.sub(r"[\W_]+", "", file_key(path.stem).lower())


def _md_for(source: Path) -> Path | None:
    """Готовый md для исходника (тот же документ, другое расширение)."""
    return next((md for md in SAMPLES.glob("*.md")
                 if md != source and _name_key(md) == _name_key(source)), None)


def _pptx_for(md: Path) -> Path | None:
    return next((p for p in SAMPLES.glob("*.pptx")
                 if not p.name.startswith("~$") and _name_key(p) == _name_key(md)), None)


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


def tab_conversion() -> None:
    st.header("Конвертация в md")
    st.caption("Формат определяется по содержимому файла, способ разбора выбирается сам: "
               "pptx — текст, таблицы и данные диаграмм из XML с проверкой; docx — заголовки, "
               "абзацы, таблицы, рисунки (похожие на скриншоты слайдов отмечаются); готовый md "
               "— проверка разметки. Картинки и значения цветом разберёт модель на этапе П2.")
    files = sources(SAMPLES)
    if not files:
        st.info("В data/samples нет файлов. Добавьте их на вкладке «Документы».")
        return
    detections = {f: detect(f) for f in files}
    st.dataframe(
        [{"Файл": f.name, "Определён как": d.format, "Как будет разобран": d.method,
          "Готовый md": (_md_for(f).name if d.format in ("pptx", "docx") and _md_for(f)
                         else "—")}
         for f, d in detections.items()],
        width="stretch", hide_index=True,
    )
    source = st.selectbox("Файл", files, format_func=lambda p: p.name)
    if st.button("Конвертировать", type="primary"):
        result = convert(source)
        target = None
        if result.markdown is not None and result.detection.format in ("pptx", "docx"):
            DRAFTS_CONVERTED.mkdir(parents=True, exist_ok=True)
            target = DRAFTS_CONVERTED / f"{source.stem}.md"
            target.write_text(result.markdown, encoding="utf-8")
        elif result.detection.format == "md_slides" and _pptx_for(source):
            # Готовый md презентации проверяется против её pptx, если он рядом.
            result.check = check(_pptx_for(source), result.markdown or "")
        st.session_state["conversion"] = (str(source), str(target) if target else None, result)

    state = st.session_state.get("conversion")
    if not state or state[0] != str(source):
        return
    _, target, result = state
    st.markdown(f"**Определён как:** {result.detection.format} ({result.detection.reason}). "
                f"**Разбор:** {result.detection.method}.")
    if result.markdown is None:
        st.error(result.error)
        return
    st.caption(result.notes)
    if result.needs_vlm:
        st.info(f"Разбор изображений (этап П2) понадобится: {result.needs_vlm} — до него "
                "содержимое картинок и тепловых карт в поиске не участвует.")
    if result.check is not None:
        show_check(result.check, result.markdown, result.needs_vlm)
    markdown = result.markdown
    if target:
        target_path = Path(target)
        st.download_button("Скачать md", markdown.encode("utf-8"), file_name=target_path.name,
                           key="download_converted")
        existing = _md_for(source)
        if existing:
            st.caption(f"В data/samples уже есть {existing.name}: чтобы не было двух версий "
                       "одного документа в индексе, сначала удалите или замените её вручную.")
        elif st.button("Принять в data/samples", disabled=not result.ok):
            accepted = SAMPLES / target_path.name
            accepted.write_text(markdown, encoding="utf-8")
            st.success(f"Сохранён {accepted.relative_to(ROOT)}. Дальше — вкладка «Документы»: "
                       "черновик в реестр и индексация.")
    with st.expander("Весь md"):
        st.code(markdown[:20000], language="markdown")


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
