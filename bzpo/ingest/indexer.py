"""Индексация документов реестра: md -> чанки -> векторы -> chunks (SPEC 4).

Версия индекса определяется моделью эмбеддера, её ревизией и параметрами
чанкинга: смена любого из них даёт новую версию, старая остаётся активной до
конца сборки новой (SPEC 4.4). Документ, чей файл не изменился и уже есть в
версии, пропускается — повторный прогон ничего не пересчитывает.

На каждый чанк два вектора: `embedding` по text_enriched (поиск) и
`embedding_raw` по text_raw (дедупликация, MMR) — см. миграцию 0001.

Публичность доходит до текста чанка: в шапку text_enriched идёт имя заказчика
для open и обезличенное название для anon (`customer_label`).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bzpo.config import Settings
from bzpo.ingest.changes import registry_fingerprint
from bzpo.ingest.document import chunk_document
from bzpo.ingest.slides import chunk_slides, is_slide_markdown
from bzpo.parsing.convert import convert
from bzpo.providers.base import Embedder
from bzpo.storage import repo

logger = logging.getLogger(__name__)

PARSER_VERSION = "slides-md-1"
# Шапка чанка собирается детерминированно из frontmatter и реестра, без LLM.
ENRICH_PROMPT_VERSION = "header-from-registry-1"


@dataclass(slots=True)
class IndexStats:
    version: str
    indexed: list[str]
    skipped: list[str]
    failed: dict[str, str]
    chunks: int
    seconds: float


def chunk_params(settings: Settings) -> dict[str, Any]:
    """Всё, что влияет на нарезку: и для слайдов, и для документов (docx, md)."""
    chunking = settings.chunking
    return {"slides": "one_per_slide", "hard_limit": chunking.chunk_hard_limit_embed,
            "document_target": chunking.chunk_target, "document_min": chunking.chunk_min}


def index_version_name(settings: Settings) -> str:
    local = settings.local
    model = local.embed_model.split("/")[-1]
    params = "|".join([PARSER_VERSION, ENRICH_PROMPT_VERSION,
                       *(f"{k}={v}" for k, v in sorted(chunk_params(settings).items()))])
    digest = hashlib.sha256(params.encode()).hexdigest()[:8]
    return f"{model}@{local.embed_model_revision[:8]}-{digest}"


def index_documents(
    conn: Any,
    settings: Settings,
    report: Any,
    embedder: Embedder,
    count_tokens: Callable[[str], int],
    *,
    data_root_key: Callable[[Any], str],
) -> IndexStats:
    started = time.perf_counter()
    version = index_version_name(settings)
    repo.ensure_index_version(
        conn, version=version, embedder_model=settings.local.embed_model,
        embedder_revision=settings.local.embed_model_revision, parser_version=PARSER_VERSION,
        chunk_params=chunk_params(settings),
        enrich_prompt_version=ENRICH_PROMPT_VERSION,
    )
    conn.commit()
    projects = {p.code: p for p in report.projects}
    stats = IndexStats(version, [], [], {}, 0, 0.0)

    for document in report.documents:
        key = data_root_key(document.file)
        raw = document.file.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        # Формат определяется по содержимому: md индексируется как есть, pptx и
        # docx конвертируются на лету тем же конвертером, что и в интерфейсе.
        # Ошибка разбора одного файла не должна останавливать весь прогон.
        try:
            conversion = convert(document.file)
        except Exception as exc:  # noqa: BLE001 - файл помечается, прогон идёт дальше
            stats.failed[key] = f"конвертация: {type(exc).__name__}: {exc}"
            logger.exception("конвертация %s не удалась", key)
            continue
        if conversion.markdown is None:
            stats.failed[key] = conversion.error or "не удалось получить md"
            continue
        if conversion.check is not None and not conversion.check.passed:
            # Проверка против XML не пройдена: числа без опоры или потерян текст —
            # такой md не должен попадать в поиск и в черновики молча.
            stats.failed[key] = (
                f"проверка конвертации не пройдена: полнота {conversion.check.coverage:.1%}, "
                f"чисел без опоры {conversion.check.unsupported}"
            )
            continue
        text = conversion.markdown

        project = projects[document.project_code]
        fingerprint = registry_fingerprint(project, document)
        doc_id, (previous_hash, previous_registry) = repo.upsert_document(
            conn, project_code=document.project_code, file_key=key, title=document.title,
            doc_type=document.doc_type, doc_date=document.doc_date, file_hash=file_hash,
            page_count=None, parser_version=PARSER_VERSION,
        )
        conn.commit()
        # Пропуск — только если не изменились ни md, ни строки реестра: заказчик и
        # название реестра входят в шапки фрагментов.
        if (previous_hash == file_hash and previous_registry == fingerprint
                and repo.document_indexed(conn, doc_id, version)):
            stats.skipped.append(key)
            continue

        label = project.anon_label if project.publicity == "anon" else project.client
        try:
            if is_slide_markdown(text):
                deck = chunk_slides(
                    text, count_tokens, hard_limit=settings.chunking.chunk_hard_limit_embed,
                    customer_label=label,
                )
            else:
                deck = chunk_document(
                    text, count_tokens, target=settings.chunking.chunk_target,
                    minimum=settings.chunking.chunk_min,
                    hard_limit=settings.chunking.chunk_hard_limit_embed, customer_label=label,
                )
            enriched = embedder.embed([c.text_enriched for c in deck.chunks])
            plain = embedder.embed([c.text_raw for c in deck.chunks])
            rows = [
                repo.ChunkRow(
                    ord=c.slide, section_title=c.section, subsection_title=c.subsection,
                    page=c.page, text_raw=c.text_raw, text_enriched=c.text_enriched,
                    content_hash=hashlib.sha256(c.text_raw.encode()).hexdigest(),
                    embedding=e, embedding_raw=r, token_count=c.token_count, is_table=c.is_table,
                )
                for c, e, r in zip(deck.chunks, enriched, plain, strict=True)
            ]
            repo.replace_chunks(
                conn, doc_id=doc_id, project_code=document.project_code, version=version,
                chunks=rows,
            )
            repo.mark_indexed(conn, doc_id, file_hash, fingerprint)
            conn.commit()
        except Exception as exc:  # noqa: BLE001 - документ помечается, прогон идёт дальше
            conn.rollback()
            repo.set_document_status(conn, doc_id, "failed", f"{type(exc).__name__}: {exc}")
            conn.commit()
            stats.failed[key] = str(exc)
            logger.exception("индексация %s не удалась", key)
            continue
        stats.indexed.append(key)
        stats.chunks += len(rows)
        logger.info("%s: %d чанков, предел документа %d ток.", key, len(rows), deck.max_tokens)

    if not stats.failed:
        repo.activate_index_version(conn, version)
        conn.commit()
    stats.seconds = time.perf_counter() - started
    return stats
