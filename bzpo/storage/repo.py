"""Репозитории PostgreSQL: реестр, документы, версии индекса, чанки, поиск.

Весь SQL системы живёт в storage/ (граница модулей, tests/test_boundaries.py).

Идентификаторы детерминированные (uuid5 от естественного ключа: код проекта,
путь файла, номер чанка в версии индекса): повторная загрузка реестра и
повторная индексация обновляют те же строки, а не плодят дубли.

Векторы передаются строкой '[…]' с приведением ::vector — отдельный адаптер
pgvector для psycopg не нужен.

Каждый поисковый запрос содержит фильтр версии индекса, канонических чанков и
публичности проекта (SPEC 5.2): без него закрытый проект попадает в выдачу.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any

_NAMESPACE = uuid.UUID("6f1c2a0e-5b7d-4c43-9a51-2d1b7e0f9c11")


def stable_id(kind: str, key: str) -> uuid.UUID:
    return uuid.uuid5(_NAMESPACE, f"{kind}:{key}")


@contextmanager
def connect(dsn: str, statement_timeout_ms: int = 30_000) -> Iterator[Any]:
    import psycopg  # локальный импорт: драйвер нужен только здесь

    with psycopg.connect(dsn) as conn:
        conn.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")
        yield conn


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


# ----------------------------------------------------------------- реестр


def save_registry(conn: Any, report: Any) -> uuid.UUID:
    """Записать проекты из LoadReport и отчёт загрузки; вернуть upload_id."""
    with conn.transaction():
        for project in report.projects:
            client_id = stable_id("client", project.client)
            aliases = [a for a in (project.client_contract,) if a and a != project.client]
            conn.execute(
                "INSERT INTO clients (client_id, canonical_name, aliases) VALUES (%s, %s, %s) "
                "ON CONFLICT (client_id) DO UPDATE SET aliases = (SELECT array_agg(DISTINCT a) "
                "FROM unnest(clients.aliases || EXCLUDED.aliases) a)",
                (client_id, project.client, aliases),
            )
            conn.execute(
                """
                INSERT INTO projects (project_id, name, client_id, industry, publicity, anon_label,
                                      tender_result, date_start, date_end, contract_no, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')
                ON CONFLICT (project_id) DO UPDATE SET
                  name = EXCLUDED.name, client_id = EXCLUDED.client_id,
                  industry = EXCLUDED.industry, publicity = EXCLUDED.publicity,
                  anon_label = EXCLUDED.anon_label, tender_result = EXCLUDED.tender_result,
                  date_start = EXCLUDED.date_start, date_end = EXCLUDED.date_end,
                  contract_no = EXCLUDED.contract_no
                """,
                (stable_id("project", project.code), project.name, client_id, project.industry,
                 project.publicity, project.anon_label, project.tender_result,
                 project.date_start, project.date_end, project.contract_no),
            )
        upload_id = uuid.uuid4()
        conn.execute(
            "INSERT INTO registry_uploads (upload_id, source_file, row_count, rejected_count, "
            "report) VALUES (%s, %s, %s, %s, %s)",
            (upload_id, report.source_file, report.row_count, len(report.rejected),
             json.dumps(report.as_json(), ensure_ascii=False)),
        )
    return upload_id


# --------------------------------------------------------------- документы


def upsert_document(
    conn: Any, *, project_code: str, file_key: str, title: str, doc_type: str,
    doc_date: date | None, file_hash: str, page_count: int | None, parser_version: str,
) -> tuple[uuid.UUID, str | None]:
    """Создать или обновить документ; вернуть doc_id и прежний file_hash (None — новый)."""
    doc_id = stable_id("document", file_key)
    row = conn.execute("SELECT file_hash FROM documents WHERE doc_id = %s", (doc_id,)).fetchone()
    conn.execute(
        """
        INSERT INTO documents (doc_id, project_id, title, file_path, file_hash, doc_type,
                               doc_date, status, page_count, parser_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s)
        ON CONFLICT (doc_id) DO UPDATE SET
          project_id = EXCLUDED.project_id, title = EXCLUDED.title,
          file_hash = EXCLUDED.file_hash, doc_type = EXCLUDED.doc_type,
          doc_date = EXCLUDED.doc_date, page_count = EXCLUDED.page_count,
          parser_version = EXCLUDED.parser_version
        """,
        (doc_id, stable_id("project", project_code), title, file_key, file_hash, doc_type,
         doc_date, page_count, parser_version),
    )
    return doc_id, (row[0] if row else None)


def set_document_status(
    conn: Any, doc_id: uuid.UUID, status: str, error: str | None = None
) -> None:
    conn.execute(
        "UPDATE documents SET status = %s, last_error = %s, attempts = attempts + 1, "
        "parsed_at = CASE WHEN %s = 'indexed' THEN now() ELSE parsed_at END WHERE doc_id = %s",
        (status, error, status, doc_id),
    )


def document_indexed(conn: Any, doc_id: uuid.UUID, version: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM chunks WHERE doc_id = %s AND index_version = %s LIMIT 1", (doc_id, version)
    ).fetchone()
    return row is not None


# ----------------------------------------------------------- версии индекса


def ensure_index_version(
    conn: Any, *, version: str, embedder_model: str, embedder_revision: str,
    parser_version: str, chunk_params: dict[str, Any], enrich_prompt_version: str,
) -> None:
    conn.execute(
        """
        INSERT INTO index_versions (version, embedder_model, embedder_revision, parser_version,
                                    chunk_params, enrich_prompt_version, status)
        VALUES (%s, %s, %s, %s, %s, %s, 'building')
        ON CONFLICT (version) DO NOTHING
        """,
        (version, embedder_model, embedder_revision, parser_version,
         json.dumps(chunk_params), enrich_prompt_version),
    )


def activate_index_version(conn: Any, version: str) -> None:
    with conn.transaction():
        conn.execute(
            "UPDATE index_versions SET status = 'retired' "
            "WHERE status = 'active' AND version <> %s",
            (version,),
        )
        conn.execute("UPDATE index_versions SET status = 'active' WHERE version = %s", (version,))


def active_index_version(conn: Any) -> str | None:
    row = conn.execute(
        "SELECT version FROM index_versions WHERE status = 'active' "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


@dataclass(frozen=True, slots=True)
class ChunkRow:
    ord: int
    section_title: str | None
    subsection_title: str | None
    page: int | None
    text_raw: str
    text_enriched: str
    content_hash: str
    embedding: Sequence[float]
    embedding_raw: Sequence[float]
    token_count: int
    is_table: bool


def replace_chunks(
    conn: Any, *, doc_id: uuid.UUID, project_code: str, version: str, chunks: Sequence[ChunkRow]
) -> None:
    """Заменить чанки документа в версии индекса одной транзакцией."""
    project_id = stable_id("project", project_code)
    with conn.transaction():
        conn.execute(
            "DELETE FROM chunks WHERE doc_id = %s AND index_version = %s", (doc_id, version)
        )
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks (chunk_id, doc_id, project_id, ord, section_title,
                                    subsection_title, page_start, page_end, text_raw,
                                    text_enriched, content_hash, embedding, embedding_raw,
                                    token_count, is_table, index_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s::vector,
                        %s, %s, %s)
                """,
                [
                    (stable_id("chunk", f"{doc_id}:{version}:{c.ord}"), doc_id, project_id, c.ord,
                     c.section_title, c.subsection_title, c.page, c.page, c.text_raw,
                     c.text_enriched, c.content_hash, _vector(c.embedding),
                     _vector(c.embedding_raw), c.token_count, c.is_table, version)
                    for c in chunks
                ],
            )


# ------------------------------------------------------------------- поиск

_SEARCH_FILTER = """
  c.index_version = %(version)s AND c.is_canonical
  AND p.publicity = ANY(%(publicity)s)
  AND (%(doc_types)s::text[] IS NULL OR d.doc_type = ANY(%(doc_types)s::text[]))
  AND (%(doc_ids)s::uuid[] IS NULL OR c.doc_id = ANY(%(doc_ids)s::uuid[]))
"""
_SEARCH_JOIN = """
  chunks c JOIN projects p ON p.project_id = c.project_id
  JOIN documents d ON d.doc_id = c.doc_id
"""


@dataclass(frozen=True, slots=True)
class SearchScope:
    """Сужение выдачи пользователем: типы документов и конкретные документы (None — все)."""

    doc_types: Sequence[str] | None = None
    doc_ids: Sequence[str] | None = None

    def params(self) -> dict[str, Any]:
        return {
            "doc_types": list(self.doc_types) if self.doc_types else None,
            "doc_ids": list(self.doc_ids) if self.doc_ids else None,
        }


def vector_search(
    conn: Any, vector: Sequence[float], *, version: str, publicity: Sequence[str], k: int,
    scope: SearchScope | None = None,
) -> list[str]:
    rows = conn.execute(
        f"""
        SELECT c.chunk_id FROM {_SEARCH_JOIN}
        WHERE {_SEARCH_FILTER}
        ORDER BY c.embedding <=> %(vector)s::vector, c.chunk_id
        LIMIT %(k)s
        """,
        {"version": version, "publicity": list(publicity), "vector": _vector(vector), "k": k,
         **(scope or SearchScope()).params()},
    ).fetchall()
    return [str(r[0]) for r in rows]


def fts_search(
    conn: Any, terms: Sequence[str], *, version: str, publicity: Sequence[str], k: int,
    scope: SearchScope | None = None,
) -> list[str]:
    """Полнотекстовый поиск дизъюнкцией лемм (SPEC 5.2: `|`, не конъюнкция).

    `terms` — уже леммы (из `rarest_terms`), поэтому запрос строится в
    конфигурации simple: повторная русская морфология лемму не меняет, но
    может испортить. Леммы очищаются до букв и цифр — синтаксис tsquery в
    запрос не попадает.
    """
    words = [w for w in (re.sub(r"[^\w]", "", t) for t in terms) if len(w) >= 2]
    if not words:
        return []
    rows = conn.execute(
        f"""
        WITH q AS (SELECT to_tsquery('simple', %(query)s) AS q)
        SELECT c.chunk_id FROM {_SEARCH_JOIN}, q
        WHERE {_SEARCH_FILTER} AND c.tsv @@ q.q
        ORDER BY ts_rank_cd(c.tsv, q.q) DESC, c.chunk_id
        LIMIT %(k)s
        """,
        {"version": version, "publicity": list(publicity), "k": k,
         "query": " | ".join(words), **(scope or SearchScope()).params()},
    ).fetchall()
    return [str(r[0]) for r in rows]


def rarest_terms(conn: Any, text: str, *, version: str, limit: int = 8) -> list[str]:
    """Леммы запроса, отсортированные по редкости в корпусе (меньше df — раньше)."""
    rows = conn.execute(
        """
        WITH lex AS (
          SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('russian', %(text)s))) AS lexeme
        )
        SELECT lex.lexeme,
               (SELECT count(*) FROM chunks c
                 WHERE c.index_version = %(version)s
                   AND c.tsv @@ to_tsquery('simple', quote_literal(lex.lexeme))) AS df
        FROM lex
        ORDER BY df ASC, lex.lexeme
        """,
        {"text": text, "version": version},
    ).fetchall()
    # Лемма, которой нет в корпусе (df=0), запросу не помогает.
    return [lexeme for lexeme, df in rows if df > 0][:limit]


@dataclass(frozen=True, slots=True)
class ChunkHit:
    chunk_id: str
    text_raw: str
    text_enriched: str
    embedding_raw: list[float]
    page: int | None
    section_title: str | None
    doc_title: str
    doc_date: date | None
    industry: str | None
    tender_result: str | None
    publicity: str
    client: str | None
    anon_label: str | None
    project: str | None = None

    @property
    def locator(self) -> str:
        """Где в документе: «слайд N» для презентаций, раздел — для документов."""
        if self.page is not None:
            return f"слайд {self.page}"
        return f"раздел «{self.section_title}»" if self.section_title else "без раздела"


def fetch_chunks(conn: Any, chunk_ids: Sequence[str]) -> dict[str, ChunkHit]:
    if not chunk_ids:
        return {}
    rows = conn.execute(
        """
        SELECT c.chunk_id, c.text_raw, c.text_enriched, c.embedding_raw::text, c.page_start,
               c.section_title, d.title, d.doc_date, p.industry, p.tender_result, p.publicity,
               cl.canonical_name, p.anon_label, p.name
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        JOIN projects p ON p.project_id = c.project_id
        LEFT JOIN clients cl ON cl.client_id = p.client_id
        WHERE c.chunk_id = ANY(%s::uuid[])
        """,
        (list(chunk_ids),),
    ).fetchall()
    return {
        str(r[0]): ChunkHit(str(r[0]), r[1], r[2], json.loads(r[3]), r[4], r[5], r[6], r[7],
                            r[8], r[9], r[10], r[11], r[12], r[13])
        for r in rows
    }


# ---------------------------------------------------------------- обзор


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    doc_id: str
    title: str
    file_path: str
    doc_type: str
    status: str
    project: str
    customer: str | None
    chunks: int


def list_documents(conn: Any, version: str | None) -> list[DocumentInfo]:
    """Документы с числом чанков в версии индекса — для интерфейса и фильтров."""
    rows = conn.execute(
        """
        SELECT d.doc_id, d.title, d.file_path, d.doc_type, d.status, p.name,
               CASE WHEN p.publicity = 'anon' THEN p.anon_label ELSE cl.canonical_name END,
               count(c.chunk_id)
        FROM documents d
        JOIN projects p ON p.project_id = d.project_id
        LEFT JOIN clients cl ON cl.client_id = p.client_id
        LEFT JOIN chunks c ON c.doc_id = d.doc_id AND c.index_version = %s
        GROUP BY d.doc_id, p.name, p.publicity, p.anon_label, cl.canonical_name
        ORDER BY d.title
        """,
        (version,),
    ).fetchall()
    return [DocumentInfo(str(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows]
