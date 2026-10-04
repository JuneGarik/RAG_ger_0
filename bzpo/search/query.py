"""Поиск по одной грани (SPEC 5.2) без HyDE: два списка -> RRF -> реранк -> boost -> MMR.

Списки гипотезы HyDE подключаются, когда появится разбор запроса (этап 3.5):
`standard_lists` уже принимает их, схема вырождается в два списка штатно.

FTS строится по 8 самым редким в корпусе леммам запроса (SPEC 5.2), а не по
всем словам: частые слова («компания», «анализ») тянут половину корпуса.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bzpo.config import Settings
from bzpo.providers.base import Reranker
from bzpo.search.boost import BoostInput, BoostWeights, apply_boost
from bzpo.search.fusion import standard_lists, weighted_rrf
from bzpo.search.mmr import MMRCandidate, mmr
from bzpo.storage import repo


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    score: float
    rerank_score: float
    boost: tuple[str, ...]
    lists: dict[str, int]  # из каких списков и на каком месте пришёл кандидат
    chunk: repo.ChunkHit

    @property
    def customer(self) -> str | None:
        """Как называть заказчика в выдаче: с учётом публичности проекта."""
        return self.chunk.anon_label if self.chunk.publicity == "anon" else self.chunk.client


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    version: str
    hits: list[SearchHit]
    terms: list[str]
    not_found: bool


def search(
    conn: Any,
    settings: Settings,
    embedder: Any,
    reranker: Reranker,
    query: str,
    *,
    top_k: int | None = None,
    query_industry: str | None = None,
    scope: repo.SearchScope | None = None,
    candidates: int | None = None,
) -> SearchResult:
    """Найти фрагменты по одному запросу.

    `candidates` — сколько кандидатов после RRF отдать реранкеру (по умолчанию
    search.candidates). Реранкер — самая дорогая стадия: при подборе контекста
    по многим граням меньший срез заметно ускоряет поиск.
    """
    cfg = settings.search
    version = cfg.active_index_version or repo.active_index_version(conn)
    if version is None:
        raise LookupError("нет активной версии индекса: запустите python -m bzpo.ingest")
    publicity = list(cfg.allowed_publicity)

    vector_ids = repo.vector_search(
        conn, embedder.embed_query(query), version=version, publicity=publicity,
        k=cfg.per_list_k, scope=scope,
    )
    terms = repo.rarest_terms(conn, query, version=version)
    fts_ids = repo.fts_search(
        conn, terms, version=version, publicity=publicity, k=cfg.per_list_k, scope=scope
    )
    fused = weighted_rrf(
        standard_lists(vector_query=vector_ids, fts_query=fts_ids),
        rrf_k=cfg.rrf_k, top_n=candidates or cfg.candidates,
    )
    chunks = repo.fetch_chunks(conn, [c.candidate_id for c in fused])
    candidates = [c for c in fused if c.candidate_id in chunks]
    if not candidates:
        return SearchResult(query, version, [], terms, not_found=True)

    texts = [getattr(chunks[c.candidate_id], cfg.rerank_input) for c in candidates]
    reranked = reranker.rerank(query, texts)
    rerank_scores = {candidates[i].candidate_id: score for i, score in reranked}
    best = max(rerank_scores.values())
    if cfg.rerank_threshold is not None and best < cfg.rerank_threshold:
        return SearchResult(query, version, [], terms, not_found=True)

    boosted = apply_boost(
        [
            BoostInput(cid, score, chunks[cid].industry, chunks[cid].tender_result,
                       chunks[cid].doc_date)
            for cid, score in rerank_scores.items()
        ],
        weights=BoostWeights.from_settings(cfg),
        query_industry=query_industry,
    )
    by_id = {b.candidate_id: b for b in boosted}
    selected = mmr(
        [MMRCandidate(b.candidate_id, b.score, chunks[b.candidate_id].embedding_raw)
         for b in boosted],
        lambda_=cfg.mmr_lambda, top_k=top_k or cfg.top_k_final,
    )
    provenance = {c.candidate_id: dict(c.provenance) for c in candidates}
    hits = [
        SearchHit(s.candidate_id, by_id[s.candidate_id].score, rerank_scores[s.candidate_id],
                  by_id[s.candidate_id].applied, provenance[s.candidate_id],
                  chunks[s.candidate_id])
        for s in selected
    ]
    return SearchResult(query, version, hits, terms, not_found=False)
