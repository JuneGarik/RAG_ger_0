"""Поиск по грани: связка списков, реранка, буста и MMR — на подменённых базе и моделях."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from bzpo.config import load_config
from bzpo.search import query as query_module
from bzpo.storage.repo import ChunkHit


def hit(cid: str, vector: list[float], publicity: str = "open") -> ChunkHit:
    return ChunkHit(cid, f"текст {cid}", f"шапка {cid}", vector, 1, "Раздел", "Отчёт",
                    date(2025, 7, 18), "Недвижимость", "won", publicity, "ГК Ромашка",
                    "крупный девелопер")


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def rerank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        ranked = [(i, self.scores[d.split()[-1]]) for i, d in enumerate(docs)]
        return sorted(ranked, key=lambda p: -p[1])


@pytest.fixture
def fake_repo(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}
    chunks = {
        "a": hit("a", [1.0, 0.0]),
        "b": hit("b", [0.99, 0.01]),  # почти копия a
        "c": hit("c", [0.0, 1.0], publicity="anon"),
    }

    def vector_search(conn, vector, *, version, publicity, k, scope=None):
        calls["publicity"] = publicity
        calls["version"] = version
        calls["scope"] = scope
        return ["a", "b", "c"]

    monkeypatch.setattr(query_module.repo, "active_index_version", lambda conn: "v1")
    monkeypatch.setattr(query_module.repo, "vector_search", vector_search)
    monkeypatch.setattr(query_module.repo, "rarest_terms", lambda conn, q, version: ["лемма"])
    monkeypatch.setattr(query_module.repo, "fts_search", lambda conn, t, **kw: ["c", "a"])
    monkeypatch.setattr(query_module.repo, "fetch_chunks",
                        lambda conn, ids: {i: chunks[i] for i in ids})
    return calls


def test_search_pipeline_ranks_and_diversifies(fake_repo: dict[str, Any]) -> None:
    settings = load_config(use_yaml=False)
    settings.search.mmr_lambda = 0.5
    # b по оценке выше c, но почти копия a: MMR должен предпочесть c.
    reranker = FakeReranker({"a": 0.9, "b": 0.89, "c": 0.88})
    result = query_module.search(None, settings, FakeEmbedder(), reranker, "запрос", top_k=2)

    assert fake_repo["version"] == "v1"
    assert fake_repo["publicity"] == ["open"]  # фильтр публичности передан в базу
    assert [h.chunk_id for h in result.hits][0] == "a"
    # MMR: вторым идёт не почти-дубль b, а другой по содержанию c.
    assert [h.chunk_id for h in result.hits] == ["a", "c"]
    assert result.hits[0].lists == {"vector:query": 1, "fts:query": 2}
    assert "won" in result.hits[0].boost
    assert result.hits[1].customer == "крупный девелопер"  # anon -> обезличенное имя


def test_threshold_gives_not_found(fake_repo: dict[str, Any]) -> None:
    settings = load_config(use_yaml=False)
    settings.search.rerank_threshold = 0.95
    reranker = FakeReranker({"a": 0.9, "b": 0.85, "c": 0.6})
    result = query_module.search(None, settings, FakeEmbedder(), reranker, "запрос")
    assert result.not_found and result.hits == []


def test_no_active_index_is_explicit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(query_module.repo, "active_index_version", lambda conn: None)
    with pytest.raises(LookupError, match="bzpo.ingest"):
        query_module.search(None, load_config(use_yaml=False), FakeEmbedder(),
                            FakeReranker({}), "запрос")


def test_scope_reaches_repository(fake_repo: dict[str, Any]) -> None:
    from bzpo.storage.repo import SearchScope

    scope = SearchScope(doc_types=["kp"], doc_ids=["d1"])
    query_module.search(None, load_config(use_yaml=False), FakeEmbedder(),
                        FakeReranker({"a": 0.9, "b": 0.85, "c": 0.88}), "запрос", scope=scope)
    assert fake_repo["scope"] is scope
    assert scope.params() == {"doc_types": ["kp"], "doc_ids": ["d1"]}
    assert SearchScope().params() == {"doc_types": None, "doc_ids": None}
