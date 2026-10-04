#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Разведка внешнего сервиса эмбеддингов и реранкинга.

Запускается один раз, когда придут адрес и токен. Отвечает на вопросы,
которые иначе пришлось бы выяснять у владельца сервиса:

  1. Какой диалект у эндпоинта реранкера (TEI / Cohere-Jina).
  2. Отдаёт ли эндпоинт эмбеддингов разреженные векторы.
  3. Какова реальная максимальная длина входа.
  4. Есть ли что-то пригодное на роль проверки доступности.
  5. Как считается лимит запросов: по HTTP-вызовам или по элементам.

Использование:
    export EMB_URL="https://адрес-сервера"
    export EMB_TOKEN="токен"
    python3 probe_api.py
"""

import os, json, time, urllib.request, urllib.error

BASE  = os.environ.get("EMB_URL", "").rstrip("/")
TOKEN = os.environ.get("EMB_TOKEN", "")
EMB_MODEL  = os.environ.get("EMB_MODEL",  "BAAI/bge-m3")
RANK_MODEL = os.environ.get("RANK_MODEL", "BAAI/bge-reranker-v2-m3")

if not BASE or not TOKEN:
    raise SystemExit("Задайте EMB_URL и EMB_TOKEN в переменных окружения")

HDR = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
report = {}


def call(path, payload=None, method=None, timeout=30):
    """Возвращает (код, тело, миллисекунды). Тело — dict/list либо строка."""
    url = BASE + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=HDR,
                                 method=method or ("POST" if data else "GET"))
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
            ms = int((time.time() - t0) * 1000)
            try:    return r.status, json.loads(body), ms
            except: return r.status, body[:400], ms
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        return e.code, body, int((time.time() - t0) * 1000)
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", int((time.time() - t0) * 1000)


def head(t):
    print("\n" + "=" * 66); print(t); print("=" * 66)


# ---------------------------------------------------------------- 1. модели
head("1. Перечень моделей и кандидат на проверку доступности")
for p in ["/v1/models", "/models", "/health", "/healthz", "/ping", "/info"]:
    code, body, ms = call(p)
    mark = "OK " if code == 200 else "   "
    print(f"{mark}GET {p:12s} → {code} за {ms} мс")
    if code == 200:
        report.setdefault("alive_endpoints", []).append(p)
        if p in ("/v1/models", "/models") and isinstance(body, dict):
            ids = [m.get("id") for m in body.get("data", [])]
            print(f"    модели: {ids}")
            report["models"] = ids

# ---------------------------------------------------------- 2. эмбеддинги
head("2. Эмбеддинги: размерность и наличие разреженных векторов")
code, body, ms = call("/v1/embeddings",
                      {"model": EMB_MODEL, "input": ["выручка выросла на 12 процентов"]})
print(f"POST /v1/embeddings → {code} за {ms} мс")
if code == 200 and isinstance(body, dict):
    item = body["data"][0]
    dense = item.get("embedding")
    print(f"    ключи ответа: {list(item.keys())}")
    print(f"    размерность плотного вектора: {len(dense) if dense else '—'}")
    report["dim"] = len(dense) if dense else None
    sparse_keys = [k for k in item.keys()
                   if k in ("sparse_embedding", "sparse", "lexical_weights", "colbert_vecs")]
    if sparse_keys:
        print(f"    РАЗРЕЖЕННЫЕ ВЕКТОРЫ ЕСТЬ: {sparse_keys}")
    else:
        print("    разреженных векторов нет — лексический поиск делаем на PostgreSQL FTS")
    report["sparse"] = bool(sparse_keys)
else:
    print(f"    {body}")

# пробуем явно запросить sparse
for extra in [{"encoding_format": "float", "return_sparse": True},
              {"return_dense": True, "return_sparse": True, "return_colbert_vecs": False}]:
    payload = {"model": EMB_MODEL, "input": ["тест"], **extra}
    code, body, ms = call("/v1/embeddings", payload)
    if code == 200 and isinstance(body, dict):
        ks = list(body["data"][0].keys())
        if any(k for k in ks if "sparse" in k or "lexical" in k):
            print(f"    при параметрах {list(extra)} появились: {ks}")
            report["sparse"] = True

# --------------------------------------------------------- 3. длина входа
head("3. Реальный предел длины входа")
for n_tokens in (512, 2000, 4000, 8000, 8300):
    text = "раздел коммерческого предложения " * (n_tokens // 5)
    code, body, ms = call("/v1/embeddings", {"model": EMB_MODEL, "input": [text]})
    status = "принят" if code == 200 else f"отказ {code}"
    print(f"    ≈{n_tokens:5d} токенов → {status} за {ms} мс")
    if code != 200:
        report["max_input_hint"] = n_tokens
        break

# ------------------------------------------------------------ 4. реранкер
head("4. Диалект эндпоинта реранкера")
QUERY = "выручка выросла"
DOCS  = ["Выручка компании выросла на 12% в отчётном периоде.",
         "Чистая прибыль увеличилась незначительно.",
         "Выручка компании не выросла в отчётном периоде."]

variants = [
    ("TEI",         "/rerank",     {"query": QUERY, "texts": DOCS, "raw_scores": False}),
    ("TEI",         "/v1/rerank",  {"query": QUERY, "texts": DOCS, "raw_scores": False}),
    ("Cohere/Jina", "/v1/rerank",  {"model": RANK_MODEL, "query": QUERY, "documents": DOCS}),
    ("Cohere/Jina", "/rerank",     {"model": RANK_MODEL, "query": QUERY, "documents": DOCS}),
    ("vLLM score",  "/v1/score",   {"model": RANK_MODEL, "text_1": QUERY, "text_2": DOCS}),
]
for name, path, payload in variants:
    code, body, ms = call(path, payload)
    if code == 200:
        print(f"OK  {name:12s} POST {path:12s} → 200 за {ms} мс")
        print(f"    ответ: {json.dumps(body, ensure_ascii=False)[:300]}")
        report["rerank"] = {"dialect": name, "path": path, "request": payload}
        break
    else:
        print(f"    {name:12s} POST {path:12s} → {code}")
else:
    print("    Ни один известный диалект не подошёл — нужен запрос владельцу сервиса")

# ------------------------------------------------------------- 5. лимиты
head("5. Как считается лимит: по запросам или по элементам")
batch = ["фрагмент коммерческого предложения номер %d" % i for i in range(128)]
code, body, ms = call("/v1/embeddings", {"model": EMB_MODEL, "input": batch})
print(f"    пачка из 128 текстов одним запросом → {code} за {ms} мс")
if code == 200 and isinstance(body, dict):
    print(f"    получено векторов: {len(body.get('data', []))}")
    print(f"    usage: {body.get('usage')}")

print("\n    серия из 25 одиночных запросов подряд:")
codes = []
t0 = time.time()
for i in range(25):
    c, _, _ = call("/v1/embeddings", {"model": EMB_MODEL, "input": ["проверка лимита"]})
    codes.append(c)
el = time.time() - t0
print(f"    {len(codes)} запросов за {el:.1f} с ({len(codes)/el:.1f} в секунду)")
if 429 in codes:
    print("    получен 429 — лимит считается по HTTP-запросам, нужны повторы с задержкой")
else:
    print("    отказов нет — лимит не достигнут при такой интенсивности")

# ------------------------------------------------------------- итог
head("ИТОГ")
print(json.dumps(report, ensure_ascii=False, indent=2))
print("""
Что делать с результатом:
  · report["sparse"] = false  → лексический поиск на PostgreSQL FTS, как заложено
  · report["rerank"]          → зафиксировать диалект в модуле изоляции зависимости
  · report["alive_endpoints"] → выбрать путь для синтетической проверки доступности
  · report["dim"]             → сверить с 1024, заложенными в схему pgvector
""")
