-- 0001_initial — схема данных БЗПО (SPEC раздел 3).
--
-- Порядок объектов обязателен и повторяет SPEC: реестр версий индекса ->
-- справочники -> проекты -> загрузки реестра -> документы -> разделы ->
-- чанки -> артефакты -> рёбра графа. Циклических FK нет, скрипт исполняется
-- сверху вниз на чистой базе одной транзакцией раннера (bzpo/storage/migrate.py).
--
-- Таблица enrichment_cache (SPEC 4.3) в эту миграцию не входит: она относится
-- к конвейеру обогащения этапа 2 и приедет отдельной нумерованной миграцией.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------- реестр версий индекса ----------
CREATE TABLE index_versions (
  version         text PRIMARY KEY,
  embedder_model  text NOT NULL,
  embedder_revision text,
  parser_version  text NOT NULL,
  chunk_params    jsonb NOT NULL,        -- CHUNK_TARGET, MAX, MIN, OVERLAP
  enrich_prompt_version text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  status          text NOT NULL CHECK (status IN ('building','active','retired','failed'))
);
-- Одновременно активна ровно одна версия. Все поисковые запросы
-- фильтруются по index_version = активной. Предыдущая версия хранится
-- до подтверждения качества новой (регрессия на датасете), затем retired.

-- ---------- БД подтверждений ----------
CREATE TABLE clients (
  client_id      uuid PRIMARY KEY,
  canonical_name text NOT NULL,
  aliases        text[] NOT NULL DEFAULT '{}'
);

CREATE TABLE projects (
  project_id     uuid PRIMARY KEY,
  name           text NOT NULL,
  client_id      uuid REFERENCES clients(client_id),
  industry       text,
  publicity      text NOT NULL CHECK (publicity IN ('open','anon','closed')),
  anon_label     text,
  tender_result  text CHECK (tender_result IN ('won','lost','not_submitted')),
  amount         numeric,
  date_start     date, date_end date,
  company_role   text CHECK (company_role IN ('general','subcontractor','consultant')),
  tech_stack     text[],
  contract_no    text,
  status         text CHECK (status IN ('active','archived')),
  CONSTRAINT anon_label_required CHECK (publicity <> 'anon' OR anon_label IS NOT NULL)
);

-- Версионируемые загрузки реестра из Excel (SPEC 4.5)
CREATE TABLE registry_uploads (
  upload_id    uuid PRIMARY KEY,
  loaded_at    timestamptz NOT NULL DEFAULT now(),
  source_file  text NOT NULL,
  row_count    int,
  rejected_count int,
  report       jsonb                      -- построчные ошибки валидации
);

-- ---------- корпус ----------
CREATE TABLE documents (
  doc_id        uuid PRIMARY KEY,
  project_id    uuid NOT NULL REFERENCES projects(project_id),
  title         text NOT NULL,            -- человекочитаемое название для списка «Источники»
  file_path     text NOT NULL UNIQUE,
  file_hash     text NOT NULL,
  doc_type      text NOT NULL CHECK (doc_type IN ('kp','report','act','contract','letter','presentation')),
  doc_date      date,                     -- дата документа, для boost по свежести
  status        text NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','parsed','enriched','indexed','failed')),
  last_error    text,
  attempts      int NOT NULL DEFAULT 0,
  parsed_at     timestamptz,
  parser_version text,
  page_count    int
);
CREATE INDEX ON documents (file_hash);
-- Сопоставление файл -> проект выполняется по реестру ДО индексации (шаг 0
-- офлайн-конвейера). Файл без сопоставленного проекта в индекс не попадает
-- и остаётся в статусе failed с внятной причиной: project_id NOT NULL
-- гарантирует это на уровне схемы. Поведение по умолчанию — «считать закрытым».

CREATE TABLE sections (
  section_id    uuid PRIMARY KEY,
  doc_id        uuid NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  project_id    uuid NOT NULL REFERENCES projects(project_id),
  title         text NOT NULL,
  level         int  NOT NULL,
  page_start    int, page_end int,        -- NULL допустим: DOCX/PPTX без рендеринга страниц
  text          text NOT NULL,
  content_hash  text NOT NULL,            -- для инкрементальной переиндексации
  embedding     vector(1024) NOT NULL,
  token_count   int,
  index_version text NOT NULL REFERENCES index_versions(version)
);
CREATE INDEX ON sections (doc_id);
CREATE INDEX ON sections (index_version);

CREATE TABLE chunks (
  chunk_id        uuid PRIMARY KEY,
  doc_id          uuid NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  project_id      uuid NOT NULL REFERENCES projects(project_id),
  ord             int  NOT NULL,
  section_title   text,
  subsection_title text,
  parent_section_id uuid REFERENCES sections(section_id),
  page_start      int, page_end int,      -- NULL допустим (см. sections)
  text_raw        text NOT NULL,
  text_enriched   text NOT NULL,
  content_hash    text NOT NULL,          -- hash(text_raw), ключ кэша обогащения
  embedding       vector(1024) NOT NULL,  -- по text_enriched; используется в поиске
  embedding_raw   vector(1024) NOT NULL,  -- по text_raw; используется ТОЛЬКО для
                                          -- дедупликации, MMR и семантических рёбер.
                                          -- Обогащение добавляет заказчика и отрасль,
                                          -- поэтому близость копий типового блока по
                                          -- text_enriched падает ниже порога — дедупликация
                                          -- по нему не работает по построению
  tsv             tsvector GENERATED ALWAYS AS (to_tsvector('russian'::regconfig, text_raw)) STORED,
  token_count     int,
  is_table        boolean DEFAULT false,
  ocr_low_conf    boolean DEFAULT false,  -- потребитель: пометка «низкое качество
                                          -- распознавания» в списке «Источники»; из выдачи
                                          -- не исключается
  cluster_id      uuid,
  cluster_size    int  DEFAULT 1,         -- для флага boilerplate и метрики уникальности
  is_canonical    boolean DEFAULT true,
  index_version   text NOT NULL REFERENCES index_versions(version)
);
CREATE INDEX ON chunks USING gin(tsv);
CREATE INDEX ON chunks (project_id);
CREATE INDEX ON chunks (cluster_id, is_canonical);
CREATE INDEX ON chunks (index_version);
-- ANN-индекс НЕ создаётся: точный перебор по 20 тыс. векторов

CREATE TABLE artifacts (
  artifact_id  uuid PRIMARY KEY,
  project_id   uuid NOT NULL REFERENCES projects(project_id),
  artifact_type text NOT NULL CHECK (artifact_type IN ('act','contract','letter')),
  title        text NOT NULL,             -- наименование для списка «Подтверждения»
  number       text,
  issued_at    date,
  file_path    text NOT NULL,
  doc_id       uuid REFERENCES documents(doc_id)  -- связь с проиндексированной копией,
                                                  -- если файл артефакта входит в корпус
);

-- ---------- граф связей (MVP+) ----------
CREATE TABLE chunk_edges (
  src_chunk_id uuid REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  dst_chunk_id uuid REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  edge_type    text NOT NULL CHECK (edge_type IN ('parent','sibling','duplicate','semantic','project')),
  weight       real,
  PRIMARY KEY (src_chunk_id, dst_chunk_id, edge_type)
);
