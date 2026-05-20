-- Run this in Supabase Dashboard → SQL Editor (one-time setup).
-- Use this when SUPABASE_DB_URL cannot be used (e.g. DNS blocks db.xxx.supabase.co).

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT DEFAULT 'New Chat',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_message_at TIMESTAMPTZ DEFAULT NOW()
);

-- Canonical document registry (DB-backed; replaces document_registry.json)
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'UPLOADED',
    chunk_count INT NOT NULL DEFAULT 0,
    technology TEXT DEFAULT 'general',
    domain TEXT DEFAULT 'general',
    pdf_path TEXT,
    error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- Session-scoped document segregation (many-to-many)
CREATE TABLE IF NOT EXISTS session_documents (
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (session_id, document_id)
);
CREATE INDEX IF NOT EXISTS idx_session_documents_session ON session_documents(session_id);
CREATE INDEX IF NOT EXISTS idx_session_documents_document ON session_documents(document_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON chat_messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT,
    chunk_type TEXT,
    retrieval_text TEXT,
    page_number INT,
    section_title TEXT,
    metadata_json TEXT,
    created_at BIGINT
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS raw_tables (
    table_id TEXT PRIMARY KEY,
    document_id TEXT,
    page_number INT,
    headers_json TEXT,
    rows_json TEXT,
    created_at BIGINT
);
CREATE INDEX IF NOT EXISTS idx_raw_tables_document ON raw_tables(document_id);

CREATE TABLE IF NOT EXISTS raw_code_blocks (
    code_id TEXT PRIMARY KEY,
    document_id TEXT,
    language TEXT,
    code_text TEXT,
    page_number INT,
    created_at BIGINT
);
CREATE INDEX IF NOT EXISTS idx_raw_code_document ON raw_code_blocks(document_id);

CREATE TABLE IF NOT EXISTS raw_images (
    image_id TEXT PRIMARY KEY,
    document_id TEXT,
    page_number INT,
    caption TEXT,
    image_path TEXT,
    created_at BIGINT
);
CREATE INDEX IF NOT EXISTS idx_raw_images_document ON raw_images(document_id);

CREATE TABLE IF NOT EXISTS chunks_fts (
    id BIGSERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    chunk_id TEXT,
    document_id TEXT,
    file_name TEXT,
    technology TEXT,
    domain TEXT,
    chunk_index TEXT,
    file_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_fts_document_id ON chunks_fts(document_id);

-- Phase 1: semantic document tree (section hierarchy + reading order)
CREATE TABLE IF NOT EXISTS document_nodes (
    node_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    title TEXT,
    content_preview TEXT,
    page_start INT DEFAULT 1,
    page_end INT DEFAULT 1,
    parent_node_id TEXT,
    depth INT DEFAULT 0,
    order_index INT DEFAULT 0,
    section_path_json TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_document_nodes_document ON document_nodes(document_id);
CREATE INDEX IF NOT EXISTS idx_document_nodes_parent ON document_nodes(parent_node_id);

CREATE TABLE IF NOT EXISTS document_edges (
    edge_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    from_node_id TEXT NOT NULL,
    to_node_id TEXT NOT NULL,
    edge_type TEXT NOT NULL DEFAULT 'parent_of'
);
CREATE INDEX IF NOT EXISTS idx_document_edges_document ON document_edges(document_id);

-- LLM usage / dashboard (token counts per chat completion)
CREATE TABLE IF NOT EXISTS usage_events (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT,
    message_id BIGINT,
    event_type TEXT NOT NULL DEFAULT 'chat_completion',
    query_preview TEXT,
    prompt_tokens INT,
    completion_tokens INT,
    total_tokens INT,
    model TEXT,
    provider TEXT,
    cost_usd DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_usage_events_created ON usage_events(created_at DESC);

-- ============================================================
-- OPTIONAL: RESET / CLEAR ALL APP DATA (keeps schema)
-- ============================================================
-- Run this ONLY if you want to wipe Supabase relational data.
-- Then clear Zilliz separately (see backend/app/services/vector_store.py delete_by_document_id).
--
-- TRUNCATE TABLE
--   chat_messages,
--   session_documents,
--   chat_sessions,
--   documents,
--   chunks_fts,
--   chunks,
--   raw_tables,
--   raw_code_blocks,
--   raw_images,
--   document_edges,
--   document_nodes
-- RESTART IDENTITY;
