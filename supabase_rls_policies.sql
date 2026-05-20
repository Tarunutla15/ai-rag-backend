-- Run this in Supabase Dashboard → SQL Editor (one-time).
-- Fixes: "new row violates row-level security policy" when using the anon key.
-- This allows the backend (anon key) to INSERT/SELECT/UPDATE/DELETE on all app tables.

-- chat_sessions
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON chat_sessions;
CREATE POLICY "Allow all for anon" ON chat_sessions FOR ALL TO anon USING (true) WITH CHECK (true);

-- documents
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON documents;
CREATE POLICY "Allow all for anon" ON documents FOR ALL TO anon USING (true) WITH CHECK (true);

-- session_documents
ALTER TABLE session_documents ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON session_documents;
CREATE POLICY "Allow all for anon" ON session_documents FOR ALL TO anon USING (true) WITH CHECK (true);

-- chat_messages
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON chat_messages;
CREATE POLICY "Allow all for anon" ON chat_messages FOR ALL TO anon USING (true) WITH CHECK (true);

-- chunks
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON chunks;
CREATE POLICY "Allow all for anon" ON chunks FOR ALL TO anon USING (true) WITH CHECK (true);

-- raw_tables
ALTER TABLE raw_tables ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON raw_tables;
CREATE POLICY "Allow all for anon" ON raw_tables FOR ALL TO anon USING (true) WITH CHECK (true);

-- raw_code_blocks
ALTER TABLE raw_code_blocks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON raw_code_blocks;
CREATE POLICY "Allow all for anon" ON raw_code_blocks FOR ALL TO anon USING (true) WITH CHECK (true);

-- raw_images
ALTER TABLE raw_images ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON raw_images;
CREATE POLICY "Allow all for anon" ON raw_images FOR ALL TO anon USING (true) WITH CHECK (true);

-- chunks_fts
ALTER TABLE chunks_fts ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON chunks_fts;
CREATE POLICY "Allow all for anon" ON chunks_fts FOR ALL TO anon USING (true) WITH CHECK (true);

-- usage_events (LLM token / dashboard)
ALTER TABLE usage_events ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON usage_events;
CREATE POLICY "Allow all for anon" ON usage_events FOR ALL TO anon USING (true) WITH CHECK (true);

-- document_nodes / document_edges (Phase 1 semantic document tree)
-- Required for upload: tree is written here before chunks/Zilliz.
ALTER TABLE document_nodes ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON document_nodes;
DROP POLICY IF EXISTS "Allow all for authenticated" ON document_nodes;
CREATE POLICY "Allow all for anon" ON document_nodes FOR ALL TO anon USING (true) WITH CHECK (true);
CREATE POLICY "Allow all for authenticated" ON document_nodes FOR ALL TO authenticated USING (true) WITH CHECK (true);

ALTER TABLE document_edges ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all for anon" ON document_edges;
DROP POLICY IF EXISTS "Allow all for authenticated" ON document_edges;
CREATE POLICY "Allow all for anon" ON document_edges FOR ALL TO anon USING (true) WITH CHECK (true);
CREATE POLICY "Allow all for authenticated" ON document_edges FOR ALL TO authenticated USING (true) WITH CHECK (true);
