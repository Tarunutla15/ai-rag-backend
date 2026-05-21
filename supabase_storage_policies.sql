-- Run in Supabase SQL Editor if uploads fail with "row-level security policy".
-- Bucket name must match SUPABASE_STORAGE_BUCKET (default: rag-uploads).

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES ('rag-uploads', 'rag-uploads', false, 52428800, NULL)
ON CONFLICT (id) DO NOTHING;

DROP POLICY IF EXISTS "rag_uploads_select" ON storage.objects;
DROP POLICY IF EXISTS "rag_uploads_insert" ON storage.objects;
DROP POLICY IF EXISTS "rag_uploads_update" ON storage.objects;
DROP POLICY IF EXISTS "rag_uploads_delete" ON storage.objects;

CREATE POLICY "rag_uploads_select"
ON storage.objects FOR SELECT
USING (bucket_id = 'rag-uploads');

CREATE POLICY "rag_uploads_insert"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'rag-uploads');

CREATE POLICY "rag_uploads_update"
ON storage.objects FOR UPDATE
USING (bucket_id = 'rag-uploads')
WITH CHECK (bucket_id = 'rag-uploads');

CREATE POLICY "rag_uploads_delete"
ON storage.objects FOR DELETE
USING (bucket_id = 'rag-uploads');
