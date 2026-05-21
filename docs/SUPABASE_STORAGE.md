# Supabase Storage (PDFs + figures)

## 1. Create bucket

1. Supabase Dashboard → **Storage** → **New bucket**
2. Name: `rag-uploads` (must match `SUPABASE_STORAGE_BUCKET` in `.env`)
3. **Private** bucket recommended (files served via your FastAPI API)

## 2. Environment (Render + local)

```env
USE_SUPABASE_STORAGE=true
SUPABASE_STORAGE_BUCKET=rag-uploads
SUPABASE_URL=https://yfmawiwzcozwejugutxv.supabase.co
SUPABASE_KEY=<anon_or_publishable_key>
SUPABASE_SERVICE_ROLE_KEY=<service_role_secret>   # required for uploads (bypasses RLS)
```

Optional (AWS CLI / external tools only; the app uses `supabase-py`, not S3 SDK):

```env
SUPABASE_S3_ENDPOINT=https://yfmawiwzcozwejugutxv.storage.supabase.co/storage/v1/s3
```

## 3. Object layout

```text
rag-uploads/
  {document_id}/source.pdf
  {document_id}/images/page_3_0.png
```

`documents.pdf_path` and `raw_images.image_path` store these **object keys**, not local paths.

## 4. Fix `403 row-level security policy` (your current error)

The **publishable** key (`sb_publishable_...`) cannot upload to a private bucket until you do **one** of:

### Option A — Service role (recommended for Render)

1. Supabase Dashboard → **Project Settings** → **API**
2. Copy **service_role** secret (never put in frontend)
3. Add to `backend/.env` and Render:

```env
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOi...   # service_role JWT
```

Keep `SUPABASE_KEY` as publishable for the REST DB client.

### Option B — Storage RLS policies (publishable key)

1. Open **SQL Editor** in Supabase
2. Run the full script: [`backend/supabase_storage_policies.sql`](../supabase_storage_policies.sql)
3. Restart the API and re-upload a PDF

On startup the app also tries to run that SQL via `SUPABASE_DB_URL` if the direct DB host resolves.

## 5. Policies reference

## 6. Migrate existing local files (optional)

Re-upload documents via the UI, or run a one-off script that uploads `backend/uploads/{id}.pdf` and `backend/uploads/images/{id}/` to the bucket.
