"""
Supabase Storage for PDFs and extracted images (durable on Render).

Object layout in bucket ``{SUPABASE_STORAGE_BUCKET}``::

    {document_id}/source.pdf
    {document_id}/images/page_{N}_{idx}.png

Uses the Supabase Storage REST API (same project as SUPABASE_URL).
Optional S3 endpoint (SUPABASE_S3_ENDPOINT) is documented in env.example for AWS CLI;
this module uses supabase-py, not raw S3.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.config import settings

logger = logging.getLogger(__name__)

_STORAGE_KEY_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/",
    re.I,
)
_PAGE_FILE_RE = re.compile(r"page_(\d+)_", re.I)


def _bucket_name() -> str:
    return (getattr(settings, "SUPABASE_STORAGE_BUCKET", "") or "rag-uploads").strip()


_storage_client_cache = None
_policies_ensured = False
_rls_hint_logged = False
_upload_fail_count = 0


def _resolve_storage_api_key() -> str:
    """
    Key used for Storage API. Service role bypasses RLS.
    Legacy JWT service keys (eyJ...) in SUPABASE_KEY are also accepted.
    """
    role = (getattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "") or "").strip()
    if role:
        return role
    main = (getattr(settings, "SUPABASE_KEY", "") or "").strip()
    if main.startswith("eyJ"):
        return main
    return ""


def storage_uses_service_role() -> bool:
    return bool(_resolve_storage_api_key())


def _is_rls_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "row-level security" in msg or (
        "403" in msg and "unauthorized" in msg
    )


def ensure_storage_policies() -> bool:
    """
    Create Storage RLS policies for rag-uploads via direct Postgres (SUPABASE_DB_URL).
    Safe to call repeatedly (idempotent SQL).
    """
    global _policies_ensured
    if _policies_ensured:
        return True
    try:
        from app.services.database import _get_supabase_db_url

        db_url = _get_supabase_db_url()
        if not db_url:
            return False
        import psycopg2

        sql_path = Path(__file__).resolve().parents[2] / "supabase_storage_policies.sql"
        if not sql_path.is_file():
            logger.warning("blob_storage: missing %s", sql_path)
            return False
        sql = sql_path.read_text(encoding="utf-8")
        conn = psycopg2.connect(db_url.strip(), connect_timeout=15)
        try:
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(sql)
            cur.close()
        finally:
            conn.close()
        _policies_ensured = True
        _storage_client_cache = None
        logger.info(
            "blob_storage: applied Storage policies for bucket=%s (publishable key uploads enabled)",
            _bucket_name(),
        )
        return True
    except Exception as e:
        logger.warning("blob_storage: could not apply Storage policies via Postgres: %s", e)
        return False


def _log_storage_auth_hint() -> None:
    global _rls_hint_logged
    if _rls_hint_logged:
        return
    _rls_hint_logged = True
    if storage_uses_service_role():
        return
    logger.error(
        "blob_storage: Storage uploads need SUPABASE_SERVICE_ROLE_KEY in .env "
        "(Dashboard → Project Settings → API → service_role secret), "
        "OR run backend/supabase_storage_policies.sql in the SQL Editor. "
        "Publishable keys cannot upload to private buckets without policies."
    )


def _supabase_client():
    """Prefer service_role for Storage (RLS); fall back to app DB client."""
    global _storage_client_cache
    if _storage_client_cache is not None:
        return _storage_client_cache
    url = (getattr(settings, "SUPABASE_URL", "") or "").strip()
    if not url:
        return None
    api_key = _resolve_storage_api_key()
    if api_key:
        try:
            from supabase import create_client

            _storage_client_cache = create_client(url, api_key)
            return _storage_client_cache
        except Exception as e:
            logger.warning("blob_storage: elevated storage client failed: %s", e)
    try:
        from app.services.database import get_database

        db = get_database()
        if getattr(db, "engine", "") == "supabase" and getattr(db, "supabase", None):
            _storage_client_cache = db.supabase
            if not storage_uses_service_role():
                ensure_storage_policies()
            return _storage_client_cache
    except Exception as e:
        logger.debug("blob_storage: no supabase client: %s", e)
    return None


def storage_enabled() -> bool:
    if not getattr(settings, "USE_SUPABASE_STORAGE", True):
        return False
    if not (getattr(settings, "SUPABASE_URL", "") or "").strip():
        return False
    if not (getattr(settings, "SUPABASE_KEY", "") or "").strip():
        return False
    if not _bucket_name():
        return False
    return _supabase_client() is not None


def is_storage_object_key(path: Optional[str]) -> bool:
    """True when path is a bucket object key (not a legacy local uploads/ path)."""
    if not path or not str(path).strip():
        return False
    p = str(path).strip().replace("\\", "/")
    if p.startswith("uploads/") or ":/" in p[:3]:
        return False
    return bool(_STORAGE_KEY_RE.match(p))


def source_object_key(document_id: str, ext: str) -> str:
    e = (ext or ".pdf").lower()
    if not e.startswith("."):
        e = f".{e}"
    return f"{document_id}/source{e}"


def image_object_key(document_id: str, filename: str) -> str:
    name = Path(filename).name
    return f"{document_id}/images/{name}"


def _storage():
    client = _supabase_client()
    if not client:
        raise RuntimeError("Supabase client not available for storage")
    return client.storage.from_(_bucket_name())


def upload_bytes(
    object_key: str,
    data: bytes,
    *,
    content_type: str = "application/octet-stream",
) -> bool:
    global _upload_fail_count
    if not storage_enabled():
        return False
    key = object_key.strip().lstrip("/")
    try:
        _storage().upload(
            key,
            data,
            file_options={"content-type": content_type, "upsert": "true"},
        )
        logger.info("blob_storage: uploaded %s (%s bytes)", key, len(data))
        return True
    except Exception as e:
        _upload_fail_count += 1
        if _is_rls_error(e) and not storage_uses_service_role():
            if _upload_fail_count == 1:
                ensure_storage_policies()
                try:
                    _storage().upload(
                        key,
                        data,
                        file_options={
                            "content-type": content_type,
                            "upsert": "true",
                        },
                    )
                    logger.info(
                        "blob_storage: uploaded %s after applying policies",
                        key,
                    )
                    return True
                except Exception as retry_e:
                    e = retry_e
            _log_storage_auth_hint()
        if _upload_fail_count <= 2:
            logger.error("blob_storage: upload failed key=%s: %s", key, e)
        elif _upload_fail_count == 3:
            logger.error(
                "blob_storage: further upload failures suppressed "
                "(fix SUPABASE_SERVICE_ROLE_KEY or storage policies)"
            )
        return False


def upload_local_file(object_key: str, local_path: Union[str, Path]) -> bool:
    path = Path(local_path)
    if not path.is_file():
        return False
    ext = path.suffix.lower()
    ct = (
        "application/pdf"
        if ext == ".pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if ext == ".docx"
        else "image/png"
        if ext == ".png"
        else "image/jpeg"
        if ext in (".jpg", ".jpeg")
        else "application/octet-stream"
    )
    return upload_bytes(object_key, path.read_bytes(), content_type=ct)


def download_bytes(object_key: str) -> Optional[bytes]:
    if not storage_enabled():
        return None
    key = object_key.strip().lstrip("/")
    try:
        return _storage().download(key)
    except Exception as e:
        logger.debug("blob_storage: download miss key=%s: %s", key, e)
        return None


def object_exists(object_key: str) -> bool:
    if not storage_enabled():
        return False
    key = object_key.strip().lstrip("/")
    folder, _, name = key.rpartition("/")
    prefix = folder or ""
    try:
        items = _storage().list(prefix) or []
    except Exception:
        return download_bytes(key) is not None
    for item in items:
        if (item.get("name") or "") == name:
            return True
    return download_bytes(key) is not None


def list_objects_under(prefix: str) -> List[str]:
    if not storage_enabled():
        return []
    base = prefix.strip().strip("/")
    keys: List[str] = []

    def _walk(folder: str) -> None:
        try:
            items = _storage().list(folder) or []
        except Exception as e:
            logger.debug("blob_storage: list %s: %s", folder, e)
            return
        for item in items:
            name = item.get("name") or ""
            if not name:
                continue
            full = f"{folder}/{name}" if folder else name
            meta = item.get("metadata") or {}
            if meta.get("size") is None and not str(name).count("."):
                _walk(full)
            else:
                keys.append(full)

    _walk(base)
    return keys


def find_page_image_key(document_id: str, page_number: int) -> Optional[str]:
    prefix = f"{document_id}/images"
    for pattern in (f"page_{page_number}_", f"page_{int(page_number)}_"):
        for key in list_objects_under(prefix):
            if pattern in Path(key).name:
                return key
    return None


def page_image_available(document_id: str, page_number: int) -> bool:
    if find_disk_page_image_local(document_id, page_number):
        return True
    if storage_enabled():
        return find_page_image_key(document_id, page_number) is not None
    return False


def find_disk_page_image_local(document_id: str, page_number: int) -> Optional[Path]:
    base = Path(settings.UPLOAD_DIR) / "images" / document_id
    if not base.is_dir():
        return None
    try:
        page = int(page_number)
    except (TypeError, ValueError):
        return None
    for pattern in (f"page_{page}_*.png", f"page_{page}_*.jpg", f"page_{page}_*.jpeg"):
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def download_to_temp_file(
    object_key: str,
    *,
    suffix: Optional[str] = None,
) -> Optional[Path]:
    data = download_bytes(object_key)
    if not data:
        return None
    suf = suffix or Path(object_key).suffix or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suf, prefix="rag_blob_")
    try:
        import os

        with open(fd, "wb") as f:
            f.write(data)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return Path(tmp)


def resolve_registry_path_to_local(
    document_id: str,
    registry_path: Optional[str],
    upload_dir: str,
) -> Optional[Path]:
    """
    Return a local Path suitable for pdfplumber / FileResponse.
    Downloads from Supabase to a temp file when needed.
    """
    from app.services.document_processor import resolve_stored_document_path

    local = resolve_stored_document_path(document_id, upload_dir, registry_path)
    if local and local.is_file():
        return local
    if registry_path and is_storage_object_key(registry_path):
        return download_to_temp_file(registry_path)
    if storage_enabled():
        for ext in (".pdf", ".docx"):
            key = source_object_key(document_id, ext)
            if object_exists(key):
                return download_to_temp_file(key, suffix=ext)
    return None


def persist_source_file(
    document_id: str,
    local_path: Union[str, Path],
) -> Optional[str]:
    """
    Upload source PDF/DOCX to storage. Returns object key for documents.pdf_path.
    """
    path = Path(local_path)
    if not path.is_file():
        return None
    if not storage_enabled():
        return str(path) if getattr(settings, "STORE_PDF_AFTER_INGEST", True) else None
    key = source_object_key(document_id, path.suffix)
    if upload_local_file(key, path):
        return key
    return str(path)


def sync_extracted_images_to_storage(
    document_id: str,
    upload_dir: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Upload local ``uploads/images/{document_id}/`` crops to the bucket."""
    if not storage_enabled():
        return
    img_dir = Path(upload_dir) / "images" / document_id
    if not img_dir.is_dir():
        return
    for png in sorted(img_dir.glob("page_*.*")):
        key = image_object_key(document_id, png.name)
        upload_local_file(key, png)
    if blocks:
        for block in blocks:
            if block.get("block_type") != "image":
                continue
            meta = block.setdefault("image_meta", {})
            name = meta.get("name") or ""
            fname = Path(str(name)).name
            if fname.startswith("page_"):
                meta["name"] = image_object_key(document_id, fname)


def upload_document_assets(
    document_id: str,
    local_source_path: Union[str, Path],
    upload_dir: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Upload source + image crops after local ingest extraction."""
    source_key = persist_source_file(document_id, local_source_path)
    sync_extracted_images_to_storage(document_id, upload_dir, blocks)
    return source_key


def delete_document_objects(document_id: str) -> None:
    if not storage_enabled():
        return
    to_remove: List[str] = []
    for ext in (".pdf", ".docx"):
        to_remove.append(source_object_key(document_id, ext))
    try:
        for item in _storage().list(document_id) or []:
            name = (item.get("name") or "").strip()
            if not name:
                continue
            if name == "images":
                for img in _storage().list(f"{document_id}/images") or []:
                    iname = img.get("name") or ""
                    if iname:
                        to_remove.append(f"{document_id}/images/{iname}")
            else:
                to_remove.append(f"{document_id}/{name}")
    except Exception as e:
        logger.debug("blob_storage: list for delete doc=%s: %s", document_id[:8], e)
    to_remove = list(dict.fromkeys(to_remove))
    if not to_remove:
        return
    try:
        _storage().remove(to_remove)
        logger.info("blob_storage: removed %s objects for doc=%s", len(to_remove), document_id[:8])
    except Exception as e:
        logger.warning("blob_storage: delete doc=%s: %s", document_id[:8], e)


def resolve_image_path_for_serving(
    document_id: str,
    image_path_or_key: Optional[str],
    page_number: Optional[int] = None,
) -> Optional[Path]:
    """Local path for serving an image (disk or temp download from storage)."""
    if image_path_or_key:
        if is_storage_object_key(image_path_or_key):
            return download_to_temp_file(image_path_or_key)
        p = Path(image_path_or_key)
        if p.is_file():
            return p
        fname = p.name
        if fname.startswith("page_"):
            local = find_disk_page_image_local(document_id, _page_from_filename(fname))
            if local:
                return local
            if storage_enabled():
                key = image_object_key(document_id, fname)
                return download_to_temp_file(key)
    if page_number is not None:
        local = find_disk_page_image_local(document_id, page_number)
        if local:
            return local
        if storage_enabled():
            key = find_page_image_key(document_id, int(page_number))
            if key:
                return download_to_temp_file(key)
    return None


def _page_from_filename(name: str) -> int:
    m = _PAGE_FILE_RE.search(name)
    return int(m.group(1)) if m else -1
