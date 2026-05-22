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


def list_image_keys(document_id: str) -> List[str]:
    """List object keys under ``{document_id}/images/`` in Supabase Storage."""
    if not storage_enabled():
        return []
    prefix = f"{document_id}/images"
    keys: List[str] = []
    try:
        for item in _storage().list(prefix) or []:
            name = (item.get("name") or "").strip()
            if name and "." in name:
                keys.append(f"{prefix}/{name}")
    except Exception as e:
        logger.debug("blob_storage: list_image_keys %s: %s", prefix, e)
    return sorted(keys)


def list_objects_under(prefix: str) -> List[str]:
    """List files under a storage prefix (e.g. ``{document_id}/images``)."""
    if not storage_enabled():
        return []
    base = prefix.strip().strip("/")
    parts = base.split("/")
    if parts and parts[-1] == "images":
        doc_id = "/".join(parts[:-1]) if len(parts) > 1 else (parts[0] if parts else "")
        if doc_id:
            return list_image_keys(doc_id)
    keys: List[str] = []
    try:
        for item in _storage().list(base) or []:
            name = (item.get("name") or "").strip()
            if name and "." in name:
                keys.append(f"{base}/{name}")
    except Exception as e:
        logger.debug("blob_storage: list_objects_under %s: %s", base, e)
    return keys


def find_page_image_key(
    document_id: str,
    page_number: int,
    *,
    basename: Optional[str] = None,
) -> Optional[str]:
    """Match ``page_{N}_*.png`` in Storage; prefer exact ``basename`` when given."""
    try:
        page = int(page_number)
    except (TypeError, ValueError):
        return None
    if page <= 0:
        return None
    keys = list_image_keys(document_id)
    if basename:
        bn = Path(basename.replace("\\", "/")).name
        for key in keys:
            if Path(key).name == bn:
                return key
    for key in keys:
        if Path(key).name.startswith(f"page_{page}_"):
            return key
    return None


def _media_type_for_filename(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    return "application/octet-stream"


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
        path_norm = str(image_path_or_key).strip().replace("\\", "/")
        if is_storage_object_key(path_norm):
            return download_to_temp_file(path_norm)
        fname = Path(path_norm).name
        if fname.startswith("page_"):
            if storage_enabled():
                key = find_page_image_key(
                    document_id, _page_from_filename(fname), basename=fname
                ) or image_object_key(document_id, fname)
                tmp = download_to_temp_file(key)
                if tmp:
                    return tmp
            local = find_disk_page_image_local(
                document_id, _page_from_filename(fname)
            )
            if local:
                return local
        p = Path(image_path_or_key)
        if p.is_file():
            return p
    if page_number is not None:
        try:
            page_i = int(page_number)
        except (TypeError, ValueError):
            page_i = -1
        if page_i > 0:
            if storage_enabled():
                key = find_page_image_key(document_id, page_i)
                if key:
                    tmp = download_to_temp_file(key)
                    if tmp:
                        return tmp
            local = find_disk_page_image_local(document_id, page_i)
            if local:
                return local
    return None


def storage_key_for_raw_image_id(document_id: str, image_id: str) -> Optional[str]:
    """
    Map ``raw_images.image_id`` (UUID) to the exact Storage object ``page_{N}_{idx}.png``.

    When ``raw_images`` rows are missing, align image_caption chunks on the same PDF page
    (sorted by chunk_id) with sorted ``page_{N}_*.png`` keys so two crops on page 4 do not
    both resolve to ``page_4_0.png``.
    """
    import json

    from app.services.chunk_store import get_chunk_store
    from app.services.raw_block_store import get_raw_block_store

    did = (document_id or "").strip()
    iid = (image_id or "").strip()
    if not did or not iid:
        return None

    row = get_raw_block_store().get_image(iid)
    if row:
        path_str = (row.get("image_path") or "").strip().replace("\\", "/")
        if path_str:
            if is_storage_object_key(path_str):
                return path_str
            bn = Path(path_str).name
            if bn.startswith("page_") and storage_enabled():
                key = find_page_image_key(
                    did, _page_from_filename(bn), basename=bn
                )
                if key:
                    return key

    by_page: Dict[int, List[tuple]] = {}
    try:
        chunks = get_chunk_store().list_chunks_by_type(did, "image_caption", limit=80)
    except Exception:
        chunks = []
    for ch in chunks:
        try:
            meta = json.loads(ch.get("metadata_json") or "{}")
        except Exception:
            meta = {}
        rid = (meta.get("raw_image_id") or "").strip()
        if not rid:
            continue
        try:
            page = int(ch.get("page_number") or 0)
        except (TypeError, ValueError):
            continue
        if page < 1:
            continue
        cid = str(ch.get("chunk_id") or "")
        by_page.setdefault(page, []).append((cid, rid))

    for page, pairs in by_page.items():
        pairs.sort(key=lambda x: x[0])
        rids = [r for _, r in pairs]
        if iid not in rids:
            continue
        idx = rids.index(iid)
        page_keys = sorted(
            k
            for k in list_image_keys(did)
            if Path(k).name.startswith(f"page_{page}_")
        )
        if idx < len(page_keys):
            return page_keys[idx]
    return None


def resolve_image_bytes_for_serving(
    document_id: str,
    image_id: str,
) -> Optional[tuple]:
    """
    Resolve image bytes for GET /documents/{id}/images/{image_id}.

    Storage layout: ``{document_id}/images/page_{page}_{idx}.png``
    DB ``raw_images.image_id`` is a UUID; ``image_path`` may be a local path from ingest.
    """
    from app.services.raw_block_store import get_raw_block_store

    did = (document_id or "").strip()
    iid = (image_id or "").strip()
    if not did or not iid:
        return None

    key = storage_key_for_raw_image_id(did, iid)
    if key and storage_enabled():
        data = download_bytes(key)
        if data:
            name = Path(key).name
            return data, _media_type_for_filename(name), name

    row = get_raw_block_store().get_image(iid)
    basename: Optional[str] = None
    page: Optional[int] = None
    if row:
        page = row.get("page_number")
        path_str = (row.get("image_path") or "").strip()
        if path_str:
            path_norm = path_str.replace("\\", "/")
            basename = Path(path_norm).name
            if is_storage_object_key(path_norm):
                data = download_bytes(path_norm)
                if data:
                    return data, _media_type_for_filename(basename), basename
            p = Path(path_str)
            if p.is_file():
                return p.read_bytes(), _media_type_for_filename(p.name), p.name
            if basename.startswith("page_") and storage_enabled():
                exact = find_page_image_key(
                    did, _page_from_filename(basename), basename=basename
                )
                if exact:
                    data = download_bytes(exact)
                    if data:
                        return data, _media_type_for_filename(basename), basename

    if page is None:
        try:
            from app.services.chunk_store import get_chunk_store

            for chunk in get_chunk_store().list_chunks_by_type(
                did, "image_caption", limit=80
            ):
                meta: Dict[str, Any] = {}
                try:
                    import json

                    meta = json.loads(chunk.get("metadata_json") or "{}")
                except Exception:
                    meta = {}
                if meta.get("raw_image_id") != iid:
                    continue
                page = chunk.get("page_number")
                break
        except Exception:
            pass

    if page is not None:
        try:
            page_i = int(page)
        except (TypeError, ValueError):
            page_i = -1
        if page_i > 0:
            if storage_enabled():
                key = (
                    storage_key_for_raw_image_id(did, iid)
                    or find_page_image_key(did, page_i, basename=basename)
                )
                if key:
                    data = download_bytes(key)
                    if data:
                        name = Path(key).name
                        return data, _media_type_for_filename(name), name
            local = find_disk_page_image_local(did, page_i)
            if local:
                return (
                    local.read_bytes(),
                    _media_type_for_filename(local.name),
                    local.name,
                )

    logger.debug(
        "blob_storage: no image bytes doc=%s image_id=%s page=%s basename=%s storage_keys=%s",
        did[:8],
        iid[:8],
        page,
        basename,
        list_image_keys(did)[:5],
    )
    return None


def _page_from_filename(name: str) -> int:
    m = _PAGE_FILE_RE.search(name)
    return int(m.group(1)) if m else -1
