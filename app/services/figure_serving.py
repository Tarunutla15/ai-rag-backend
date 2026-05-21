"""Serve and resolve cropped PDF figures from disk + Supabase raw_images."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings

_PAGE_FILE_RE = re.compile(r"page_(\d+)_", re.I)


def disk_image_dir(document_id: str) -> Path:
    return Path(settings.UPLOAD_DIR) / "images" / document_id


def find_disk_page_image(document_id: str, page_number: int) -> Optional[Path]:
    """Find page image on local disk or download from Supabase Storage to a temp file."""
    from app.services.blob_storage import (
        find_disk_page_image_local,
        find_page_image_key,
        download_to_temp_file,
        storage_enabled,
    )

    local = find_disk_page_image_local(document_id, page_number)
    if local:
        return local
    if storage_enabled():
        key = find_page_image_key(document_id, page_number)
        if key:
            return download_to_temp_file(key)
    return None


def page_figure_exists(document_id: str, page_number: int) -> bool:
    from app.services.blob_storage import page_image_available

    return page_image_available(document_id, page_number)


def view_url_for_page(document_id: str, page_number: int) -> str:
    return f"/documents/{document_id}/images/page/{int(page_number)}"


def figure_from_page(
    document_id: str,
    page_number: int,
    *,
    caption: str = "",
    image_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not page_figure_exists(document_id, page_number):
        return None
    return {
        "image_id": image_id,
        "document_id": document_id,
        "view_url": view_url_for_page(document_id, page_number),
        "caption": (caption or f"Figure on page {page_number}")[:500],
        "page_number": int(page_number),
    }


def resolve_figure_view_url(
    document_id: str,
    *,
    image_id: Optional[str] = None,
    page_number: Optional[int] = None,
    existing_view_url: Optional[str] = None,
    image_path: Optional[str] = None,
) -> Optional[str]:
    """
    Prefer a working page-based URL (disk file exists). UUID routes often 404 when
    raw_images rows were purged but crops remain under uploads/images/.
    """
    did = (document_id or "").strip()
    if not did:
        return None

    existing = (existing_view_url or "").strip()
    if existing.startswith(f"/documents/{did}/images/page/"):
        try:
            page = int(existing.rsplit("/page/", 1)[-1])
            if find_disk_page_image(did, page):
                return existing
        except (ValueError, IndexError):
            pass

    if page_number is not None:
        try:
            page = int(page_number)
            if page > 0 and page_figure_exists(did, page):
                return view_url_for_page(did, page)
        except (TypeError, ValueError):
            pass

    if image_path:
        from app.services.blob_storage import is_storage_object_key

        if is_storage_object_key(image_path):
            m = _PAGE_FILE_RE.search(Path(image_path).name)
            if m and page_figure_exists(did, int(m.group(1))):
                return view_url_for_page(did, int(m.group(1)))
        p = Path(image_path)
        if p.is_file():
            m = _PAGE_FILE_RE.search(p.name)
            if m:
                page = int(m.group(1))
                if page_figure_exists(did, page):
                    return view_url_for_page(did, page)

    if existing.startswith(f"/documents/{did}/images/") and "/page/" not in existing:
        return existing

    return None


def build_figure_dict(
    *,
    document_id: str,
    page_number: Optional[int] = None,
    image_id: Optional[str] = None,
    caption: str = "",
    existing_view_url: Optional[str] = None,
    image_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build a figure payload with a view_url that actually serves from disk when possible."""
    did = (document_id or "").strip()
    if not did:
        return None
    view_url = resolve_figure_view_url(
        did,
        image_id=image_id,
        page_number=page_number,
        existing_view_url=existing_view_url,
        image_path=image_path,
    )
    if not view_url:
        return None
    return {
        "image_id": image_id,
        "document_id": did,
        "view_url": view_url,
        "caption": (caption or "Figure from document")[:500],
        "page_number": page_number,
    }


def _query_figure_relevance_score(text: str, query: str) -> float:
    """Score how well chunk/caption text matches the user question."""
    text_l = (text or "").lower()
    q = (query or "").lower()
    score = 0.05
    for term in re.findall(r"[a-z][a-z0-9-]{3,}", q):
        if term in text_l:
            score += 0.45
    if "visual description" in text_l:
        score += 0.2
    if "diagram" in q and "diagram" in text_l:
        score += 0.15
    if "architecture" in q and "architecture" in text_l:
        score += 0.35
    if "transformer" in q and "transformer" in text_l:
        score += 0.4
    if "attention" in q and "attention" in text_l:
        score += 0.45
    if "scaled" in q and "scaled" in text_l:
        score += 0.35
    if "dot" in q and "dot" in text_l:
        score += 0.2
    if "product" in q and "product" in text_l:
        score += 0.25
    if "lifecycle" in q and "lifecycle" in text_l:
        score += 0.4
    return score


def figures_from_disk(
    scope_file_ids: List[str],
    query: str = "",
    *,
    max_figures: int = 1,
) -> List[Dict[str, Any]]:
    """List cropped images on disk when raw_images DB rows are missing."""
    seen: set = set()
    figures: List[Dict[str, Any]] = []
    ranked: List[tuple] = []

    from app.services.blob_storage import list_objects_under, storage_enabled

    for did in scope_file_ids:
        if not did:
            continue
        base = disk_image_dir(did)
        if base.is_dir():
            for path in sorted(base.glob("page_*.*")):
                m = _PAGE_FILE_RE.search(path.name)
                if not m:
                    continue
                page = int(m.group(1))
                ranked.append((_query_figure_relevance_score(f"page {page}", query), did, page, path))
        if storage_enabled():
            for key in list_objects_under(f"{did}/images"):
                fname = Path(key).name
                m = _PAGE_FILE_RE.search(fname)
                if not m:
                    continue
                page = int(m.group(1))
                ranked.append((_query_figure_relevance_score(f"page {page}", query), did, page, None))
    ranked.sort(key=lambda x: (-x[0], x[2]))
    for _score, did, page, _path in ranked:
        url = view_url_for_page(did, page)
        if url in seen:
            continue
        seen.add(url)
        fig = figure_from_page(did, page, caption=f"Figure on page {page}")
        if fig:
            figures.append(fig)
        if len(figures) >= max_figures:
            break
    return figures


def resolve_image_file_path(document_id: str, image_id: str) -> Optional[Path]:
    """
    Resolve a cropped figure file for GET /documents/{id}/images/{image_id}.
    Supports Supabase Storage keys, local disk, and page fallback.
    """
    from app.services.blob_storage import resolve_image_path_for_serving
    from app.services.raw_block_store import get_raw_block_store

    did = (document_id or "").strip()
    iid = (image_id or "").strip()
    if not did or not iid:
        return None

    row = get_raw_block_store().get_image(iid)
    if row:
        path_str = (row.get("image_path") or "").strip()
        page = row.get("page_number")
        found = resolve_image_path_for_serving(
            did, path_str or None, page_number=page
        )
        if found:
            return found

    try:
        from app.services.chunk_store import get_chunk_store

        for chunk in get_chunk_store().list_chunks_by_type(did, "image_caption", limit=80):
            meta: Dict[str, Any] = {}
            try:
                import json

                meta = json.loads(chunk.get("metadata_json") or "{}")
            except Exception:
                meta = {}
            if meta.get("raw_image_id") != iid:
                continue
            page = chunk.get("page_number")
            found = resolve_image_path_for_serving(did, None, page_number=page)
            if found:
                return found
    except Exception:
        pass

    return None


def hydrate_raw_image_fallback(
    chunk: Dict[str, Any],
    document_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build raw_image dict from disk when Supabase raw_images row is missing."""
    did = document_id or chunk.get("document_id")
    page = chunk.get("page_number")
    if not did or page is None:
        return None
    path = find_disk_page_image(did, page)
    if not path:
        return None
    meta = chunk.get("metadata") or {}
    raw_id = meta.get("raw_image_id")
    try:
        page_i = int(page)
    except (TypeError, ValueError):
        return None
    return build_figure_dict(
        document_id=did,
        page_number=page_i,
        image_id=raw_id,
        caption=(chunk.get("text") or "")[:500],
        image_path=str(path),
        existing_view_url=view_url_for_page(did, page_i),
    )
