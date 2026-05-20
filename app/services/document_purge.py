"""
Async document purge (Phase: scalable deletes without thread pools).

Uses asyncio.gather for concurrent I/O; blocking Supabase/SQLite calls run in asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: F401 - Callable used in _run_step

from app.config import settings
from app.services.chat_service import get_chat_service
from app.services.chunk_store import get_chunk_store
from app.services.document_graph_store import get_document_graph_store
from app.services.document_store import get_document_store
from app.services.keyword_search import get_keyword_search_service
from app.services.raw_block_store import get_raw_block_store

logger = logging.getLogger(__name__)


async def _run_step(label: str, fn: Callable[[], Any]) -> Optional[str]:
    try:
        await asyncio.to_thread(fn)
        return None
    except Exception as e:
        logger.warning("purge failed step=%s: %s", label, e)
        return f"{label}: {e}"


async def _purge_zilliz(vector_store, document_id: str) -> Optional[str]:
    try:
        await vector_store.delete_by_document_id_async(document_id)
        return None
    except Exception as e:
        logger.warning("purge failed step=zilliz: %s", e)
        return f"zilliz: {e}"


async def purge_index_data_async(document_id: str, vector_store) -> List[str]:
    """
    Delete indexed data only (sessions scope, FTS, chunks, graph, raw, Zilliz).
    Does not remove files or documents registry row.
    """
    results = await asyncio.gather(
        _run_step("session_documents", lambda: get_chat_service().remove_document_from_all_sessions(document_id)),
        _run_step("chunks_fts", lambda: get_keyword_search_service().delete_document(document_id)),
        _run_step("chunks", lambda: get_chunk_store().delete_document(document_id)),
        _run_step("document_graph", lambda: get_document_graph_store().delete_document_graph(document_id)),
        _run_step("raw_blocks", lambda: get_raw_block_store().delete_document(document_id)),
        _purge_zilliz(vector_store, document_id),
    )
    return [e for e in results if e]


async def purge_document_everywhere_async(document_id: str, vector_store) -> Dict[str, Any]:
    """Full purge: index data + local files + documents row."""
    errors: List[str] = []
    store = get_document_store()
    if not await asyncio.to_thread(store.get_document, document_id):
        return {"deleted": False, "document_id": document_id, "errors": ["Document not found"]}

    t0 = time.perf_counter()
    errors.extend(await purge_index_data_async(document_id, vector_store))
    logger.info(
        "purge_document_everywhere_async index doc=%s elapsed=%.2fs errors=%s",
        document_id[:8],
        time.perf_counter() - t0,
        len(errors),
    )

    upload_root = Path(settings.UPLOAD_DIR)

    def _unlink_files():
        for ext in (".pdf", ".docx"):
            p = upload_root / f"{document_id}{ext}"
            if p.is_file():
                p.unlink()
        img_dir = upload_root / "images" / document_id
        if img_dir.exists() and img_dir.is_dir():
            shutil.rmtree(img_dir, ignore_errors=True)

    try:
        await asyncio.to_thread(_unlink_files)
    except Exception as e:
        errors.append(f"files: {e}")

    try:
        await asyncio.to_thread(store.delete_document_row, document_id)
    except Exception as e:
        errors.append(f"documents_table: {e}")

    return {"deleted": True, "document_id": document_id, "errors": errors}


async def purge_many_documents_async(
    doc_ids: List[str],
    vector_store,
    *,
    max_concurrent: Optional[int] = None,
) -> Tuple[int, List[str]]:
    """Purge multiple documents with an asyncio.Semaphore (not threads)."""
    if not doc_ids:
        return 0, []
    limit = max_concurrent or int(getattr(settings, "DELETE_PARALLEL_WORKERS", 4))
    limit = max(1, min(limit, len(doc_ids)))
    sem = asyncio.Semaphore(limit)
    deleted_count = 0
    errors: List[str] = []

    async def _one(did: str) -> Dict[str, Any]:
        async with sem:
            return await purge_document_everywhere_async(did, vector_store)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[_one(did) for did in doc_ids])
    deleted_count = sum(1 for r in results if r.get("deleted"))
    for did, result in zip(doc_ids, results):
        for err in result.get("errors") or []:
            errors.append(f"{did}: {err}")
    logger.info(
        "purge_many_documents_async count=%s deleted=%s concurrent=%s elapsed=%.1fs",
        len(doc_ids),
        deleted_count,
        limit,
        time.perf_counter() - t0,
    )
    return deleted_count, errors
