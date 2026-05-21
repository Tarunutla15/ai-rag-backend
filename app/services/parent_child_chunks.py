"""
Hierarchical parent–child chunks (Small-to-Big retrieval).

At ingest: build section-level parent text; link each searchable child via parent_chunk_id.
At query: vector/keyword hit a child; LLM receives parent section context (not isolated snippet).
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

CHUNK_TYPE_SECTION_PARENT = "section_parent"

# Child types indexed for retrieval (embedded + FTS)
_SEARCHABLE_CHILD_TYPES = frozenset({
    "paragraph",
    "heading",
    "list",
    "table_summary",
    "code_summary",
    "image_caption",
})


def is_searchable_child_chunk(meta: Dict[str, Any]) -> bool:
    """True if this chunk should be embedded and keyword-indexed (not a section parent)."""
    ct = (meta.get("chunk_type") or "paragraph").lower().strip()
    return ct in _SEARCHABLE_CHILD_TYPES


def _section_group_key(meta: Dict[str, Any]) -> str:
    """
    Stable key to group children under one parent section.

    Prefer explicit section path / PDF section_title + page so we do not collapse
    an entire document into a single parent when the tree root is the only node id.
    """
    path = meta.get("section_path")
    if isinstance(path, list) and path:
        return "path:" + " > ".join(str(p).strip() for p in path if p)

    title = (meta.get("section_title") or meta.get("parent_section_title") or "").strip()
    page = meta.get("page_number")
    try:
        page_num = int(page) if page is not None and int(page) > 0 else None
    except (TypeError, ValueError):
        page_num = None

    if title and page_num:
        return f"page:{page_num}:sec:{title[:240]}"
    if title:
        return f"sec:{title[:240]}"
    if page_num:
        return f"page:{page_num}"

    sid = meta.get("parent_section_id") or meta.get("parent_node_id")
    if sid:
        return f"node:{sid}"

    order = meta.get("order_index")
    if order is not None:
        try:
            bucket = int(order) // 8
            return f"order_bucket:{bucket}"
        except (TypeError, ValueError):
            pass

    return "_document"


def _format_section_header(meta: Dict[str, Any]) -> str:
    path = meta.get("section_path")
    if isinstance(path, list) and path:
        return " > ".join(str(p).strip() for p in path if p)
    return (
        (meta.get("parent_section_title") or meta.get("section_title") or "").strip()
        or "(section)"
    )


def _cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n\n[section truncated]"


def build_section_parents(
    chunks: List[str],
    metadata_list: List[Dict[str, Any]],
    *,
    document_id: str,
    file_name: str = "",
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Group searchable children by section, build parent bodies, set parent_chunk_id on children.

    Returns:
        (parent_texts, parent_metadata) — append to chunk lists before chunk_store.insert_chunks.
        Parents are NOT embedded in Zilliz (chunk_type=section_parent).
    """
    if not chunks or len(chunks) != len(metadata_list):
        return [], []

    max_parent_chars = int(getattr(settings, "PARENT_SECTION_MAX_CHARS", 12000))

    # index -> (text, meta) for searchable children only
    indexed: List[Tuple[int, str, Dict[str, Any]]] = []
    for i, (text, meta) in enumerate(zip(chunks, metadata_list)):
        if not is_searchable_child_chunk(meta):
            continue
        indexed.append((i, (text or "").strip(), meta))

    if not indexed:
        return [], []

    groups: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {}
    for item in indexed:
        key = _section_group_key(item[2])
        groups.setdefault(key, []).append(item)

    parent_texts: List[str] = []
    parent_metas: List[Dict[str, Any]] = []

    for group_key, items in groups.items():
        items.sort(
            key=lambda x: (
                int(x[2].get("order_index") if x[2].get("order_index") is not None else x[0]),
                x[0],
            )
        )
        parent_id = str(uuid.uuid4())
        child_ids: List[str] = []
        parts: List[str] = []

        header = _format_section_header(items[0][2])
        if header and header != "(section)":
            parts.append(f"## {header}")

        for _idx, text, meta in items:
            cid = meta.get("chunk_id")
            if cid:
                child_ids.append(cid)
                meta["parent_chunk_id"] = parent_id
            if not text:
                continue
            ct = (meta.get("chunk_type") or "paragraph").lower()
            if ct == "heading":
                parts.append(f"### {text}")
            elif ct == "table_summary":
                parts.append(f"[Table]\n{text}")
            elif ct == "code_summary":
                parts.append(f"[Code]\n{text}")
            elif ct == "image_caption":
                parts.append(f"[Figure]\n{text}")
            else:
                parts.append(text)

        body = _cap_text("\n\n".join(parts), max_parent_chars)
        if not body.strip():
            continue

        sample = items[0][2]
        positive_pages = [
            int(m.get("page_number"))
            for _, _, m in items
            if m.get("page_number") is not None and int(m.get("page_number", -1)) > 0
        ]
        page_number = min(positive_pages) if positive_pages else sample.get("page_number", -1)

        parent_meta: Dict[str, Any] = {
            "chunk_id": parent_id,
            "chunk_type": CHUNK_TYPE_SECTION_PARENT,
            "document_id": document_id,
            "file_id": document_id,
            "file_name": file_name,
            "parent_section_id": sample.get("parent_section_id"),
            "parent_node_id": sample.get("parent_node_id"),
            "parent_section_title": sample.get("parent_section_title") or sample.get("section_title"),
            "section_title": sample.get("section_title") or sample.get("parent_section_title"),
            "section_path": sample.get("section_path"),
            "page_number": page_number,
            "child_chunk_ids": child_ids,
            "section_group_key": group_key,
            "is_parent": True,
        }
        if parent_meta.get("section_path") and isinstance(parent_meta["section_path"], list):
            parent_meta["section_path_json"] = json.dumps(
                parent_meta["section_path"], ensure_ascii=False
            )

        parent_texts.append(body)
        parent_metas.append(parent_meta)

    logger.info(
        "Parent-child ingest: document_id=%s parents=%s children_linked=%s",
        (document_id or "")[:8],
        len(parent_texts),
        sum(1 for m in metadata_list if m.get("parent_chunk_id")),
    )
    return parent_texts, parent_metas


def split_searchable_for_index(
    chunks: List[str],
    metadata_list: List[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    After build_section_parents, return only child rows for embedding / FTS / Zilliz.
    """
    search_chunks: List[str] = []
    search_metadata: List[Dict[str, Any]] = []
    search_chunk_ids: List[str] = []
    for text, meta in zip(chunks, metadata_list):
        if not is_searchable_child_chunk(meta):
            continue
        search_chunks.append(text)
        search_metadata.append(meta)
        search_chunk_ids.append(meta.get("chunk_id") or "")
    return search_chunks, search_metadata, search_chunk_ids
