"""
Query-time parent context resolution (retrieve child → return parent to LLM).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Set

from app.config import settings
from app.services.chunk_store import get_chunk_store
from app.services.parent_child_chunks import CHUNK_TYPE_SECTION_PARENT

logger = logging.getLogger(__name__)

# Always pass through vision captions; never collapse into one mega-parent block.
_ALWAYS_KEEP_CHUNK_TYPES = frozenset({"image_caption"})


def _parse_meta(chunk: Dict[str, Any]) -> Dict[str, Any]:
    if chunk.get("metadata") and isinstance(chunk["metadata"], dict):
        return chunk["metadata"]
    mj = chunk.get("metadata_json")
    if not mj:
        return {}
    try:
        return json.loads(mj) if isinstance(mj, str) else (mj or {})
    except Exception:
        return {}


def _parse_parent_meta(parent_row: Dict[str, Any]) -> Dict[str, Any]:
    return _parse_meta(parent_row)


def _chunk_score(chunk: Dict[str, Any]) -> float:
    return float(chunk.get("score") or chunk.get("hybrid_score") or 0.0)


def _cap_parent_text(text: str) -> str:
    max_chars = int(getattr(settings, "PARENT_CONTEXT_MAX_CHARS", 10000))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 24].rstrip() + "\n\n[context truncated]"


def _is_mega_parent(parent_row: Dict[str, Any]) -> bool:
    """Document-wide parent: expanding it hides diverse hits (e.g. whole PDF in one parent)."""
    threshold = int(getattr(settings, "PARENT_MEGA_CHILD_THRESHOLD", 12))
    meta = _parse_parent_meta(parent_row)
    children = meta.get("child_chunk_ids") or []
    if isinstance(children, list) and len(children) >= threshold:
        return True
    if meta.get("section_group_key") == "_document":
        return True
    body = (parent_row.get("retrieval_text") or "").strip()
    if len(body) >= int(getattr(settings, "PARENT_SECTION_MAX_CHARS", 12000)) * 0.85:
        return True
    return False


def resolve_parent_context(context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Replace or prepend child snippet with section parent text.

    - Dedupes by parent_chunk_id (one block per section parent), up to PARENT_CONTEXT_MAX_PARENTS.
    - Skips mega-parent merge (keeps child hits) when ingest grouped the whole doc.
    - Always keeps image_caption chunks (Groq vision text) even if same parent.
    """
    if not context_chunks:
        return context_chunks
    if not getattr(settings, "ENABLE_PARENT_CHILD_RETRIEVAL", True):
        return context_chunks

    mode = (getattr(settings, "PARENT_CONTEXT_MODE", "prepend") or "prepend").lower().strip()
    if mode not in ("replace", "prepend"):
        mode = "prepend"

    max_parents = max(1, int(getattr(settings, "PARENT_CONTEXT_MAX_PARENTS", 8)))

    parent_ids: Set[str] = set()
    for c in context_chunks:
        ct = (c.get("chunk_type") or "").lower()
        if ct == CHUNK_TYPE_SECTION_PARENT:
            continue
        meta = _parse_meta(c)
        pid = meta.get("parent_chunk_id")
        if pid:
            parent_ids.add(pid)

    if not parent_ids:
        return context_chunks

    store = get_chunk_store()
    parent_rows = store.get_chunks(list(parent_ids))

    ordered = sorted(context_chunks, key=_chunk_score, reverse=True)
    seen_parents: Set[str] = set()
    seen_keep_ids: Set[str] = set()
    result: List[Dict[str, Any]] = []

    for c in ordered:
        ct = (c.get("chunk_type") or "").lower()
        if ct == CHUNK_TYPE_SECTION_PARENT:
            continue

        meta = _parse_meta(c)
        pid = meta.get("parent_chunk_id")
        child_text = (c.get("text") or "").strip()
        child_id = c.get("chunk_id") or meta.get("chunk_id")

        # Vision / figure captions: keep child text (includes [Visual description])
        if ct in _ALWAYS_KEEP_CHUNK_TYPES:
            dedupe_key = child_id or id(c)
            if dedupe_key in seen_keep_ids:
                continue
            seen_keep_ids.add(dedupe_key)
            result.append(c)
            continue

        if not pid or pid not in parent_rows:
            if child_id and child_id not in seen_keep_ids:
                seen_keep_ids.add(child_id)
            result.append(c)
            continue

        parent_row = parent_rows[pid]

        if _is_mega_parent(parent_row):
            if child_id and child_id not in seen_keep_ids:
                seen_keep_ids.add(child_id)
                result.append(c)
            continue

        if pid in seen_parents:
            continue

        parent_text = _cap_parent_text((parent_row.get("retrieval_text") or "").strip())
        if not parent_text:
            if child_id and child_id not in seen_keep_ids:
                seen_keep_ids.add(child_id)
                result.append(c)
            continue

        if len(seen_parents) >= max_parents:
            if child_id and child_id not in seen_keep_ids:
                seen_keep_ids.add(child_id)
                result.append(c)
            continue

        seen_parents.add(pid)
        hit = dict(c)
        hit["child_chunk_id"] = child_id
        hit["child_snippet"] = child_text
        hit["parent_chunk_id"] = pid
        hit["context_source"] = "parent_section"

        if mode == "replace":
            hit["text"] = parent_text
        else:
            excerpt = child_text[:1200] + ("..." if len(child_text) > 1200 else "")
            hit["text"] = (
                f"### Section context\n{parent_text}\n\n"
                f"### Matched excerpt\n{excerpt}"
            )

        result.append(hit)

    if len(result) != len(context_chunks):
        logger.info(
            ">>> PARENT-CHILD: resolved %s context blocks from %s hits (%s parents expanded, %s passthrough/mega)",
            len(result),
            len(context_chunks),
            len(seen_parents),
            len(result) - len(seen_parents),
        )
    return result if result else context_chunks
