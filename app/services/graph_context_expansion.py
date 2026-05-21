"""
Phase 4: graph-aware context expansion at query time.

Expands seed retrieval hits with prev/next, cross-type links, and document_edges.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings
from app.services.chunk_store import get_chunk_store
from app.services.document_graph_store import get_document_graph_store
from app.services.parent_child_chunks import CHUNK_TYPE_SECTION_PARENT

logger = logging.getLogger(__name__)

EDGE_ILLUSTRATES = "illustrates"
EDGE_SUPPORTS = "supports"
EDGE_BELONGS_TO = "belongs_to"
EDGE_LLM_INFERRED = "llm_inferred"
_EDGE_CROSS = frozenset({EDGE_ILLUSTRATES, EDGE_SUPPORTS, EDGE_BELONGS_TO, EDGE_LLM_INFERRED})


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


def _seed_score(chunk: Dict[str, Any]) -> float:
    return float(chunk.get("score") or chunk.get("hybrid_score") or 0.0)


def _row_to_hit(row: Dict[str, Any], score: float, source: str) -> Dict[str, Any]:
    return {
        "chunk_id": row.get("chunk_id"),
        "document_id": row.get("document_id"),
        "chunk_type": row.get("chunk_type", "paragraph"),
        "text": row.get("retrieval_text", ""),
        "page_number": row.get("page_number", -1),
        "section_title": row.get("section_title", ""),
        "metadata_json": row.get("metadata_json", ""),
        "score": score,
        "hybrid_score": score,
        "source": source,
    }


def _collect_ids_from_meta(meta: Dict[str, Any], seen: Set[str], out: List[str], max_total: int) -> None:
    for key in ("prev_chunk_id", "next_chunk_id", "related_paragraph_chunk_id"):
        cid = meta.get(key)
        if cid and cid not in seen and len(out) < max_total:
            seen.add(cid)
            out.append(cid)
    for key in ("related_chunk_ids", "related_paragraph_chunk_ids"):
        for cid in meta.get(key) or []:
            if cid and cid not in seen and len(out) < max_total:
                seen.add(cid)
                out.append(cid)


def _edges_for_documents(document_ids: List[str]) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Return (all_edges, node_id -> incident edges)."""
    graph = get_document_graph_store()
    all_edges: List[Dict] = []
    by_node: Dict[str, List[Dict]] = {}
    for did in document_ids:
        edges = graph.list_edges(did, limit=8000)
        all_edges.extend(edges)
        for e in edges:
            et = e.get("edge_type")
            if et not in _EDGE_CROSS:
                continue
            fn, tn = e.get("from_node_id"), e.get("to_node_id")
            if fn:
                by_node.setdefault(fn, []).append(e)
            if tn:
                by_node.setdefault(tn, []).append(e)
    return all_edges, by_node


def _chunk_ids_from_graph_neighbors(
    node_id: Optional[str],
    edges_by_node: Dict[str, List[Dict]],
    node_to_chunks: Dict[str, List[str]],
    seed_node: Optional[str],
    direction: str,
) -> List[str]:
    """direction: 'out' follows from_node, 'in' follows to_node."""
    if not node_id:
        return []
    found: List[str] = []
    for e in edges_by_node.get(node_id, []):
        et = e.get("edge_type")
        if et not in _EDGE_CROSS:
            continue
        if direction == "out" and e.get("from_node_id") == node_id:
            other = e.get("to_node_id")
        elif direction == "in" and e.get("to_node_id") == node_id:
            other = e.get("from_node_id")
        else:
            continue
        for cid in node_to_chunks.get(other, []):
            if cid not in found:
                found.append(cid)
    return found


def _sibling_paragraph_ids(
    meta: Dict[str, Any],
    doc_chunks: List[Dict],
    seed_chunk_id: str,
    min_score_ratio: float,
    max_siblings: int,
) -> List[str]:
    section = meta.get("parent_section_id") or meta.get("parent_node_id")
    if not section:
        return []
    siblings: List[str] = []
    for row in doc_chunks:
        cid = row.get("chunk_id")
        if not cid or cid == seed_chunk_id:
            continue
        try:
            rm = json.loads(row.get("metadata_json") or "{}")
        except Exception:
            rm = {}
        if (rm.get("chunk_type") or "").lower() != "paragraph":
            continue
        if (rm.get("parent_section_id") or rm.get("parent_node_id")) != section:
            continue
        siblings.append(cid)
        if len(siblings) >= max_siblings:
            break
    return siblings


def pack_context_chunks(
    chunks: List[Dict[str, Any]],
    max_chars: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Keep highest-scoring chunks until character budget (seeds first by score)."""
    if not chunks:
        return []
    budget = max_chars or int(getattr(settings, "CONTEXT_EXPANSION_MAX_CHARS", 28000))
    ordered = sorted(chunks, key=_seed_score, reverse=True)
    packed: List[Dict[str, Any]] = []
    used = 0
    seen_ids: Set[str] = set()
    for c in ordered:
        cid = c.get("chunk_id")
        if cid and cid in seen_ids:
            continue
        text_len = len(c.get("text") or "")
        if used + text_len > budget and packed:
            continue
        packed.append(c)
        if cid:
            seen_ids.add(cid)
        used += text_len
        if used >= budget:
            break
    return packed


def expand_context_graph(
    context_chunks: List[Dict[str, Any]],
    *,
    query: str = "",
    max_hops: int = 1,
    query_seeks_figure: bool = False,
) -> List[Dict[str, Any]]:
    """
    Expand seed hits using chunk metadata links and document_edges.
    """
    if not context_chunks:
        return context_chunks

    decay = float(getattr(settings, "CONTEXT_EXPANSION_DECAY", 0.85))
    max_add = int(getattr(settings, "CONTEXT_EXPANSION_MAX_ADD", 14))
    sibling_margin = float(getattr(settings, "CONTEXT_EXPANSION_SIBLING_MARGIN", 0.92))

    seen: Set[str] = {c.get("chunk_id") for c in context_chunks if c.get("chunk_id")}
    seeds = list(context_chunks)
    max_seed = max((_seed_score(c) for c in seeds), default=0.25)

    to_fetch: List[str] = []
    doc_ids = list({c.get("document_id") for c in context_chunks if c.get("document_id")})

    for c in seeds:
        meta = _parse_meta(c)
        _collect_ids_from_meta(meta, seen, to_fetch, max_add)

    _, edges_by_node = _edges_for_documents(doc_ids)

    chunk_store = get_chunk_store()
    doc_chunk_rows: Dict[str, List[Dict]] = {}
    node_to_chunks: Dict[str, List[str]] = {}

    for did in doc_ids[:5]:
        rows = chunk_store.list_chunks_by_document(did, limit=400)
        doc_chunk_rows[did] = rows
        for row in rows:
            try:
                rm = json.loads(row.get("metadata_json") or "{}")
            except Exception:
                rm = {}
            nid = rm.get("node_id")
            if nid:
                node_to_chunks.setdefault(nid, []).append(row["chunk_id"])

    for c in seeds:
        meta = _parse_meta(c)
        nid = meta.get("node_id")
        ct = (meta.get("chunk_type") or c.get("chunk_type") or "").lower()

        if ct == "paragraph" or query_seeks_figure:
            for cid in _chunk_ids_from_graph_neighbors(nid, edges_by_node, node_to_chunks, nid, "in"):
                if cid not in seen and len(to_fetch) < max_add:
                    seen.add(cid)
                    to_fetch.append(cid)
            for cid in _chunk_ids_from_graph_neighbors(nid, edges_by_node, node_to_chunks, nid, "out"):
                if cid not in seen and len(to_fetch) < max_add:
                    seen.add(cid)
                    to_fetch.append(cid)

        if ct in ("table_summary", "code_summary"):
            for cid in _chunk_ids_from_graph_neighbors(nid, edges_by_node, node_to_chunks, nid, "out"):
                if cid not in seen and len(to_fetch) < max_add:
                    seen.add(cid)
                    to_fetch.append(cid)

        if ct == "image_caption":
            for cid in _chunk_ids_from_graph_neighbors(nid, edges_by_node, node_to_chunks, nid, "out"):
                if cid not in seen and len(to_fetch) < max_add:
                    seen.add(cid)
                    to_fetch.append(cid)

        section_id = meta.get("parent_section_id")
        if section_id:
            for cid in node_to_chunks.get(section_id, []):
                if cid in seen or len(to_fetch) >= max_add:
                    continue
                row = next(
                    (r for r in doc_chunk_rows.get(c.get("document_id") or "", []) if r.get("chunk_id") == cid),
                    None,
                )
                if row and (row.get("chunk_type") or "").lower() == "heading":
                    seen.add(cid)
                    to_fetch.append(cid)

        did = c.get("document_id")
        if did and _seed_score(c) >= max_seed * sibling_margin:
            for sid in _sibling_paragraph_ids(meta, doc_chunk_rows.get(did, []), c.get("chunk_id") or "", sibling_margin, 3):
                if sid not in seen and len(to_fetch) < max_add:
                    seen.add(sid)
                    to_fetch.append(sid)

    to_fetch = to_fetch[:max_add]
    if not to_fetch:
        return pack_context_chunks(context_chunks)

    rows = chunk_store.get_chunks(to_fetch)
    merged = list(context_chunks)
    for cid in to_fetch:
        row = rows.get(cid)
        if not row:
            continue
        if (row.get("chunk_type") or "").lower() == CHUNK_TYPE_SECTION_PARENT:
            continue
        best_seed = max_seed
        link_score = best_seed * decay
        merged.append(_row_to_hit(row, link_score, "graph_expand"))

    deduped: Dict[str, Dict[str, Any]] = {}
    for c in merged:
        key = c.get("chunk_id") or _chunk_key_fallback(c)
        prev = deduped.get(key)
        if prev is None or _seed_score(c) > _seed_score(prev):
            deduped[key] = c

    result = list(deduped.values())
    added = len(result) - len(context_chunks)
    if added > 0:
        logger.info(
            ">>> RETRIEVAL: graph expansion added=%s total=%s decay=%.2f",
            added,
            len(result),
            decay,
        )
    return pack_context_chunks(result)


def _chunk_key_fallback(c: Dict[str, Any]) -> str:
    return f"{c.get('document_id')}:{c.get('page_number')}:{(c.get('text') or '')[:40]}"
