"""
Phase 2: explicit cross-type links (table / figure / code ↔ prose).

Proximity rules applied after chunk_ids exist. Optional LLM pass for ambiguous
sections (multiple figures on one page).
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

EDGE_SUPPORTS = "supports"
EDGE_ILLUSTRATES = "illustrates"
EDGE_BELONGS_TO = "belongs_to"
EDGE_LLM_INFERRED = "llm_inferred"

_CHUNK_PARAGRAPH = frozenset({"paragraph", "heading"})
_CHUNK_TABLE = "table_summary"
_CHUNK_CODE = "code_summary"
_CHUNK_IMAGE = "image_caption"


def _section_key(meta: Dict[str, Any]) -> Optional[str]:
    return meta.get("parent_section_id") or meta.get("parent_node_id")


def _dedupe_ids(ids: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in ids:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _append_related(meta: Dict[str, Any], chunk_id: str) -> None:
    if not chunk_id:
        return
    existing = list(meta.get("related_chunk_ids") or [])
    if chunk_id not in existing:
        existing.append(chunk_id)
    meta["related_chunk_ids"] = existing


def _make_edge(document_id: str, from_id: str, to_id: str, edge_type: str) -> Dict[str, str]:
    return {
        "edge_id": str(uuid.uuid4()),
        "document_id": document_id,
        "from_node_id": from_id,
        "to_node_id": to_id,
        "edge_type": edge_type,
    }


def apply_cross_type_chunk_links(
    document_id: str,
    metadata_list: List[Dict[str, Any]],
    blocks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    """
    Apply deterministic proximity links to chunk metadata and return new graph edges.

    - table_summary → last paragraph chunk in same section (supports)
    - image_caption → section (belongs_to) + same-page prose chunks (illustrates)
    - code_summary → section (belongs_to) + previous paragraph if present (supports)
    """
    if not document_id or not metadata_list:
        return []

    block_id_to_chunk_ids: Dict[str, List[str]] = defaultdict(list)
    for meta in metadata_list:
        bid = meta.get("source_block_id")
        cid = meta.get("chunk_id")
        if bid and cid:
            block_id_to_chunk_ids[bid].append(cid)

    last_para_by_section: Dict[str, Dict[str, str]] = {}
    last_para_chunk_id: Optional[str] = None
    last_para_node_id: Optional[str] = None
    page_para_chunk_ids: Dict[int, List[str]] = defaultdict(list)
    new_edges: List[Dict[str, str]] = []

    for meta in metadata_list:
        chunk_id = meta.get("chunk_id")
        chunk_type = (meta.get("chunk_type") or "paragraph").lower()
        node_id = meta.get("node_id")
        section_id = _section_key(meta)
        page = int(meta.get("page_number") or -1)

        if chunk_type in _CHUNK_PARAGRAPH and chunk_id:
            last_para_chunk_id = chunk_id
            last_para_node_id = node_id
            if section_id:
                last_para_by_section[section_id] = {
                    "chunk_id": chunk_id,
                    "node_id": node_id,
                }
            if page > 0:
                page_para_chunk_ids[page].append(chunk_id)

        elif chunk_type == _CHUNK_TABLE and chunk_id:
            para = last_para_by_section.get(section_id) if section_id else None
            if not para and last_para_chunk_id:
                para = {"chunk_id": last_para_chunk_id, "node_id": last_para_node_id}

            if para and para.get("chunk_id"):
                meta["related_paragraph_chunk_id"] = para["chunk_id"]
                _append_related(meta, para["chunk_id"])
                if node_id and para.get("node_id"):
                    new_edges.append(
                        _make_edge(document_id, node_id, para["node_id"], EDGE_SUPPORTS)
                    )

        elif chunk_type == _CHUNK_CODE and chunk_id:
            if section_id:
                meta["related_section_node_id"] = section_id
                if node_id:
                    new_edges.append(
                        _make_edge(document_id, node_id, section_id, EDGE_BELONGS_TO)
                    )
            if last_para_chunk_id and last_para_node_id:
                _append_related(meta, last_para_chunk_id)
                if node_id:
                    new_edges.append(
                        _make_edge(document_id, node_id, last_para_node_id, EDGE_SUPPORTS)
                    )

        elif chunk_type == _CHUNK_IMAGE and chunk_id:
            if section_id:
                meta["related_section_node_id"] = section_id
                if node_id:
                    new_edges.append(
                        _make_edge(document_id, node_id, section_id, EDGE_BELONGS_TO)
                    )

            related_para_ids: List[str] = []
            ctx_block_id = meta.get("page_context_block_id")
            if ctx_block_id and block_id_to_chunk_ids.get(ctx_block_id):
                related_para_ids.extend(block_id_to_chunk_ids[ctx_block_id])
            elif page > 0:
                related_para_ids.extend(page_para_chunk_ids.get(page, []))

            related_para_ids = _dedupe_ids(related_para_ids)
            if related_para_ids:
                meta["related_paragraph_chunk_ids"] = related_para_ids
                for pid in related_para_ids:
                    _append_related(meta, pid)

            if node_id and related_para_ids:
                para_node = _node_id_for_chunk(metadata_list, related_para_ids[-1])
                if para_node:
                    new_edges.append(
                        _make_edge(document_id, node_id, para_node, EDGE_ILLUSTRATES)
                    )

    logger.info(
        "Phase 2 cross-type links: document_id=%s edges_added=%s chunks=%s",
        document_id[:8],
        len(new_edges),
        len(metadata_list),
    )
    return new_edges


def _node_id_for_chunk(metadata_list: List[Dict[str, Any]], chunk_id: str) -> Optional[str]:
    for meta in metadata_list:
        if meta.get("chunk_id") == chunk_id:
            return meta.get("node_id")
    return None


def _ambiguous_figure_sections(
    metadata_list: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Sections with 2+ image_caption chunks on the same page → LLM disambiguation."""
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for meta in metadata_list:
        if (meta.get("chunk_type") or "").lower() != _CHUNK_IMAGE:
            continue
        section_id = _section_key(meta) or "unknown"
        page = int(meta.get("page_number") or -1)
        buckets[(section_id, page)].append(meta)

    ambiguous: Dict[str, List[Dict[str, Any]]] = {}
    for (section_id, _page), items in buckets.items():
        if len(items) >= 2:
            ambiguous.setdefault(section_id, []).extend(items)
    return ambiguous


def infer_llm_cross_links(
    document_id: str,
    metadata_list: List[Dict[str, Any]],
    llm_service=None,
) -> List[Dict[str, str]]:
    """
    Optional LLM pass: when multiple figures share a page in one section,
    infer which figure illustrates which paragraph/table summary.
    """
    if not getattr(settings, "ENABLE_LLM_CROSS_TYPE_LINKS", False):
        return []
    if llm_service is None:
        try:
            from app.config import get_completion_client_config
            from app.services.llm import LLMService

            key, model, base_url = get_completion_client_config()
            llm_service = LLMService(api_key=key, model=model, base_url=base_url)
        except Exception as e:
            logger.warning("LLM cross-type links skipped (no LLM): %s", e)
            return []

    ambiguous = _ambiguous_figure_sections(metadata_list)
    if not ambiguous:
        return []

    para_by_section: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for meta in metadata_list:
        ct = (meta.get("chunk_type") or "").lower()
        if ct not in ("paragraph", "table_summary"):
            continue
        sid = _section_key(meta) or ""
        if not sid or not meta.get("chunk_id"):
            continue
        preview = (meta.get("section_title") or "")[:80]
        para_by_section[sid].append(
            {
                "chunk_id": meta["chunk_id"],
                "node_id": meta.get("node_id"),
                "chunk_type": ct,
                "preview": preview or (meta.get("retrieval_preview") or "")[:200],
            }
        )

    edges: List[Dict[str, str]] = []
    for section_id, figure_metas in ambiguous.items():
        prose_items = para_by_section.get(section_id, [])
        if not prose_items:
            continue

        section_title = (figure_metas[0].get("section_title") or figure_metas[0].get("parent_section_title") or "")[:200]
        figures_payload = []
        for fm in figure_metas:
            figures_payload.append(
                {
                    "figure_chunk_id": fm.get("chunk_id"),
                    "figure_node_id": fm.get("node_id"),
                    "page": fm.get("page_number"),
                    "caption_preview": (fm.get("caption") or "")[:300],
                }
            )
        prose_payload = [
            {
                "chunk_id": p["chunk_id"],
                "chunk_type": p["chunk_type"],
                "preview": p.get("preview", "")[:300],
            }
            for p in prose_items[:12]
        ]

        prompt = f"""You link PDF figures to the prose they illustrate.

Section title: {section_title or "(untitled)"}

Figures (same page, ambiguous):
{json.dumps(figures_payload, ensure_ascii=False)}

Paragraph/table chunks in this section:
{json.dumps(prose_payload, ensure_ascii=False)}

Return ONE JSON object only (no markdown fences):
{{"links": [{{"figure_chunk_id": "...", "paragraph_chunk_id": "...", "confidence": 0.0-1.0}}]}}

Rules:
- Each figure_chunk_id must appear at most once.
- paragraph_chunk_id must be from the prose list.
- Only include links you are reasonably confident about."""

        try:
            raw = llm_service.generate_response_simple(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
            )
        except Exception as e:
            logger.warning("LLM cross-type inference failed section=%s: %s", section_id[:8], e)
            continue

        links = _parse_llm_links(raw)
        chunk_to_node = {m.get("chunk_id"): m.get("node_id") for m in metadata_list if m.get("chunk_id")}

        for link in links:
            fig_cid = link.get("figure_chunk_id")
            para_cid = link.get("paragraph_chunk_id")
            if not fig_cid or not para_cid:
                continue
            fig_meta = next((m for m in figure_metas if m.get("chunk_id") == fig_cid), None)
            if not fig_meta:
                continue
            _append_related(fig_meta, para_cid)
            existing = list(fig_meta.get("related_paragraph_chunk_ids") or [])
            if para_cid not in existing:
                fig_meta["related_paragraph_chunk_ids"] = existing + [para_cid]
            fig_nid = fig_meta.get("node_id") or chunk_to_node.get(fig_cid)
            para_nid = chunk_to_node.get(para_cid)
            if fig_nid and para_nid:
                edges.append(
                    _make_edge(document_id, fig_nid, para_nid, EDGE_LLM_INFERRED)
                )

    if edges:
        logger.info(
            "Phase 2 LLM cross-type links: document_id=%s edges=%s",
            document_id[:8],
            len(edges),
        )
    return edges


def _parse_llm_links(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    links = data.get("links")
    return links if isinstance(links, list) else []


def build_phase2_graph_links(
    document_id: str,
    metadata_list: List[Dict[str, Any]],
    blocks: Optional[List[Dict[str, Any]]] = None,
    llm_service=None,
) -> List[Dict[str, str]]:
    """Run deterministic links, then optional LLM pass. Returns edges to merge into document_edges."""
    edges = apply_cross_type_chunk_links(document_id, metadata_list, blocks=blocks)
    edges.extend(infer_llm_cross_links(document_id, metadata_list, llm_service=llm_service))
    return edges
