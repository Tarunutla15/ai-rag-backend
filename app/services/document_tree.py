"""
Phase 1: semantic document tree over PDF blocks.

Builds a section hierarchy (stack-based outline), assigns stable node IDs,
and enriches blocks before chunking. Chunk sequence links (prev/next) are applied at insert time.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

NODE_DOCUMENT = "document"
NODE_SECTION = "section"
NODE_PARAGRAPH = "paragraph"
NODE_TABLE = "table"
NODE_CODE = "code"
NODE_FIGURE = "figure"

EDGE_PARENT_OF = "parent_of"
EDGE_NEXT = "next"


def infer_heading_level(title: str) -> int:
    """Estimate outline level (1 = top) from heading text."""
    t = (title or "").strip()
    if not t:
        return 2
    m = re.match(r"^(\d+(?:\.\d+)*)\s+\S", t)
    if m:
        return max(1, min(6, m.group(1).count(".") + 1))
    if len(t) <= 80 and t.isupper() and len(t) >= 4:
        return 1
    if t.endswith(":") and len(t.split()) <= 10:
        return 2
    return 2


def _block_to_node_type(block_type: str) -> str:
    bt = (block_type or "text").lower()
    if bt == "heading":
        return NODE_SECTION
    if bt == "table":
        return NODE_TABLE
    if bt == "code":
        return NODE_CODE
    if bt == "image":
        return NODE_FIGURE
    return NODE_PARAGRAPH


class DocumentTreeBuilder:
    """Turn ordered PDF blocks into a tree + enriched blocks for chunking."""

    def build(
        self,
        blocks: List[Dict[str, Any]],
        document_id: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns:
            enriched_blocks (same order, tree fields added),
            nodes (rows for document_nodes),
            edges (rows for document_edges),
        """
        if not document_id:
            raise ValueError("document_id required")

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        enriched: List[Dict[str, Any]] = []

        root_id = str(uuid.uuid4())
        nodes.append({
            "node_id": root_id,
            "document_id": document_id,
            "node_type": NODE_DOCUMENT,
            "title": "",
            "content_preview": "",
            "page_start": 1,
            "page_end": 1,
            "parent_node_id": None,
            "depth": 0,
            "order_index": 0,
            "section_path_json": "[]",
        })

        # stack: (node_id, level, title)
        section_stack: List[Tuple[str, int, str]] = [(root_id, 0, "")]
        order_index = 0
        prev_content_node_id: Optional[str] = None

        for block in blocks:
            order_index += 1
            b = dict(block)
            block_type = b.get("block_type", "text")
            page_number = int(b.get("page_number") or -1)
            content = (b.get("content") or "").strip()

            if block_type == "heading" and content:
                level = infer_heading_level(content)
                while len(section_stack) > 1 and section_stack[-1][1] >= level:
                    section_stack.pop()
                parent_id = section_stack[-1][0]
                section_id = str(uuid.uuid4())
                path_titles = [t for _, _, t in section_stack[1:] if t] + [content]
                section_path = path_titles

                nodes.append({
                    "node_id": section_id,
                    "document_id": document_id,
                    "node_type": NODE_SECTION,
                    "title": content[:500],
                    "content_preview": content[:240],
                    "page_start": page_number if page_number > 0 else 1,
                    "page_end": page_number if page_number > 0 else 1,
                    "parent_node_id": parent_id,
                    "depth": len(section_stack),
                    "order_index": order_index,
                    "section_path_json": _json_path(section_path),
                })
                edges.append({
                    "edge_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "from_node_id": parent_id,
                    "to_node_id": section_id,
                    "edge_type": EDGE_PARENT_OF,
                })
                if prev_content_node_id:
                    edges.append({
                        "edge_id": str(uuid.uuid4()),
                        "document_id": document_id,
                        "from_node_id": prev_content_node_id,
                        "to_node_id": section_id,
                        "edge_type": EDGE_NEXT,
                    })
                    prev_content_node_id = None

                section_stack.append((section_id, level, content))
                b["node_id"] = section_id
                b["parent_node_id"] = parent_id
                b["parent_section_id"] = section_id
                b["parent_section_title"] = content
                b["section_path"] = section_path
                b["tree_depth"] = len(section_stack) - 1
                b["order_index"] = order_index
                b["section_title"] = content
                enriched.append(b)
                continue

            # Content block under current section
            current_section_id = section_stack[-1][0]
            current_section_title = section_stack[-1][2] if len(section_stack[-1]) > 2 else ""
            path_titles = [t for _, _, t in section_stack[1:] if t]
            section_path = path_titles

            node_id = str(uuid.uuid4())
            ntype = _block_to_node_type(block_type)
            preview = content[:240] if content else (b.get("table_summary") or b.get("code_summary") or "")[:240]

            nodes.append({
                "node_id": node_id,
                "document_id": document_id,
                "node_type": ntype,
                "title": current_section_title[:500] if current_section_title else "",
                "content_preview": preview,
                "page_start": page_number if page_number > 0 else 1,
                "page_end": page_number if page_number > 0 else 1,
                "parent_node_id": current_section_id,
                "depth": len(section_stack),
                "order_index": order_index,
                "section_path_json": _json_path(section_path),
            })
            edges.append({
                "edge_id": str(uuid.uuid4()),
                "document_id": document_id,
                "from_node_id": current_section_id,
                "to_node_id": node_id,
                "edge_type": EDGE_PARENT_OF,
            })
            if prev_content_node_id:
                edges.append({
                    "edge_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "from_node_id": prev_content_node_id,
                    "to_node_id": node_id,
                    "edge_type": EDGE_NEXT,
                })

            b["node_id"] = node_id
            b["parent_node_id"] = current_section_id
            b["parent_section_id"] = current_section_id
            b["parent_section_title"] = current_section_title or b.get("section_title", "")
            b["section_path"] = section_path
            b["tree_depth"] = len(section_stack)
            b["order_index"] = order_index
            if not b.get("section_title") and current_section_title:
                b["section_title"] = current_section_title

            enriched.append(b)
            prev_content_node_id = node_id

        logger.info(
            "Document tree built: document_id=%s nodes=%s edges=%s blocks=%s",
            document_id[:8],
            len(nodes),
            len(edges),
            len(enriched),
        )
        return enriched, nodes, edges


def _json_path(path: List[str]) -> str:
    import json
    return json.dumps(path, ensure_ascii=False)


def apply_chunk_sequence_links(metadata_list: List[Dict[str, Any]]) -> None:
    """Set prev_chunk_id / next_chunk_id on chunk metadata after chunk_ids are assigned."""
    if not metadata_list:
        return
    for i, meta in enumerate(metadata_list):
        cid = meta.get("chunk_id")
        if i > 0:
            prev = metadata_list[i - 1].get("chunk_id")
            if prev and cid:
                meta["prev_chunk_id"] = prev
        if i < len(metadata_list) - 1:
            nxt = metadata_list[i + 1].get("chunk_id")
            if nxt and cid:
                meta["next_chunk_id"] = nxt
        if meta.get("order_index") is None:
            meta["order_index"] = i


def merge_tree_fields_into_chunk_meta(block: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Copy tree fields from enriched block onto chunk metadata."""
    for key in (
        "node_id",
        "parent_node_id",
        "parent_section_id",
        "parent_section_title",
        "section_path",
        "tree_depth",
        "order_index",
    ):
        if key in block and block[key] is not None:
            meta[key] = block[key]
    if meta.get("section_path") and isinstance(meta["section_path"], list):
        meta["section_path_json"] = _json_path(meta["section_path"])
