"""
Phase 3: contextual embedding strings (structural prefix at embed time only).

retrieval_text / Zilliz `text` stay as display text; embedding_text is stored in
metadata_json and used only for vector embedding generation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Human labels for related chunks in [Related: ...] line
_TYPE_LABEL = {
    "table_summary": "Table",
    "image_caption": "Figure",
    "code_summary": "Code",
    "heading": "Section",
    "paragraph": "Paragraph",
    "list": "List",
}


def build_chunk_display_labels(metadata_list: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Assign stable labels per document order: Table 1, Figure 2, Code 1, etc.
    """
    counters: Dict[str, int] = {}
    labels: Dict[str, str] = {}
    ordered = sorted(
        enumerate(metadata_list),
        key=lambda x: (
            int((x[1].get("order_index") if x[1].get("order_index") is not None else x[0])),
            x[0],
        ),
    )
    for _idx, meta in ordered:
        cid = meta.get("chunk_id")
        if not cid:
            continue
        ct = (meta.get("chunk_type") or "paragraph").lower()
        base = _TYPE_LABEL.get(ct, "Chunk")
        counters[base] = counters.get(base, 0) + 1
        labels[cid] = f"{base} {counters[base]}"
    return labels


def _format_section_path(meta: Dict[str, Any]) -> str:
    path = meta.get("section_path")
    if isinstance(path, list) and path:
        return " > ".join(str(p).strip() for p in path if p)
    title = (meta.get("section_title") or meta.get("parent_section_title") or "").strip()
    return title


def _format_subsection_title(meta: Dict[str, Any]) -> str:
    """Innermost section/heading label (retrieval anchor)."""
    path = meta.get("section_path")
    if isinstance(path, list) and path:
        return str(path[-1]).strip()
    return (meta.get("section_title") or meta.get("parent_section_title") or "").strip()


def _format_page_line(meta: Dict[str, Any]) -> str:
    try:
        page = int(meta.get("page_number", -1))
    except (TypeError, ValueError):
        page = -1
    if page > 0:
        return f"[Page: {page}]"
    return ""


def _format_related_line(meta: Dict[str, Any], labels: Dict[str, str]) -> str:
    parts: List[str] = []
    cid = meta.get("chunk_id")

    rp = meta.get("related_paragraph_chunk_id")
    if rp and rp in labels:
        parts.append(f"{labels[rp]} (chunk {rp[:8]})")

    for rid in meta.get("related_paragraph_chunk_ids") or []:
        if rid and rid in labels and rid != rp:
            parts.append(f"{labels[rid]} (chunk {rid[:8]})")

    for rid in meta.get("related_chunk_ids") or []:
        if not rid or rid == cid or rid in (meta.get("related_paragraph_chunk_ids") or []):
            continue
        if rid in labels:
            parts.append(f"{labels[rid]} (chunk {rid[:8]})")

    if not parts:
        return ""
    return ", ".join(parts[:6])


def build_embedding_text(
    retrieval_text: str,
    meta: Dict[str, Any],
    *,
    document_title: str = "",
    chunk_labels: Optional[Dict[str, str]] = None,
    llm_context_line: Optional[str] = None,
) -> str:
    """
    Build the string sent to the embedding model (display text unchanged).
    """
    labels = chunk_labels or {}
    lines: List[str] = []

    doc_name = (document_title or meta.get("file_name") or "").strip()
    if doc_name:
        lines.append(f"[Document: {doc_name}]")

    section = _format_section_path(meta)
    if section:
        lines.append(f"[Section: {section}]")

    subsection = _format_subsection_title(meta)
    page_line = _format_page_line(meta)
    if page_line:
        lines.append(page_line)

    chunk_type = (meta.get("chunk_type") or "paragraph").lower()
    lines.append(f"[Type: {chunk_type}]")

    body = (retrieval_text or "").strip()
    if chunk_type == "heading" and body:
        lines.append(f"[Heading: {body}]")
        if subsection and subsection.lower() != body.lower():
            lines.append(f"[Subsection: {subsection}]")

    related = _format_related_line(meta, labels)
    if related:
        lines.append(f"[Related: {related}]")

    if llm_context_line and llm_context_line.strip():
        lines.append(f"[Context: {llm_context_line.strip()[:400]}]")

    if not body:
        body = "(empty)"
    lines.append(body)
    return "\n".join(lines)


def build_fts_index_text(
    retrieval_text: str,
    meta: Dict[str, Any],
    *,
    document_title: str = "",
) -> str:
    """
    Enriched text for keyword/FTS index — aligns with vector embedding context.

    Plain tokens (no brackets) so Supabase ILIKE matches section titles and document names.
    """
    parts: List[str] = []
    doc_name = (document_title or meta.get("file_name") or "").strip()
    if doc_name:
        parts.append(doc_name)

    path = meta.get("section_path")
    if isinstance(path, list):
        for segment in path:
            seg = str(segment).strip()
            if seg:
                parts.append(seg)

    section = _format_section_path(meta)
    if section and section not in parts:
        parts.append(section)

    subsection = _format_subsection_title(meta)
    if subsection and subsection not in parts:
        parts.append(subsection)

    parent_title = (meta.get("parent_section_title") or "").strip()
    if parent_title and parent_title not in parts:
        parts.append(parent_title)

    try:
        page = int(meta.get("page_number", -1))
    except (TypeError, ValueError):
        page = -1
    if page > 0:
        parts.append(f"page {page}")

    chunk_type = (meta.get("chunk_type") or "paragraph").lower()
    parts.append(chunk_type)

    body = (retrieval_text or "").strip()
    if chunk_type == "heading" and body:
        # Repeat heading text so section-title queries rank strongly in ILIKE search
        parts.extend([body, body])
    elif body:
        parts.append(body)

    return "\n".join(parts)


def build_embedding_texts_for_chunks(
    chunks: List[str],
    metadata_list: List[Dict[str, Any]],
    *,
    document_title: str = "",
    llm_context_by_chunk_id: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Return embedding_text per chunk; also sets meta['embedding_text'] on each metadata dict."""
    labels = build_chunk_display_labels(metadata_list)
    llm_ctx = llm_context_by_chunk_id or {}
    out: List[str] = []
    for text, meta in zip(chunks, metadata_list):
        emb = build_embedding_text(
            text,
            meta,
            document_title=document_title,
            chunk_labels=labels,
            llm_context_line=llm_ctx.get(meta.get("chunk_id") or ""),
        )
        meta["embedding_text"] = emb
        meta["fts_index_text"] = build_fts_index_text(
            text, meta, document_title=document_title
        )
        out.append(emb)
    return out


def build_fts_index_texts_for_chunks(
    chunks: List[str],
    metadata_list: List[Dict[str, Any]],
    *,
    document_title: str = "",
) -> List[str]:
    """FTS strings with hierarchy tokens; sets meta['fts_index_text']."""
    out: List[str] = []
    for text, meta in zip(chunks, metadata_list):
        fts = build_fts_index_text(text, meta, document_title=document_title)
        meta["fts_index_text"] = fts
        out.append(fts)
    return out
