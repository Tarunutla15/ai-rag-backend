"""Tests for Phase 1 document tree builder."""
from app.services.document_tree import DocumentTreeBuilder, infer_heading_level, apply_chunk_sequence_links


def test_infer_heading_level_numbered():
    assert infer_heading_level("1 Introduction") == 1
    assert infer_heading_level("2.1 Methods") == 2
    assert infer_heading_level("3.2.1 Details") == 3


def test_build_tree_section_hierarchy():
    blocks = [
        {"block_type": "heading", "content": "Chapter One", "page_number": 1},
        {"block_type": "text", "content": "Intro paragraph.", "page_number": 1},
        {"block_type": "heading", "content": "2.1 Subsection", "page_number": 2},
        {"block_type": "text", "content": "Detail text.", "page_number": 2},
        {"block_type": "table", "content": "", "table_summary": "Cols: A, B", "table_rows": [["A", "B"]], "page_number": 2},
    ]
    enriched, nodes, edges = DocumentTreeBuilder().build(blocks, "doc-test-1")
    assert len(enriched) == 5
    assert any(n["node_type"] == "document" for n in nodes)
    assert any(n["node_type"] == "section" and "Chapter One" in (n.get("title") or "") for n in nodes)
    parent_edges = [e for e in edges if e["edge_type"] == "parent_of"]
    assert len(parent_edges) >= 4
    text_block = next(b for b in enriched if b.get("block_type") == "text" and "Intro" in b.get("content", ""))
    assert text_block.get("parent_section_id")
    assert text_block.get("section_path") == ["Chapter One"]


def test_apply_chunk_sequence_links():
    meta = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    apply_chunk_sequence_links(meta)
    assert meta[0]["next_chunk_id"] == "b"
    assert meta[1]["prev_chunk_id"] == "a" and meta[1]["next_chunk_id"] == "c"
    assert meta[2]["prev_chunk_id"] == "b"
