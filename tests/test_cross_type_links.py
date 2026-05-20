"""Tests for Phase 2 cross-type chunk links."""
from app.services.chunking import ChunkingService
from app.services.document_tree import DocumentTreeBuilder, apply_chunk_sequence_links
from app.services.cross_type_links import (
    apply_cross_type_chunk_links,
    EDGE_SUPPORTS,
    EDGE_BELONGS_TO,
    EDGE_ILLUSTRATES,
)


def _chunk_pipeline(blocks, document_id="doc-phase2"):
    enriched, nodes, edges = DocumentTreeBuilder().build(blocks, document_id)
    chunks, metadata = ChunkingService(chunk_size=500, chunk_overlap=0).chunk_blocks(
        enriched, document_id=document_id, file_name="test.pdf"
    )
    for i, meta in enumerate(metadata):
        meta["chunk_id"] = f"chunk-{i}"
    apply_chunk_sequence_links(metadata)
    phase2 = apply_cross_type_chunk_links(document_id, metadata, blocks=enriched)
    return metadata, phase2, enriched


def test_table_supports_last_paragraph_in_section():
    blocks = [
        {"block_type": "heading", "content": "Results", "page_number": 1},
        {"block_type": "text", "content": "Revenue grew in Q4 as shown below.", "page_number": 1},
        {
            "block_type": "table",
            "content": "",
            "table_summary": "Revenue | 100",
            "table_rows": [["Revenue", "100"]],
            "page_number": 1,
        },
    ]
    metadata, phase2, _ = _chunk_pipeline(blocks)
    table_meta = next(m for m in metadata if m.get("chunk_type") == "table_summary")
    para_meta = next(m for m in metadata if m.get("chunk_type") == "paragraph")
    assert table_meta.get("related_paragraph_chunk_id") == para_meta["chunk_id"]
    assert para_meta["chunk_id"] in (table_meta.get("related_chunk_ids") or [])
    assert any(e["edge_type"] == EDGE_SUPPORTS for e in phase2)


def test_image_belongs_to_section_and_links_page_prose():
    text_bid = "text-block-1"
    blocks = [
        {"block_type": "heading", "content": "Architecture", "page_number": 2},
        {
            "block_type": "text",
            "content": "The diagram shows the system layout.",
            "page_number": 2,
            "block_id": text_bid,
        },
        {
            "block_type": "image",
            "content": "Figure on page 2",
            "page_number": 2,
            "block_id": "img-1",
            "page_context_block_id": text_bid,
        },
    ]
    metadata, phase2, _ = _chunk_pipeline(blocks)
    img_meta = next(m for m in metadata if m.get("chunk_type") == "image_caption")
    para_meta = next(m for m in metadata if m.get("chunk_type") == "paragraph")
    assert img_meta.get("related_section_node_id")
    assert para_meta["chunk_id"] in (img_meta.get("related_paragraph_chunk_ids") or [])
    assert any(e["edge_type"] == EDGE_BELONGS_TO for e in phase2)
    assert any(e["edge_type"] == EDGE_ILLUSTRATES for e in phase2)


def test_code_links_section_and_previous_paragraph():
    blocks = [
        {"block_type": "heading", "content": "Example", "page_number": 1},
        {"block_type": "text", "content": "Use the following snippet.", "page_number": 1},
        {"block_type": "code", "content": "def hello():\n    pass", "page_number": 1},
    ]
    metadata, phase2, _ = _chunk_pipeline(blocks)
    code_meta = next(m for m in metadata if m.get("chunk_type") == "code_summary")
    para_meta = next(m for m in metadata if m.get("chunk_type") == "paragraph")
    assert code_meta.get("related_section_node_id")
    assert para_meta["chunk_id"] in (code_meta.get("related_chunk_ids") or [])
    assert any(e["edge_type"] == EDGE_SUPPORTS for e in phase2)
