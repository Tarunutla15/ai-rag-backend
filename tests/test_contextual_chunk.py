"""Tests for Phase 3 contextual embedding strings."""
from app.services.contextual_chunk import (
    build_chunk_display_labels,
    build_embedding_text,
    build_embedding_texts_for_chunks,
    build_fts_index_text,
)


def test_build_embedding_text_includes_structure():
    meta = {
        "chunk_id": "c1",
        "chunk_type": "paragraph",
        "section_path": ["Transformer Architecture", "Multi-Head Attention"],
        "file_name": "attention.pdf",
    }
    text = build_embedding_text(
        "Multi-head attention allows the model to jointly attend.",
        meta,
        document_title="Attention Is All You Need",
        chunk_labels={"c2": "Table 1"},
    )
    assert "[Document: Attention Is All You Need]" in text
    assert "[Section: Transformer Architecture > Multi-Head Attention]" in text
    assert "[Type: paragraph]" in text
    assert "Multi-head attention" in text


def test_build_embedding_text_includes_page_and_heading_anchor():
    meta = {
        "chunk_id": "h1",
        "chunk_type": "heading",
        "section_path": ["Expense Module", "Upload Receipt"],
        "page_number": 5,
        "file_name": "guide.docx",
    }
    text = build_embedding_text("Upload Receipt", meta, document_title="Travel Guide")
    assert "[Page: 5]" in text
    assert "[Heading: Upload Receipt]" in text
    assert "[Section: Expense Module > Upload Receipt]" in text


def test_build_fts_index_text_repeats_heading_for_search():
    meta = {
        "chunk_type": "heading",
        "section_path": ["Request Advance"],
        "page_number": 3,
        "file_name": "policy.docx",
    }
    fts = build_fts_index_text("Request Advance", meta, document_title="Travel Policy")
    assert fts.count("Request Advance") >= 2
    assert "page 3" in fts
    assert "Request Advance" in fts


def test_related_line_in_embedding_text():
    meta = {
        "chunk_id": "t1",
        "chunk_type": "table_summary",
        "related_paragraph_chunk_id": "p1",
        "section_title": "Results",
    }
    labels = {"p1": "Paragraph 1", "t1": "Table 1"}
    text = build_embedding_text("Revenue | 100", meta, chunk_labels=labels)
    assert "[Related:" in text
    assert "Paragraph 1" in text


def test_build_embedding_texts_for_chunks_sets_metadata():
    chunks = ["Hello world", "Cols: A | B"]
    meta = [
        {"chunk_id": "a", "chunk_type": "paragraph", "order_index": 1, "section_path": ["Intro"]},
        {"chunk_id": "b", "chunk_type": "table_summary", "order_index": 2, "related_paragraph_chunk_id": "a"},
    ]
    texts = build_embedding_texts_for_chunks(chunks, meta, document_title="Doc")
    assert len(texts) == 2
    assert meta[0].get("embedding_text")
    assert "[Document: Doc]" in meta[0]["embedding_text"]
    labels = build_chunk_display_labels(meta)
    assert labels["b"] == "Table 1"
