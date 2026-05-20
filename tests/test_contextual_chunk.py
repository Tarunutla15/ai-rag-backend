"""Tests for Phase 3 contextual embedding strings."""
from app.services.contextual_chunk import (
    build_chunk_display_labels,
    build_embedding_text,
    build_embedding_texts_for_chunks,
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
