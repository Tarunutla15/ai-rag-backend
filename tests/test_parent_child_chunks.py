"""Tests for hierarchical parent–child chunking and context resolution."""
import json

from app.services.parent_child_chunks import (
    build_section_parents,
    is_searchable_child_chunk,
    CHUNK_TYPE_SECTION_PARENT,
)
from app.services.parent_context_resolution import resolve_parent_context


def test_is_searchable_child_excludes_section_parent():
    assert not is_searchable_child_chunk({"chunk_type": CHUNK_TYPE_SECTION_PARENT})
    assert is_searchable_child_chunk({"chunk_type": "paragraph"})


def test_build_section_parents_links_children():
    chunks = [
        "The system improves latency by 42%.",
        "Methodology uses batch processing.",
    ]
    metadata = [
        {
            "chunk_id": "a1",
            "chunk_type": "paragraph",
            "parent_section_id": "sec-1",
            "parent_section_title": "Results",
            "section_path": ["Results"],
            "order_index": 1,
            "page_number": 2,
        },
        {
            "chunk_id": "a2",
            "chunk_type": "paragraph",
            "parent_section_id": "sec-1",
            "parent_section_title": "Results",
            "section_path": ["Results"],
            "order_index": 2,
            "page_number": 2,
        },
    ]
    parent_texts, parent_metas = build_section_parents(
        chunks, metadata, document_id="doc-1", file_name="test.docx"
    )
    assert len(parent_texts) == 1
    assert len(parent_metas) == 1
    assert parent_metas[0]["chunk_type"] == CHUNK_TYPE_SECTION_PARENT
    assert "42%" in parent_texts[0]
    assert "Methodology" in parent_texts[0]
    assert metadata[0].get("parent_chunk_id") == metadata[1].get("parent_chunk_id")
    assert metadata[0]["parent_chunk_id"] == parent_metas[0]["chunk_id"]


def test_resolve_parent_context_prepend_mode(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ENABLE_PARENT_CHILD_RETRIEVAL", True)
    monkeypatch.setattr(settings, "PARENT_CONTEXT_MODE", "prepend")

    parent_id = "parent-uuid"
    child_id = "child-uuid"

    class FakeStore:
        def get_chunks(self, ids):
            return {
                parent_id: {
                    "chunk_id": parent_id,
                    "retrieval_text": "Full section about latency experiments and batch methodology.",
                    "chunk_type": CHUNK_TYPE_SECTION_PARENT,
                }
            }

    monkeypatch.setattr(
        "app.services.parent_context_resolution.get_chunk_store",
        lambda: FakeStore(),
    )

    seeds = [
        {
            "chunk_id": child_id,
            "text": "it improves latency by 42%",
            "score": 0.9,
            "metadata_json": '{"parent_chunk_id": "%s", "chunk_type": "paragraph"}' % parent_id,
        }
    ]
    out = resolve_parent_context(seeds)
    assert len(out) == 1
    assert "Full section about latency" in out[0]["text"]
    assert "42%" in out[0]["text"]
    assert out[0].get("child_snippet") == "it improves latency by 42%"
    assert out[0].get("context_source") == "parent_section"


def test_section_group_key_uses_page_and_title():
    from app.services.parent_child_chunks import _section_group_key

    key = _section_group_key(
        {
            "parent_section_id": "root-same-for-all",
            "section_path": [],
            "section_title": "Component Lifecycle",
            "page_number": 11,
        }
    )
    assert key == "page:11:sec:Component Lifecycle"


def test_build_section_parents_splits_by_page_and_title():
    chunks = ["lifecycle text", "diagram caption"]
    metadata = [
        {
            "chunk_id": "c1",
            "chunk_type": "paragraph",
            "parent_section_id": "root",
            "section_title": "Lifecycle",
            "page_number": 10,
            "order_index": 1,
        },
        {
            "chunk_id": "c2",
            "chunk_type": "image_caption",
            "parent_section_id": "root",
            "section_title": "Diagram",
            "page_number": 22,
            "order_index": 2,
        },
    ]
    parent_texts, parent_metas = build_section_parents(
        chunks, metadata, document_id="doc-2", file_name="t.pdf"
    )
    assert len(parent_texts) == 2
    assert metadata[0]["parent_chunk_id"] != metadata[1]["parent_chunk_id"]


def test_resolve_parent_context_dedupes_same_parent(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ENABLE_PARENT_CHILD_RETRIEVAL", True)
    monkeypatch.setattr(settings, "PARENT_CONTEXT_MODE", "replace")

    parent_id = "parent-uuid"

    class FakeStore:
        def get_chunks(self, ids):
            return {
                parent_id: {
                    "chunk_id": parent_id,
                    "retrieval_text": "Section body with full context.",
                    "chunk_type": CHUNK_TYPE_SECTION_PARENT,
                }
            }

    monkeypatch.setattr(
        "app.services.parent_context_resolution.get_chunk_store",
        lambda: FakeStore(),
    )

    seeds = [
        {
            "chunk_id": "c1",
            "text": "first hit",
            "score": 0.9,
            "metadata_json": '{"parent_chunk_id": "%s"}' % parent_id,
        },
        {
            "chunk_id": "c2",
            "text": "second hit same section",
            "score": 0.8,
            "metadata_json": '{"parent_chunk_id": "%s"}' % parent_id,
        },
    ]
    out = resolve_parent_context(seeds)
    assert len(out) == 1
    assert out[0]["text"] == "Section body with full context."


def test_resolve_parent_mega_parent_keeps_children(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ENABLE_PARENT_CHILD_RETRIEVAL", True)
    monkeypatch.setattr(settings, "PARENT_MEGA_CHILD_THRESHOLD", 3)

    parent_id = "mega-parent"

    class FakeStore:
        def get_chunks(self, ids):
            return {
                parent_id: {
                    "chunk_id": parent_id,
                    "retrieval_text": "X" * 500,
                    "chunk_type": CHUNK_TYPE_SECTION_PARENT,
                    "metadata_json": json.dumps(
                        {
                            "child_chunk_ids": ["a", "b", "c", "d"],
                            "section_group_key": "_document",
                        }
                    ),
                }
            }

    monkeypatch.setattr(
        "app.services.parent_context_resolution.get_chunk_store",
        lambda: FakeStore(),
    )

    seeds = [
        {
            "chunk_id": "c1",
            "text": "lifecycle methods",
            "score": 0.9,
            "metadata_json": '{"parent_chunk_id": "%s", "chunk_type": "paragraph"}' % parent_id,
        },
        {
            "chunk_id": "c2",
            "text": "[Visual description]: diagram",
            "score": 0.85,
            "chunk_type": "image_caption",
            "metadata_json": '{"parent_chunk_id": "%s", "chunk_type": "image_caption"}' % parent_id,
        },
    ]
    out = resolve_parent_context(seeds)
    assert len(out) == 2
    assert out[0]["text"] == "lifecycle methods"
    assert "Visual description" in out[1]["text"]


def test_resolve_parent_image_caption_always_kept(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ENABLE_PARENT_CHILD_RETRIEVAL", True)
    parent_id = "p1"

    class FakeStore:
        def get_chunks(self, ids):
            return {
                parent_id: {
                    "chunk_id": parent_id,
                    "retrieval_text": "Small section.",
                    "chunk_type": CHUNK_TYPE_SECTION_PARENT,
                    "metadata_json": json.dumps({"child_chunk_ids": ["x", "y"]}),
                }
            }

    monkeypatch.setattr(
        "app.services.parent_context_resolution.get_chunk_store",
        lambda: FakeStore(),
    )

    seeds = [
        {
            "chunk_id": "img1",
            "text": "figure with [Visual description]: flow",
            "score": 0.7,
            "chunk_type": "image_caption",
            "metadata_json": '{"parent_chunk_id": "%s", "chunk_type": "image_caption"}' % parent_id,
        },
        {
            "chunk_id": "para1",
            "text": "lifecycle hooks",
            "score": 0.95,
            "metadata_json": '{"parent_chunk_id": "%s", "chunk_type": "paragraph"}' % parent_id,
        },
    ]
    out = resolve_parent_context(seeds)
    assert len(out) == 2
    texts = {c["chunk_id"]: c["text"] for c in out}
    assert "Visual description" in texts["img1"]
    assert "lifecycle" in texts["para1"]
