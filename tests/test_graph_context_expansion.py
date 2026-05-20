"""Tests for Phase 4 graph context expansion."""
from app.services.graph_context_expansion import pack_context_chunks, expand_context_graph


def test_pack_context_chunks_respects_budget():
    chunks = [
        {"chunk_id": "a", "text": "x" * 100, "score": 0.9},
        {"chunk_id": "b", "text": "y" * 100, "score": 0.5},
        {"chunk_id": "c", "text": "z" * 10000, "score": 0.8},
    ]
    packed = pack_context_chunks(chunks, max_chars=250)
    assert len(packed) >= 1
    total = sum(len(c.get("text") or "") for c in packed)
    assert total <= 300


def test_expand_context_graph_follows_prev_next(monkeypatch):
    store_rows = {
        "p2": {
            "chunk_id": "p2",
            "document_id": "doc1",
            "chunk_type": "paragraph",
            "retrieval_text": "Next paragraph text.",
            "page_number": 1,
            "section_title": "Intro",
            "metadata_json": '{"node_id":"n2","chunk_type":"paragraph"}',
        },
    }

    class FakeChunkStore:
        def get_chunks(self, ids):
            return {i: store_rows[i] for i in ids if i in store_rows}

        def list_chunks_by_document(self, document_id, limit=400):
            return list(store_rows.values())

    class FakeGraphStore:
        def list_edges(self, document_id, limit=8000):
            return []

    monkeypatch.setattr(
        "app.services.graph_context_expansion.get_chunk_store",
        lambda: FakeChunkStore(),
    )
    monkeypatch.setattr(
        "app.services.graph_context_expansion.get_document_graph_store",
        lambda: FakeGraphStore(),
    )

    seeds = [
        {
            "chunk_id": "p1",
            "document_id": "doc1",
            "chunk_type": "paragraph",
            "text": "First paragraph.",
            "score": 0.7,
            "metadata_json": '{"prev_chunk_id":null,"next_chunk_id":"p2","node_id":"n1"}',
        }
    ]
    out = expand_context_graph(seeds, query="explain intro")
    ids = {c.get("chunk_id") for c in out}
    assert "p1" in ids
    assert "p2" in ids
