"""Tests for document graph store batching helpers."""
from app.services.document_graph_store import _edge_row, _node_row


def test_edge_row_generates_id():
    row = _edge_row({
        "document_id": "doc-1",
        "from_node_id": "a",
        "to_node_id": "b",
        "edge_type": "parent_of",
    })
    assert row["edge_id"]
    assert row["document_id"] == "doc-1"


def test_node_row_serializes_path_list():
    row = _node_row({
        "node_id": "n1",
        "document_id": "doc-1",
        "node_type": "section",
        "section_path_json": ["Intro", "Details"],
    })
    assert "Intro" in row["section_path_json"]
