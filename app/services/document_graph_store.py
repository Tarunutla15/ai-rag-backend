"""Persist document tree nodes and edges (Supabase or SQLite)."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from app.services.database import get_database

logger = logging.getLogger(__name__)

# PostgREST payload limits: batch large trees (200+ nodes) for reliable inserts.
_INSERT_BATCH_SIZE = 80


class DocumentGraphStore:
    def __init__(self):
        self.db = get_database()
        self._use_supabase = self.db.engine == "supabase"

    def delete_document_graph(self, document_id: str) -> None:
        if not document_id:
            return
        if self._use_supabase:
            self.db.supabase.table("document_edges").delete().eq("document_id", document_id).execute()
            self.db.supabase.table("document_nodes").delete().eq("document_id", document_id).execute()
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM document_edges WHERE document_id = %s", (document_id,))
                cur.execute("DELETE FROM document_nodes WHERE document_id = %s", (document_id,))

    def insert_graph(
        self,
        document_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> None:
        if not document_id:
            return
        if not nodes:
            return

        self._ensure_tables_ready()
        self.delete_document_graph(document_id)

        try:
            if self._use_supabase:
                self._insert_supabase_batched(nodes, edges)
            else:
                self._insert_sqlite(nodes, edges)
            self._verify_persisted(document_id, len(nodes), len(edges))
        except Exception:
            self.delete_document_graph(document_id)
            raise

        logger.info(
            "Inserted document graph: document_id=%s nodes=%s edges=%s",
            document_id[:8],
            len(nodes),
            len(edges),
        )

    def _ensure_tables_ready(self) -> None:
        """Probe document_nodes; trigger schema ensure if missing (Supabase REST-only setups)."""
        if not self._use_supabase:
            return
        try:
            self.db.supabase.table("document_nodes").select("node_id").limit(1).execute()
        except Exception as e:
            err = str(e).lower()
            if "document_nodes" not in err and "document_edges" not in err:
                raise
            from app.config import settings
            from app.services.database import _ensure_supabase_tables_via_postgres, _get_supabase_db_url

            db_url = _get_supabase_db_url()
            if db_url and _ensure_supabase_tables_via_postgres(db_url):
                import time
                time.sleep(2)
                self.db.supabase.table("document_nodes").select("node_id").limit(1).execute()
                logger.info("document_nodes/document_edges tables ensured via SUPABASE_DB_URL")
                return
            raise RuntimeError(
                "document_nodes / document_edges tables are missing in Supabase. "
                "Run backend/supabase_schema.sql and backend/supabase_rls_policies.sql in the SQL Editor."
            ) from e

    def _insert_supabase_batched(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> None:
        node_rows = [_node_row(n) for n in nodes]
        for i in range(0, len(node_rows), _INSERT_BATCH_SIZE):
            batch = node_rows[i : i + _INSERT_BATCH_SIZE]
            self.db.supabase.table("document_nodes").insert(batch).execute()

        if edges:
            edge_rows = [_edge_row(e) for e in edges]
            for i in range(0, len(edge_rows), _INSERT_BATCH_SIZE):
                batch = edge_rows[i : i + _INSERT_BATCH_SIZE]
                self.db.supabase.table("document_edges").insert(batch).execute()

    def _insert_sqlite(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> None:
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            for n in nodes:
                r = _node_row(n)
                cur.execute(
                    """INSERT INTO document_nodes
                       (node_id, document_id, node_type, title, content_preview,
                        page_start, page_end, parent_node_id, depth, order_index, section_path_json)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        r["node_id"], r["document_id"], r["node_type"], r["title"],
                        r["content_preview"], r["page_start"], r["page_end"],
                        r["parent_node_id"], r["depth"], r["order_index"], r["section_path_json"],
                    ),
                )
            for e in edges:
                r = _edge_row(e)
                cur.execute(
                    """INSERT INTO document_edges
                       (edge_id, document_id, from_node_id, to_node_id, edge_type)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (r["edge_id"], r["document_id"], r["from_node_id"], r["to_node_id"], r["edge_type"]),
                )

    def _verify_persisted(self, document_id: str, expected_nodes: int, expected_edges: int) -> None:
        if expected_nodes <= 0:
            return
        stored = len(self.list_nodes(document_id, limit=max(expected_nodes + 10, 50)))
        count = stored
        if count < 1:
            raise RuntimeError(
                f"document_nodes insert reported success but 0 rows found for document_id={document_id}. "
                "Check Supabase RLS policies (run backend/supabase_rls_policies.sql)."
            )
        if count < expected_nodes * 0.5:
            logger.warning(
                "document_nodes partial persist: expected ~%s got %s for %s",
                expected_nodes,
                count,
                document_id[:8],
            )

    def list_edges(self, document_id: str, limit: int = 8000) -> List[Dict[str, Any]]:
        if not document_id:
            return []
        if self._use_supabase:
            resp = (
                self.db.supabase.table("document_edges")
                .select("*")
                .eq("document_id", document_id)
                .limit(limit)
                .execute()
            )
            return list(resp.data or [])
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM document_edges WHERE document_id = %s LIMIT %s",
                (document_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]

    def list_nodes(self, document_id: str, limit: int = 5000) -> List[Dict[str, Any]]:
        if not document_id:
            return []
        if self._use_supabase:
            resp = (
                self.db.supabase.table("document_nodes")
                .select("*")
                .eq("document_id", document_id)
                .order("order_index")
                .limit(limit)
                .execute()
            )
            return list(resp.data or [])
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM document_nodes WHERE document_id = %s ORDER BY order_index LIMIT %s",
                (document_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]


def _node_row(n: Dict[str, Any]) -> Dict[str, Any]:
    path = n.get("section_path_json")
    if isinstance(path, list):
        path = json.dumps(path, ensure_ascii=False)
    return {
        "node_id": n["node_id"],
        "document_id": n["document_id"],
        "node_type": n.get("node_type", "paragraph"),
        "title": (n.get("title") or "")[:500],
        "content_preview": (n.get("content_preview") or "")[:500],
        "page_start": int(n.get("page_start") or 1),
        "page_end": int(n.get("page_end") or 1),
        "parent_node_id": n.get("parent_node_id"),
        "depth": int(n.get("depth") or 0),
        "order_index": int(n.get("order_index") or 0),
        "section_path_json": path if isinstance(path, str) else "[]",
    }


def _edge_row(e: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "edge_id": e.get("edge_id") or str(uuid.uuid4()),
        "document_id": e["document_id"],
        "from_node_id": e["from_node_id"],
        "to_node_id": e["to_node_id"],
        "edge_type": e.get("edge_type", "parent_of"),
    }


_store: Optional[DocumentGraphStore] = None


def get_document_graph_store() -> DocumentGraphStore:
    global _store
    if _store is None:
        _store = DocumentGraphStore()
    return _store
