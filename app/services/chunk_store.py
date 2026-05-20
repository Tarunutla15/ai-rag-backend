"""
Canonical chunk store for retrieval targets.
Supports Supabase (REST API) and SQLite fallback.
"""
from typing import List, Dict, Optional
import json
import uuid
import time
import logging

from app.services.database import get_database
from app.services.document_tree import apply_chunk_sequence_links
from app.services.cross_type_links import build_phase2_graph_links

logger = logging.getLogger(__name__)


class ChunkStore:
    def __init__(self):
        self.db = get_database()
        self._use_supabase = self.db.engine == "supabase"

    def prepare_chunk_ids_and_phase2_links(
        self,
        metadata_list: List[Dict],
        document_id: str,
        blocks: Optional[List[Dict]] = None,
    ) -> tuple:
        """
        Assign chunk_ids, sequence links, and Phase 2 cross-type metadata.
        Returns (chunk_ids, phase2_edges).
        """
        if not metadata_list:
            return [], []
        chunk_ids: List[str] = []
        for meta in metadata_list:
            if not meta.get("chunk_id"):
                meta["chunk_id"] = str(uuid.uuid4())
            chunk_ids.append(meta["chunk_id"])
        apply_chunk_sequence_links(metadata_list)
        phase2_edges = build_phase2_graph_links(document_id, metadata_list, blocks=blocks)
        return chunk_ids, phase2_edges

    def insert_chunks(
        self,
        chunks: List[str],
        metadata_list: List[Dict],
        document_id: str,
    ) -> List[str]:
        if not chunks:
            return []
        if len(chunks) != len(metadata_list):
            raise ValueError("Chunks and metadata length mismatch")

        chunk_ids: List[str] = []
        now_ts = int(time.time())

        for chunk, meta in zip(chunks, metadata_list):
            if not meta.get("chunk_id"):
                meta["chunk_id"] = str(uuid.uuid4())
            chunk_ids.append(meta["chunk_id"])
        if not any(m.get("prev_chunk_id") or m.get("next_chunk_id") for m in metadata_list):
            apply_chunk_sequence_links(metadata_list)

        if self._use_supabase:
            rows_to_insert = []
            for chunk, meta in zip(chunks, metadata_list):
                chunk_id = meta["chunk_id"]
                rows_to_insert.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_type": meta.get("chunk_type", "paragraph"),
                    "retrieval_text": chunk,
                    "page_number": meta.get("page_number", -1),
                    "section_title": meta.get("section_title", ""),
                    "metadata_json": json.dumps(meta, ensure_ascii=False),
                    "created_at": now_ts,
                })
            self.db.supabase.table("chunks").insert(rows_to_insert).execute()
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                for chunk, meta in zip(chunks, metadata_list):
                    chunk_id = meta["chunk_id"]
                    cur.execute(
                        """INSERT INTO chunks (chunk_id, document_id, chunk_type, retrieval_text,
                        page_number, section_title, metadata_json, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                        (
                            chunk_id, document_id, meta.get("chunk_type", "paragraph"),
                            chunk, meta.get("page_number", -1), meta.get("section_title", ""),
                            json.dumps(meta, ensure_ascii=False), now_ts,
                        ),
                    )

        return chunk_ids

    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        if self._use_supabase:
            response = self.db.supabase.table("chunks").select("*").eq("chunk_id", chunk_id).execute()
            if response.data:
                row = response.data[0]
                return {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "chunk_type": row["chunk_type"],
                    "retrieval_text": row["retrieval_text"],
                    "page_number": row["page_number"],
                    "section_title": row["section_title"],
                    "metadata_json": row["metadata_json"],
                    "created_at": row["created_at"],
                }
            return None
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT * FROM chunks WHERE chunk_id = %s", (chunk_id,))
                row = cur.fetchone()
            if not row:
                return None
            return {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "chunk_type": row["chunk_type"],
                "retrieval_text": row["retrieval_text"],
                "page_number": row["page_number"],
                "section_title": row["section_title"],
                "metadata_json": row["metadata_json"],
                "created_at": row["created_at"],
            }

    def get_chunks(self, chunk_ids: List[str]) -> Dict[str, Dict]:
        if not chunk_ids:
            return {}
        
        if self._use_supabase:
            response = self.db.supabase.table("chunks").select("*").in_("chunk_id", chunk_ids).execute()
            result = {}
            for row in response.data:
                result[row["chunk_id"]] = {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "chunk_type": row["chunk_type"],
                    "retrieval_text": row["retrieval_text"],
                    "page_number": row["page_number"],
                    "section_title": row["section_title"],
                    "metadata_json": row["metadata_json"],
                    "created_at": row["created_at"],
                }
            return result
        else:
            placeholders = ",".join(["%s"] * len(chunk_ids))
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
                rows = cur.fetchall()
            result = {}
            for row in rows:
                result[row["chunk_id"]] = {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "chunk_type": row["chunk_type"],
                    "retrieval_text": row["retrieval_text"],
                    "page_number": row["page_number"],
                    "section_title": row["section_title"],
                    "metadata_json": row["metadata_json"],
                    "created_at": row["created_at"],
                }
            return result

    def list_chunks_by_document(self, document_id: str, limit: int = 400) -> List[Dict]:
        """All chunks for a document (graph expansion / node mapping)."""
        if not document_id:
            return []
        limit = max(1, min(int(limit), 2000))
        if self._use_supabase:
            resp = (
                self.db.supabase.table("chunks")
                .select("*")
                .eq("document_id", document_id)
                .order("page_number")
                .limit(limit)
                .execute()
            )
            rows = resp.data or []
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """SELECT * FROM chunks WHERE document_id = %s
                       ORDER BY page_number ASC, created_at ASC LIMIT %s""",
                    (document_id, limit),
                )
                rows = cur.fetchall()
        out: List[Dict] = []
        for row in rows:
            out.append({
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "chunk_type": row["chunk_type"],
                "retrieval_text": row["retrieval_text"],
                "page_number": row["page_number"],
                "section_title": row["section_title"],
                "metadata_json": row["metadata_json"],
                "created_at": row["created_at"],
            })
        return out

    def list_chunks_by_type(
        self, document_id: str, chunk_type: str, limit: int = 20
    ) -> List[Dict]:
        """Return chunk rows for a document (e.g. image_caption) for supplemental retrieval."""
        if not document_id or not chunk_type:
            return []
        limit = max(1, min(int(limit), 100))
        if self._use_supabase:
            resp = (
                self.db.supabase.table("chunks")
                .select("*")
                .eq("document_id", document_id)
                .eq("chunk_type", chunk_type)
                .order("page_number")
                .limit(limit)
                .execute()
            )
            rows = resp.data or []
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """SELECT * FROM chunks WHERE document_id = %s AND chunk_type = %s
                       ORDER BY page_number ASC LIMIT %s""",
                    (document_id, chunk_type, limit),
                )
                rows = cur.fetchall()
        out: List[Dict] = []
        for row in rows:
            out.append({
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "chunk_type": row["chunk_type"],
                "retrieval_text": row["retrieval_text"],
                "page_number": row["page_number"],
                "section_title": row["section_title"],
                "metadata_json": row["metadata_json"],
                "created_at": row["created_at"],
            })
        return out

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document_id."""
        if not document_id:
            return
        if self._use_supabase:
            self.db.supabase.table("chunks").delete().eq("document_id", document_id).execute()
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))


_chunk_store: Optional[ChunkStore] = None


def get_chunk_store() -> ChunkStore:
    global _chunk_store
    if _chunk_store is None:
        _chunk_store = ChunkStore()
    return _chunk_store
