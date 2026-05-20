"""
Keyword (full-text) search: Supabase (ILIKE) or SQLite FTS5 depending on database backend.
"""

import re
from typing import List, Dict, Optional, Sequence
import logging

from app.services.database import get_database

logger = logging.getLogger(__name__)


_KEYWORD_STOPWORDS = frozenset({
    "what", "are", "is", "was", "the", "a", "an", "and", "or", "give", "some", "how",
    "to", "in", "of", "for", "with", "about", "explain", "tell", "me", "can", "you",
    "it", "this", "that", "those", "them", "depth", "examples", "example", "please",
    "python", "javascript", "java", "csharp", "react", "node", "backend", "frontend",
    "good", "best", "great", "nice", "very", "really", "just", "also", "nad", "adn", "nd",
})


def _tokenize_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", query or "")
    kept = []
    for t in tokens:
        low = t.lower()
        if len(low) < 3 or low in _KEYWORD_STOPWORDS:
            continue
        kept.append(low)
        if low.endswith("s") and len(low) > 4:
            kept.append(low[:-1])
    # dedupe preserve order
    seen: set = set()
    out: list = []
    for t in kept:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return " ".join(out) if out else ""


def _build_fts_query_sqlite(query: str) -> Optional[str]:
    """FTS5: OR of tokens."""
    tokens = re.findall(r"[A-Za-z0-9_]+", query or "")
    tokens = [t for t in tokens if len(t) >= 2]
    return " OR ".join(tokens) if tokens else None


class KeywordSearchService:
    """Full-text search for document chunks (Supabase ILIKE or SQLite FTS5)."""

    def __init__(self):
        self.db = get_database()
        self._use_supabase = self.db.engine == "supabase"
        logger.info(
            "KeywordSearchService initialized (%s)",
            "Supabase" if self._use_supabase else "SQLite FTS5",
        )

    def delete_document(self, document_id: str):
        if not document_id or not str(document_id).strip():
            return
        if self._use_supabase:
            self.db.supabase.table("chunks_fts").delete().eq("document_id", document_id).execute()
        else:
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM chunks_fts WHERE document_id = %s", (document_id,))

    def index_chunks(
        self,
        chunks: List[str],
        chunk_ids: List[str],
        document_id: str,
        file_name: str,
        technology: str,
        domain: str,
        file_id: Optional[str] = None,
        start_chunk_index: int = 0,
    ):
        if not chunks:
            return
        if chunk_ids and len(chunk_ids) != len(chunks):
            raise ValueError("chunk_ids length mismatch")

        if self._use_supabase:
            rows_to_insert = []
            for i, chunk in enumerate(chunks):
                chunk_id = chunk_ids[i] if chunk_ids else ""
                idx = str(start_chunk_index + i)
                rows_to_insert.append({
                    "text": chunk,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "file_name": file_name,
                    "technology": technology,
                    "domain": domain,
                    "chunk_index": idx,
                    "file_id": file_id or "",
                })
            self.db.supabase.table("chunks_fts").insert(rows_to_insert).execute()
        else:
            rows = []
            for i, chunk in enumerate(chunks):
                chunk_id = chunk_ids[i] if chunk_ids else ""
                idx = str(start_chunk_index + i)
                rows.append((chunk, chunk_id, document_id, file_name, technology, domain, idx, file_id or ""))
            with self.db.get_connection() as conn:
                cur = conn.cursor()
                cur.executemany(
                    """INSERT INTO chunks_fts (text, chunk_id, document_id, file_name, technology, domain, chunk_index, file_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    rows,
                )

    def search(
        self,
        query: str,
        top_k: int = 5,
        technology: Optional[str] = None,
        domain: Optional[str] = None,
        document_id: Optional[str] = None,
        file_id: Optional[str] = None,
        document_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        if self._use_supabase:
            return self._search_supabase(
                query, top_k, technology, domain, document_id, file_id, document_ids
            )
        return self._search_sqlite(
            query, top_k, technology, domain, document_id, file_id, document_ids
        )

    def _search_supabase(
        self,
        query: str,
        top_k: int,
        technology: Optional[str],
        domain: Optional[str],
        document_id: Optional[str],
        file_id: Optional[str],
        document_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        search_str = _tokenize_query(query)
        if not search_str:
            return []

        # Build query with ILIKE for each token (OR logic)
        tokens = search_str.split()
        sb_query = self.db.supabase.table("chunks_fts").select("*")

        # Apply filters
        if technology:
            sb_query = sb_query.eq("technology", technology)
        if domain:
            sb_query = sb_query.eq("domain", domain)
        if document_ids:
            ids = [str(d).strip() for d in document_ids if d and str(d).strip()]
            if len(ids) == 1:
                sb_query = sb_query.eq("document_id", ids[0])
            elif ids:
                sb_query = sb_query.in_("document_id", ids)
        elif document_id:
            sb_query = sb_query.eq("document_id", document_id)
        elif file_id:
            sb_query = sb_query.eq("file_id", file_id)

        # Use OR across all tokens. This avoids the old bug where only tokens[0] was used.
        # supabase-py expects PostgREST OR syntax: "col.op.value,col.op.value"
        if tokens:
            or_filters = ",".join([f"text.ilike.%{t}%" for t in tokens[:10]])
            sb_query = sb_query.or_(or_filters)

        sb_query = sb_query.limit(top_k * 3)  # Fetch more to score and filter
        response = sb_query.execute()

        # Score results by how many tokens match
        results = []
        for row in response.data:
            text_lower = (row.get("text") or "").lower()
            match_count = sum(1 for t in tokens if t.lower() in text_lower)
            if match_count > 0:
                results.append({
                    "text": row.get("text", ""),
                    "chunk_id": row.get("chunk_id") or "",
                    "document_id": row.get("document_id") or "",
                    "file_name": row.get("file_name") or "",
                    "technology": row.get("technology") or "general",
                    "domain": row.get("domain") or "general",
                    "chunk_index": int(row.get("chunk_index") or 0),
                    "score": 1.0 / (1.0 + match_count),  # Lower score = better match
                    "_match_count": match_count,
                })

        # Sort by match count (descending) and return top_k
        results.sort(key=lambda x: -x["_match_count"])
        for r in results:
            del r["_match_count"]
        return results[:top_k]

    def _search_sqlite(
        self,
        query: str,
        top_k: int,
        technology: Optional[str],
        domain: Optional[str],
        document_id: Optional[str],
        file_id: Optional[str],
        document_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        fts_query = _build_fts_query_sqlite(query)
        if not fts_query:
            return []
        sql = """
            SELECT rowid, text, chunk_id, document_id, file_name, technology, domain,
                   chunk_index, file_id, bm25(chunks_fts) AS bm25
            FROM chunks_fts
            WHERE chunks_fts MATCH %s
        """
        params: list = [fts_query]
        if technology:
            sql += " AND technology = %s"
            params.append(technology)
        if domain:
            sql += " AND domain = %s"
            params.append(domain)
        if document_ids:
            ids = [str(d).strip() for d in document_ids if d and str(d).strip()]
            if len(ids) == 1:
                sql += " AND document_id = %s"
                params.append(ids[0])
            elif ids:
                placeholders = ", ".join(["%s"] * len(ids))
                sql += f" AND document_id IN ({placeholders})"
                params.extend(ids)
        elif document_id:
            sql += " AND document_id = %s"
            params.append(document_id)
        elif file_id:
            sql += " AND file_id = %s"
            params.append(file_id)
        sql += " ORDER BY bm25 ASC LIMIT %s"
        params.append(top_k)
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
        results = []
        for r in rows:
            results.append({
                "text": r["text"],
                "chunk_id": r["chunk_id"] or "",
                "document_id": r["document_id"] or "",
                "file_name": r["file_name"] or "",
                "technology": r["technology"] or "general",
                "domain": r["domain"] or "general",
                "chunk_index": int(r["chunk_index"]) if r["chunk_index"] is not None else 0,
                "score": float(r["bm25"]) if r["bm25"] is not None else 0.0,
            })
        return results


_keyword_service: Optional[KeywordSearchService] = None


def get_keyword_search_service() -> KeywordSearchService:
    global _keyword_service
    if _keyword_service is None:
        _keyword_service = KeywordSearchService()
    return _keyword_service
