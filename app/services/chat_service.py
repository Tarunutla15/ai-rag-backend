"""
Chat Service for managing chat sessions and messages.
Supports Supabase (REST API) and SQLite fallback.
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import logging

from .database import get_database

logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat sessions and messages."""
    
    def __init__(self):
        """Initialize chat service with database connection."""
        self.db = get_database()
        self._use_supabase = self.db.engine == "supabase"
    
    # ==================== SESSION OPERATIONS ====================
    
    def create_session(self, title: str = "New Chat") -> Dict:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        if self._use_supabase:
            self.db.supabase.table("chat_sessions").insert({
                "id": session_id,
                "title": title,
                "created_at": now,
                "last_message_at": now
            }).execute()
        else:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO chat_sessions (id, title, created_at, last_message_at)
                    VALUES (%s, %s, %s, %s)""",
                    (session_id, title, now, now)
                )
        
        logger.info(f"Created new session: {session_id}")
        return {"id": session_id, "title": title, "created_at": now, "last_message_at": now}
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a session by ID."""
        if self._use_supabase:
            response = self.db.supabase.table("chat_sessions").select("*").eq("id", session_id).execute()
            if response.data:
                row = response.data[0]
                return {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "last_message_at": row["last_message_at"]
                }
            return None
        else:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, title, created_at, last_message_at FROM chat_sessions WHERE id = %s",
                    (session_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "title": row["title"],
                        "created_at": row["created_at"],
                        "last_message_at": row["last_message_at"]
                    }
                return None
    
    def get_all_sessions(self) -> List[Dict]:
        """Get all chat sessions, ordered by last message time (newest first)."""
        if self._use_supabase:
            response = self.db.supabase.table("chat_sessions").select("*").order("last_message_at", desc=True).execute()
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "last_message_at": row["last_message_at"]
                }
                for row in response.data
            ]
        else:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT id, title, created_at, last_message_at 
                    FROM chat_sessions ORDER BY last_message_at DESC"""
                )
                rows = cursor.fetchall()
                return [
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "created_at": row["created_at"],
                        "last_message_at": row["last_message_at"]
                    }
                    for row in rows
                ]
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update a session's title."""
        if self._use_supabase:
            response = self.db.supabase.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()
            return len(response.data) > 0
        else:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE chat_sessions SET title = %s WHERE id = %s", (title, session_id))
                return cursor.rowcount > 0
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        if self._use_supabase:
            # Delete messages first
            self.db.supabase.table("chat_messages").delete().eq("session_id", session_id).execute()
            # Delete attached documents
            self.db.supabase.table("session_documents").delete().eq("session_id", session_id).execute()
            response = self.db.supabase.table("chat_sessions").delete().eq("id", session_id).execute()
            if response.data:
                logger.info(f"Deleted session: {session_id}")
                return True
            return False
        else:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
                cursor.execute("DELETE FROM session_documents WHERE session_id = %s", (session_id,))
                cursor.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
                if cursor.rowcount > 0:
                    logger.info(f"Deleted session: {session_id}")
                    return True
                return False

    def _delete_all_sessions_sync(self) -> int:
        """Bulk delete all sessions (blocking DB calls)."""
        if self._use_supabase:
            count_resp = self.db.supabase.table("chat_sessions").select("id", count="exact").execute()
            removed = int(count_resp.count or 0)
            if removed > 0:
                self.db.supabase.table("chat_messages").delete().neq("session_id", "").execute()
                self.db.supabase.table("session_documents").delete().neq("session_id", "").execute()
                self.db.supabase.table("chat_sessions").delete().neq("id", "").execute()
                logger.info("Deleted all chat sessions (bulk, count=%s)", removed)
            return removed

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM chat_sessions")
            removed = int(cursor.fetchone()["count"] or 0)
            if removed > 0:
                cursor.execute("DELETE FROM chat_messages")
                cursor.execute("DELETE FROM session_documents")
                cursor.execute("DELETE FROM chat_sessions")
                logger.info("Deleted all chat sessions (bulk SQLite, count=%s)", removed)
        return removed

    async def delete_all_sessions_async(self) -> int:
        """Non-blocking wrapper for bulk session delete."""
        return await asyncio.to_thread(self._delete_all_sessions_sync)

    def delete_all_sessions(self) -> int:
        """Sync entry point."""
        return self._delete_all_sessions_sync()
    
    # ==================== MESSAGE OPERATIONS ====================
    
    def add_message(self, session_id: str, role: str, content: str) -> Dict:
        """Add a message to a session."""
        now = datetime.utcnow().isoformat()
        
        if self._use_supabase:
            response = self.db.supabase.table("chat_messages").insert({
                "session_id": session_id,
                "role": role,
                "content": content,
                "created_at": now
            }).execute()
            message_id = response.data[0]["id"] if response.data else None
            
            # Update session's last_message_at
            self.db.supabase.table("chat_sessions").update({"last_message_at": now}).eq("id", session_id).execute()
            
            # Auto-generate title from first user message
            if role == "user":
                count_response = self.db.supabase.table("chat_messages").select("id", count="exact").eq("session_id", session_id).execute()
                if count_response.count == 1:
                    auto_title = content[:50] + "..." if len(content) > 50 else content
                    self.db.supabase.table("chat_sessions").update({"title": auto_title}).eq("id", session_id).execute()
        else:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO chat_messages (session_id, role, content, created_at)
                    VALUES (%s, %s, %s, %s)""",
                    (session_id, role, content, now)
                )
                message_id = cursor.lastrowid
                
                cursor.execute("UPDATE chat_sessions SET last_message_at = %s WHERE id = %s", (now, session_id))
                
                if role == "user":
                    cursor.execute("SELECT COUNT(*) as count FROM chat_messages WHERE session_id = %s", (session_id,))
                    count = cursor.fetchone()["count"]
                    if count == 1:
                        auto_title = content[:50] + "..." if len(content) > 50 else content
                        cursor.execute("UPDATE chat_sessions SET title = %s WHERE id = %s", (auto_title, session_id))
        
        logger.info(f"Added {role} message to session {session_id[:8]}...")
        return {"id": message_id, "session_id": session_id, "role": role, "content": content, "created_at": now}
    
    def get_messages(self, session_id: str, limit: Optional[int] = None, order: str = "asc") -> List[Dict]:
        """Get messages for a session."""
        if self._use_supabase:
            query = self.db.supabase.table("chat_messages").select("*").eq("session_id", session_id)
            if limit:
                query = query.order("created_at", desc=True).limit(limit)
                response = query.execute()
                rows = list(reversed(response.data))
            else:
                # No limit: get all messages (Supabase/PostgREST default limit can clip results)
                query = query.order("created_at", desc=(order.lower() == "desc")).limit(10000)
                response = query.execute()
                rows = response.data
            return [
                {"id": row["id"], "session_id": row["session_id"], "role": row["role"], "content": row["content"], "created_at": row["created_at"]}
                for row in rows
            ]
        else:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                if limit:
                    cursor.execute(
                        """SELECT id, session_id, role, content, created_at FROM chat_messages 
                        WHERE session_id = %s ORDER BY created_at DESC LIMIT %s""",
                        (session_id, limit)
                    )
                    rows = list(reversed(cursor.fetchall()))
                else:
                    order_dir = "ASC" if order.lower() == "asc" else "DESC"
                    cursor.execute(
                        f"""SELECT id, session_id, role, content, created_at FROM chat_messages 
                        WHERE session_id = %s ORDER BY created_at {order_dir}""",
                        (session_id,)
                    )
                    rows = cursor.fetchall()
                return [
                    {"id": row["id"], "session_id": row["session_id"], "role": row["role"], "content": row["content"], "created_at": row["created_at"]}
                    for row in rows
                ]
    
    def get_context_messages(self, session_id: str, context_window: int = 6) -> List[Dict]:
        """Get messages for LLM context window."""
        messages = self.get_messages(session_id, limit=context_window)
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]
    
    def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        if self._use_supabase:
            response = self.db.supabase.table("chat_messages").select("id", count="exact").eq("session_id", session_id).execute()
            return response.count or 0
        else:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM chat_messages WHERE session_id = %s", (session_id,))
                return cursor.fetchone()["count"]

    # ==================== SESSION DOCUMENT SCOPE ====================

    def get_session_documents(self, session_id: str) -> List[str]:
        """List document_ids attached to a session."""
        if self._use_supabase:
            resp = (
                self.db.supabase.table("session_documents")
                .select("document_id")
                .eq("session_id", session_id)
                .limit(10000)
                .execute()
            )
            return [r.get("document_id") for r in (resp.data or []) if r.get("document_id")]
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT document_id FROM session_documents WHERE session_id = %s", (session_id,))
            rows = cursor.fetchall()
            return [r["document_id"] for r in rows if r["document_id"]]

    def set_session_documents(self, session_id: str, document_ids: List[str]) -> List[str]:
        """Replace the session's document scope with provided document_ids."""
        document_ids = [d for d in (document_ids or []) if d]
        if self._use_supabase:
            # replace: delete then insert
            self.db.supabase.table("session_documents").delete().eq("session_id", session_id).execute()
            if document_ids:
                rows = [{"session_id": session_id, "document_id": d} for d in document_ids]
                self.db.supabase.table("session_documents").insert(rows).execute()
            return document_ids
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM session_documents WHERE session_id = %s", (session_id,))
            if document_ids:
                cursor.executemany(
                    "INSERT INTO session_documents (session_id, document_id) VALUES (%s, %s)",
                    [(session_id, d) for d in document_ids],
                )
        return document_ids

    def remove_document_from_all_sessions(self, document_id: str) -> None:
        """Remove document_id from every chat session scope (session_documents)."""
        if not document_id:
            return
        if self._use_supabase:
            self.db.supabase.table("session_documents").delete().eq("document_id", document_id).execute()
            return
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM session_documents WHERE document_id = %s", (document_id,))


_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create the global chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
