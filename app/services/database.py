"""
Database Service: Supabase (REST API) with SQLite fallback.
When SUPABASE_URL and SUPABASE_KEY are set, uses Supabase. Otherwise uses SQLite.
If Supabase tables are missing, creates them via direct PostgreSQL (SUPABASE_DB_URL
or SUPABASE_DB_USER/PASSWORD/HOST/PORT/NAME).
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Any, List, Dict
from urllib.parse import quote_plus
import logging

from app.config import settings

logger = logging.getLogger(__name__)


def _get_supabase_db_url() -> str:
    """Direct PostgreSQL URL for schema creation: from SUPABASE_DB_URL or from user/password/host/port/dbname."""
    import os
    url = getattr(settings, "SUPABASE_DB_URL", "") or ""
    if url.strip():
        return url.strip()
    user = (getattr(settings, "SUPABASE_DB_USER", "") or "").strip() or os.getenv("user", "")
    password = (getattr(settings, "SUPABASE_DB_PASSWORD", "") or "").strip() or os.getenv("password", "")
    host = (getattr(settings, "SUPABASE_DB_HOST", "") or "").strip() or os.getenv("host", "")
    port = (getattr(settings, "SUPABASE_DB_PORT", "") or "").strip() or os.getenv("port", "5432")
    dbname = (getattr(settings, "SUPABASE_DB_NAME", "") or "").strip() or os.getenv("dbname", "postgres")
    if user and password and host:
        return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{dbname}"
    return ""

# Optional Supabase
_SUPABASE_AVAILABLE = False
try:
    from supabase import create_client, Client
    _SUPABASE_AVAILABLE = True
except ImportError:
    logger.warning("supabase-py not installed. Will use SQLite fallback.")

# Optional PostgreSQL for schema creation
try:
    import psycopg2
    _PSYCOPG2_AVAILABLE = True
except ImportError:
    _PSYCOPG2_AVAILABLE = False


class _SQLiteCursorAdapter:
    """Wraps a SQLite cursor so execute() accepts %s placeholders (converts to ?)."""

    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)

    def execute(self, operation: str, parameters: Optional[tuple] = None):
        if parameters is not None and "%s" in operation:
            operation = operation.replace("%s", "?")
        return self._cursor.execute(operation, parameters or ())

    def executemany(self, operation: str, parameters_seq):
        if "%s" in operation:
            operation = operation.replace("%s", "?")
        return self._cursor.executemany(operation, parameters_seq)

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def lastrowid(self):
        return self._cursor.lastrowid


class _SQLiteConnectionAdapter:
    """Wraps a SQLite connection so cursor() returns our adapter."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def cursor(self):
        return _SQLiteCursorAdapter(self._conn.cursor())

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        return self._conn.close()


def _is_table_not_found_error(exc: Exception) -> bool:
    """Detect Supabase/PostgREST 'table not found' (e.g. PGRST205)."""
    msg = str(exc).lower()
    # Supabase client may wrap response in dict-like exception
    if hasattr(exc, "message"):
        msg = (getattr(exc, "message", "") or "").lower()
    if hasattr(exc, "code"):
        msg += " " + (getattr(exc, "code", "") or "").lower()
    return "pgrst205" in msg or "chat_sessions" in msg or "schema cache" in msg or "could not find the table" in msg


def _ensure_supabase_tables_via_postgres(db_url: str) -> bool:
    """Create app tables in Supabase using direct PostgreSQL. Returns True if successful."""
    if not db_url or not db_url.strip() or not _PSYCOPG2_AVAILABLE:
        return False
    try:
        conn = psycopg2.connect(db_url.strip(), connect_timeout=10)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT 'New Chat',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                last_message_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'UPLOADED',
                chunk_count INT NOT NULL DEFAULT 0,
                technology TEXT DEFAULT 'general',
                domain TEXT DEFAULT 'general',
                pdf_path TEXT,
                error TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_file_hash ON documents(file_hash)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_documents (
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                document_id TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (session_id, document_id)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_session_documents_session ON session_documents(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_session_documents_document ON session_documents(document_id)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON chat_messages(session_id, created_at)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT,
                chunk_type TEXT,
                retrieval_text TEXT,
                page_number INT,
                section_title TEXT,
                metadata_json TEXT,
                created_at BIGINT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_tables (
                table_id TEXT PRIMARY KEY,
                document_id TEXT,
                page_number INT,
                headers_json TEXT,
                rows_json TEXT,
                created_at BIGINT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_tables_document ON raw_tables(document_id)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_code_blocks (
                code_id TEXT PRIMARY KEY,
                document_id TEXT,
                language TEXT,
                code_text TEXT,
                page_number INT,
                created_at BIGINT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_code_document ON raw_code_blocks(document_id)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_images (
                image_id TEXT PRIMARY KEY,
                document_id TEXT,
                page_number INT,
                caption TEXT,
                image_path TEXT,
                created_at BIGINT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_raw_images_document ON raw_images(document_id)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks_fts (
                id BIGSERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                chunk_id TEXT,
                document_id TEXT,
                file_name TEXT,
                technology TEXT,
                domain TEXT,
                chunk_index TEXT,
                file_id TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_fts_document_id ON chunks_fts(document_id)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_nodes (
                node_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                node_type TEXT NOT NULL,
                title TEXT,
                content_preview TEXT,
                page_start INT DEFAULT 1,
                page_end INT DEFAULT 1,
                parent_node_id TEXT,
                depth INT DEFAULT 0,
                order_index INT DEFAULT 0,
                section_path_json TEXT DEFAULT '[]'
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_document_nodes_document ON document_nodes(document_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_document_nodes_parent ON document_nodes(parent_node_id)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_edges (
                edge_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                from_node_id TEXT NOT NULL,
                to_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'parent_of'
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_document_edges_document ON document_edges(document_id)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS usage_events (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT,
                message_id BIGINT,
                event_type TEXT NOT NULL DEFAULT 'chat_completion',
                query_preview TEXT,
                prompt_tokens INT,
                completion_tokens INT,
                total_tokens INT,
                model TEXT,
                provider TEXT,
                cost_usd DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_events_created ON usage_events(created_at DESC)"
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Supabase tables created or already exist (via SUPABASE_DB_URL)")
        return True
    except Exception as e:
        logger.warning("Could not create Supabase tables via direct DB: %s", e)
        return False


class Database:
    """
    Database manager: Supabase (REST API) or SQLite fallback.
    - For Supabase: use db.supabase.table('name').select().execute() etc.
    - For SQLite: use db.get_connection() with SQL queries.
    If Supabase returns table-not-found, tables are created via SUPABASE_DB_URL when set.
    """

    def __init__(self):
        self.engine: str = "sqlite"
        self.supabase: Optional[Any] = None
        self._db_path: Optional[str] = None

        supabase_url = getattr(settings, "SUPABASE_URL", "") or ""
        supabase_key = getattr(settings, "SUPABASE_KEY", "") or ""
        supabase_db_url = _get_supabase_db_url()

        if supabase_url.strip() and supabase_key.strip() and _SUPABASE_AVAILABLE:
            try:
                self.supabase = create_client(supabase_url.strip(), supabase_key.strip())
                self.supabase.table("chat_sessions").select("id").limit(1).execute()
                self.engine = "supabase"
                # usage_events was added after many deployments; ensure table exists without full reconnect failure
                try:
                    self.supabase.table("usage_events").select("id").limit(1).execute()
                except Exception as ue_e:
                    ue_str = str(ue_e)
                    missing_usage = (
                        _is_table_not_found_error(ue_e)
                        or "PGRST205" in ue_str
                        or (
                            "usage_events" in ue_str.lower()
                            and ("could not find" in ue_str.lower() or "schema cache" in ue_str.lower())
                        )
                    )
                    if missing_usage and supabase_db_url and _ensure_supabase_tables_via_postgres(supabase_db_url):
                        time.sleep(2)
                        try:
                            self.supabase.table("usage_events").select("id").limit(1).execute()
                            logger.info("usage_events table ensured (PostgreSQL migration)")
                        except Exception as retry_e:
                            logger.warning(
                                "usage_events still unavailable after schema ensure: %s. "
                                "Run the usage_events block in backend/supabase_schema.sql in Supabase SQL Editor, "
                                "or set SUPABASE_DB_URL for automatic CREATE TABLE.",
                                retry_e,
                            )
                    elif missing_usage:
                        logger.warning(
                            "Supabase is missing public.usage_events (token dashboard will stay empty). "
                            "Set SUPABASE_DB_URL (Pooler) so the API can run CREATE TABLE, or execute the "
                            "usage_events section of backend/supabase_schema.sql in the Supabase SQL Editor, "
                            "then add RLS for usage_events from backend/supabase_rls_policies.sql if you use the anon key."
                        )
                    else:
                        logger.warning("usage_events check failed (non-fatal): %s", ue_e)
                self._ensure_document_graph_tables_rest(supabase_db_url)
                logger.info("Database initialized (Supabase REST API)")
                return
            except Exception as e:
                err_str = str(e)
                # Detect table/schema missing (e.g. PGRST205)
                if _is_table_not_found_error(e) or "PGRST205" in err_str or "schema cache" in err_str:
                    if supabase_db_url and _ensure_supabase_tables_via_postgres(supabase_db_url):
                        time.sleep(2)  # Allow PostgREST to refresh schema cache
                        try:
                            self.supabase = create_client(supabase_url.strip(), supabase_key.strip())
                            self.supabase.table("chat_sessions").select("id").limit(1).execute()
                            self.engine = "supabase"
                            logger.info("Database initialized (Supabase REST API, tables created)")
                            return
                        except Exception as retry_e:
                            logger.warning("Supabase retry after schema create failed: %s", retry_e)
                    self.supabase = None
                else:
                    logger.warning("Supabase connection failed (%s). Falling back to SQLite.", e)
                    self.supabase = None

        # SQLite fallback
        backend_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = backend_dir / "data"
        data_dir.mkdir(exist_ok=True)
        self._db_path = str(data_dir / "chat.db")
        self._init_sqlite()
        logger.info("Database initialized (SQLite fallback at %s)", self._db_path)

    def _ensure_document_graph_tables_rest(self, supabase_db_url: str) -> None:
        """Warn or auto-create document_nodes/document_edges when using Supabase REST."""
        if self.engine != "supabase" or not self.supabase:
            return
        try:
            self.supabase.table("document_nodes").select("node_id").limit(1).execute()
        except Exception as e:
            err_str = str(e).lower()
            if "document_nodes" not in err_str and "document_edges" not in err_str:
                return
            if supabase_db_url and _ensure_supabase_tables_via_postgres(supabase_db_url):
                time.sleep(2)
                try:
                    self.supabase.table("document_nodes").select("node_id").limit(1).execute()
                    logger.info("document_nodes/document_edges tables ensured (PostgreSQL migration)")
                    return
                except Exception as retry_e:
                    logger.warning("document graph tables still missing after ensure: %s", retry_e)
            logger.warning(
                "Supabase is missing document_nodes/document_edges (Phase 1 graph). "
                "Run backend/supabase_schema.sql and backend/supabase_rls_policies.sql in the SQL Editor, "
                "or set SUPABASE_DB_URL for automatic CREATE TABLE."
            )

    @contextmanager
    def get_connection(self):
        """Get SQLite connection (only for SQLite backend)."""
        if self.engine != "sqlite":
            raise RuntimeError("get_connection() is only for SQLite. Use db.supabase for Supabase.")
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        wrapped = _SQLiteConnectionAdapter(conn)
        try:
            yield wrapped
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_sqlite(self):
        """Initialize SQLite schema."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        self._ensure_schema_sqlite(cur)
        conn.commit()
        conn.close()

    def _ensure_schema_sqlite(self, cursor):
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY, title TEXT DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'UPLOADED',
                chunk_count INT NOT NULL DEFAULT 0,
                technology TEXT DEFAULT 'general',
                domain TEXT DEFAULT 'general',
                pdf_path TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_documents (
                session_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, document_id),
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_documents_session ON session_documents(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_documents_document ON session_documents(document_id)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON chat_messages(session_id, created_at)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, document_id TEXT, chunk_type TEXT, retrieval_text TEXT,
                page_number INT, section_title TEXT, metadata_json TEXT, created_at INT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_tables (
                table_id TEXT PRIMARY KEY, document_id TEXT, page_number INT,
                headers_json TEXT, rows_json TEXT, created_at INT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_tables_document ON raw_tables(document_id)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_code_blocks (
                code_id TEXT PRIMARY KEY, document_id TEXT, language TEXT, code_text TEXT,
                page_number INT, created_at INT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_code_document ON raw_code_blocks(document_id)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_images (
                image_id TEXT PRIMARY KEY, document_id TEXT, page_number INT,
                caption TEXT, image_path TEXT, created_at INT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_images_document ON raw_images(document_id)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_nodes (
                node_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                node_type TEXT NOT NULL,
                title TEXT,
                content_preview TEXT,
                page_start INT DEFAULT 1,
                page_end INT DEFAULT 1,
                parent_node_id TEXT,
                depth INT DEFAULT 0,
                order_index INT DEFAULT 0,
                section_path_json TEXT DEFAULT '[]'
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_nodes_document ON document_nodes(document_id)")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_edges (
                edge_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                from_node_id TEXT NOT NULL,
                to_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'parent_of'
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_edges_document ON document_edges(document_id)")
        try:
            cursor.execute("PRAGMA table_info(chunks_fts)")
            cols = [row[1] for row in cursor.fetchall()]
            if cols and "chunk_id" not in cols:
                cursor.execute("DROP TABLE IF EXISTS chunks_fts")
        except Exception:
            pass
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, chunk_id UNINDEXED, document_id UNINDEXED, file_name UNINDEXED,
                technology UNINDEXED, domain UNINDEXED, chunk_index UNINDEXED, file_id UNINDEXED
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                message_id INTEGER,
                event_type TEXT NOT NULL DEFAULT 'chat_completion',
                query_preview TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                model TEXT,
                provider TEXT,
                cost_usd REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_events_created ON usage_events(created_at DESC)"
        )


_db: Optional[Database] = None


def get_database() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db
