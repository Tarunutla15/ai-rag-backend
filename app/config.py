"""Configuration management for the application."""
from typing import Optional, Tuple
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration (embeddings always use OpenAI; chat can use Groq — see LLM_PROVIDER)
    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    OPENAI_EMBEDDING_MODEL: str

    # Chat / completion LLM: "openai" or "groq" (Groq uses OpenAI-compatible API; embeddings stay OpenAI-only)
    LLM_PROVIDER: str
    GROQ_API_KEY: Optional[str] = None
    GROQ_API_BASE: Optional[str] = None
    GROQ_MODEL: Optional[str] = None
    # Groq on-demand tier enforces low TPM; oversized prompts return 413 / rate_limit_exceeded.
    # Total prompt ≈ system + history + retrieved context — keep context+history bounded (~4 chars ≈ 1 token).
    GROQ_MAX_CONTEXT_CHARS: int = 16000
    GROQ_MAX_CHAT_HISTORY_CHARS: int = 4000
    GROQ_MAX_CHARS_PER_CHUNK: int = 3200
    GROQ_RERANK_MAX_CHUNKS: int = 22
    GROQ_RERANK_SNIPPET_CHARS: int = 1200
    # USD per 1M tokens for dashboard cost when LLM_PROVIDER=groq (defaults ≈ Groq on-demand
    # llama-3.3-70b-versatile; see https://groq.com/pricing — override if you use another model/tier).
    GROQ_INPUT_USD_PER_1M: float = 0.59
    GROQ_OUTPUT_USD_PER_1M: float = 0.79

    # Optional cost config (USD per 1M tokens). If unset, cost logging is skipped.
    # Provide these in .env if you want cost estimates in logs.
    OPENAI_INPUT_USD_PER_1M: Optional[float] = None
    OPENAI_OUTPUT_USD_PER_1M: Optional[float] = None
    
    # Zilliz Configuration
    ZILLIZ_URI: str
    ZILLIZ_TOKEN: str
    ZILLIZ_COLLECTION_NAME: str = "pdf_chatbot_collection"
    # Large inserts: smaller batches + long timeouts reduce WriteTimeout errors to Zilliz Cloud.
    ZILLIZ_INSERT_BATCH_SIZE: int = 40
    ZILLIZ_INSERT_TIMEOUT_SECONDS: float = 300.0
    ZILLIZ_DELETE_TIMEOUT_SECONDS: float = 60.0

    # Parallel purge when deleting library documents (Zilliz + Supabase run concurrently)
    DELETE_PARALLEL_WORKERS: int = 4

    # Concurrent batch uploads (each file runs full async ingest pipeline)
    BATCH_UPLOAD_CONCURRENT: int = 2
    
    # Application Configuration
    UPLOAD_DIR: str = "uploads"
    DOCUMENT_REGISTRY_FILE: str = "document_registry.json"
    # Text chunking: target ~300-600 tokens (1 token ≈ 4 chars). Overlap 10-15%.
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CHUNK_TARGET_TOKENS: int = 500  # If > 0, overrides: size = this * 4 chars, overlap = size * CHUNK_OVERLAP_PERCENT / 100
    CHUNK_OVERLAP_PERCENT: int = 12
    MAX_CHUNKS_TO_RETRIEVE: int = 10
    # Keep original PDF and extracted images on disk after ingest (object-storage style).
    STORE_PDF_AFTER_INGEST: bool = True
    STORE_EXTRACTED_IMAGES: bool = True
    # Vision captions for cropped PDF images at ingest (Groq Llama 4 Scout default). Costs per image.
    ENABLE_VISION_IMAGE_CAPTIONS: bool = False
    VISION_PROVIDER: str = "groq"  # groq | openai | gemini
    # Optional: separate Groq key for vision only; falls back to GROQ_API_KEY when unset
    GROQ_VISION_API_KEY: Optional[str] = None
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_VISION_MODEL: str = "gemini-2.0-flash"
    VISION_CAPTION_MODEL: str = "gpt-4o-mini"  # used when VISION_PROVIDER=openai
    MAX_VISION_CAPTIONS_PER_DOCUMENT: int = 30

    # Reranker: fetch a wide pool, score top N with LLM, pass best to answer LLM
    ENABLE_RERANKER: bool = True
    RERANK_INITIAL_TOP_K: int = 45   # Vector hybrid pool size (per query)
    RERANK_TOP_N: int = 12           # Chunks sent to answer LLM after rerank
    
    # Hybrid retrieval (vector + keyword/BM25)
    ENABLE_KEYWORD_SEARCH: bool = True
    HYBRID_VECTOR_WEIGHT: float = 0.7
    HYBRID_KEYWORD_WEIGHT: float = 0.3
    KEYWORD_TOP_K: int = 45          # Match RERANK_INITIAL_TOP_K so keyword hits enter rerank pool

    # Self-refining retrieval loop
    ENABLE_QUERY_REWRITE: bool = True

    # When the client sends no file_ids and the session has no scope, search all INGESTED
    # documents in the library (retrieval + reranker still pick the best chunks per query).
    CHAT_AUTO_SCOPE_ALL_INGESTED: bool = True
    # Max documents in a single pooled vector+keyword search (in-filter). Above this, cap with a warning.
    CHAT_SCOPED_POOL_MAX_DOCUMENTS: int = 40

    # Phase 2: LLM disambiguation for figure ↔ paragraph links (multiple figures per page)
    ENABLE_LLM_CROSS_TYPE_LINKS: bool = False

    # Phase 3: prefix chunk text for embeddings (retrieval_text unchanged for display)
    ENABLE_CONTEXTUAL_EMBEDDINGS: bool = True
    # Keyword FTS: index document/section/page tokens (aligns with vector contextual prefixes)
    ENABLE_FTS_CONTEXTUAL_INDEX: bool = True
    # Hybrid retrieval: boost heading chunks when query matches section titles
    HEADING_RETRIEVAL_BOOST: float = 0.12
    # Optional Anthropic-style one-line context per chunk at ingest (extra LLM cost)
    ENABLE_LLM_CONTEXTUAL_EMBEDDINGS: bool = False

    # Phase 4: graph-aware expansion after retrieval/rerank
    ENABLE_GRAPH_CONTEXT_EXPANSION: bool = True
    CONTEXT_EXPANSION_DECAY: float = 0.85
    CONTEXT_EXPANSION_MAX_ADD: int = 14
    CONTEXT_EXPANSION_SIBLING_MARGIN: float = 0.92
    CONTEXT_EXPANSION_MAX_CHARS: int = 28000

    # Parent–child retrieval (Small-to-Big): search children, send section parent to LLM
    ENABLE_PARENT_CHILD_RETRIEVAL: bool = True
    PARENT_SECTION_MAX_CHARS: int = 12000
    PARENT_CONTEXT_MAX_CHARS: int = 10000
    # "prepend" = section + matched excerpt; "replace" = section only
    PARENT_CONTEXT_MODE: str = "prepend"
    # Max distinct section parents expanded into LLM context per turn
    PARENT_CONTEXT_MAX_PARENTS: int = 8
    # Parents with more children than this are treated as document-wide — keep child hits instead
    PARENT_MEGA_CHILD_THRESHOLD: int = 12

    # Upload-time technology/domain tagging
    # MODE: "prompt" = one LLM call (ChatGPT-style) then fallback heuristics on failure;
    #       "heuristic" = resume/filename/keywords first; optional LLM at end if USE_LLM.
    DOCUMENT_CLASSIFY_MODE: str = "heuristic"
    DOCUMENT_CLASSIFY_MIN_SCORE: int = 10  # Min weighted keyword score (heuristic content path)
    DOCUMENT_CLASSIFY_USE_LLM: bool = False  # In heuristic mode: LLM only when keywords yield general
    DOCUMENT_CLASSIFY_LLM_MAX_CHARS: int = 6000  # Text excerpt sent to the classifier model
    # Sum of resume/CV heuristic signals; above this → technology/domain general (see document_classifier)
    DOCUMENT_CLASSIFY_RESUME_THRESHOLD: int = 9
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Supabase (REST API). If empty or unreachable, app falls back to SQLite (backend/data/chat.db).
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    # Optional: service_role key for Storage uploads (bypasses RLS). Falls back to SUPABASE_KEY.
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    # Optional: direct PostgreSQL for creating missing tables. Use either SUPABASE_DB_URL
    # or the separate vars below (same as psycopg2.connect(user=..., password=..., host=..., port=..., dbname=...)).
    SUPABASE_DB_URL: str = ""
    SUPABASE_DB_USER: str = ""
    SUPABASE_DB_PASSWORD: str = ""
    SUPABASE_DB_HOST: str = ""
    SUPABASE_DB_PORT: str = "5432"
    SUPABASE_DB_NAME: str = "postgres"
    # Supabase Storage (bucket for PDFs + extracted images; survives Render redeploys)
    USE_SUPABASE_STORAGE: bool = True
    SUPABASE_STORAGE_BUCKET: str = "rag-uploads"
    # S3-compatible endpoint (optional; for AWS CLI — app uses supabase-py REST API)
    SUPABASE_S3_ENDPOINT: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


def get_completion_client_config() -> Tuple[str, str, Optional[str]]:
    """
    OpenAI-compatible chat completions: (api_key, model, base_url).
    base_url is set for Groq; None for native OpenAI.
    Embeddings always use OPENAI_API_KEY + OPENAI_EMBEDDING_MODEL separately.
    """
    prov = (settings.LLM_PROVIDER or "").lower().strip()
    if prov not in ("openai", "groq"):
        raise ValueError('LLM_PROVIDER must be "openai" or "groq".')
    if prov == "groq":
        key = (settings.GROQ_API_KEY or "").strip()
        base = (settings.GROQ_API_BASE or "").strip().rstrip("/")
        model = (settings.GROQ_MODEL or "").strip()
        if not key or not base or not model:
            raise ValueError(
                "LLM_PROVIDER=groq requires GROQ_API_KEY, GROQ_API_BASE, and GROQ_MODEL in .env. "
                "Embeddings still use OPENAI_API_KEY + OPENAI_EMBEDDING_MODEL."
            )
        return key, model, base
    # openai
    return settings.OPENAI_API_KEY, settings.OPENAI_MODEL, None


# Global settings instance
try:
    settings = Settings()
    logger.info("Settings loaded successfully")
    _prov = (settings.LLM_PROVIDER or "").lower().strip()
    logger.info("LLM provider (chat/rerank/classifier): %s", _prov)
    if _prov == "groq":
        logger.info("Groq model: %s (embeddings: OpenAI %s)", settings.GROQ_MODEL, settings.OPENAI_EMBEDDING_MODEL)
    else:
        logger.info(f"OpenAI chat model: {settings.OPENAI_MODEL}")
    logger.info(f"Embedding Model: {settings.OPENAI_EMBEDDING_MODEL}")
    logger.info(f"Zilliz Collection: {settings.ZILLIZ_COLLECTION_NAME}")
    try:
        from app.services.blob_storage import storage_enabled, _bucket_name

        if storage_enabled():
            from app.services.blob_storage import (
                _bucket_name,
                ensure_storage_policies,
                storage_uses_service_role,
            )

            if storage_uses_service_role():
                logger.info(
                    "Blob storage: Supabase bucket=%s (service role key)",
                    _bucket_name(),
                )
            else:
                ensure_storage_policies()
                logger.info(
                    "Blob storage: Supabase bucket=%s (publishable key; policies auto-applied if Postgres URL set)",
                    _bucket_name(),
                )
        else:
            logger.info("Blob storage: local UPLOAD_DIR only (Supabase storage disabled)")
    except Exception:
        pass
    logger.info(
        "Document ingest classify mode: %s",
        getattr(settings, "DOCUMENT_CLASSIFY_MODE", "heuristic"),
    )
except Exception as e:
    logger.error(f"Failed to load settings: {type(e).__name__}: {str(e)}")
    raise
