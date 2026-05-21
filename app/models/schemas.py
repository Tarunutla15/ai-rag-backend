"""Pydantic models for request/response schemas."""
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


# ==================== CHAT SESSION SCHEMAS ====================

class CreateSessionRequest(BaseModel):
    """Request model for creating a new chat session."""
    title: Optional[str] = "New Chat"


class ChatSessionResponse(BaseModel):
    """Response model for a chat session."""
    id: str
    title: str
    created_at: str
    last_message_at: str
    message_count: Optional[int] = 0


class ChatMessageResponse(BaseModel):
    """Response model for a chat message."""
    id: int
    session_id: str
    role: str  # 'user' or 'assistant'
    content: str
    created_at: str


class SessionMessagesResponse(BaseModel):
    """Response model for getting session messages."""
    session_id: str
    messages: List[ChatMessageResponse]
    total_count: int


# ==================== UPLOAD SCHEMAS ====================

class UploadResponse(BaseModel):
    """Response model for single PDF upload."""
    message: str
    file_id: str
    chunks_created: int


class SingleFileResult(BaseModel):
    """Result for a single file in batch upload."""
    file_name: str
    file_id: Optional[str] = None
    chunks_created: int = 0
    technology: str = "general"
    domain: str = "general"
    status: str  # "success", "error", "duplicate"
    message: str


class BatchUploadResponse(BaseModel):
    """Response model for multiple PDF upload."""
    message: str
    total_files: int
    successful: int
    failed: int
    results: List[SingleFileResult]


class ChatRequest(BaseModel):
    """Request model for chat query."""
    query: str
    session_id: Optional[str] = None  # Optional: Creates new session if not provided
    file_id: Optional[str] = None
    # Preferred: allow scoping to multiple documents. If provided, overrides file_id.
    file_ids: Optional[List[str]] = None


class ChatFigure(BaseModel):
    """Extracted PDF figure served from /documents/{document_id}/images/{image_id}."""
    image_id: Optional[str] = None
    document_id: Optional[str] = None
    view_url: str
    caption: Optional[str] = None
    page_number: Optional[int] = None


class ChatResponse(BaseModel):
    """Response model for chat query."""
    answer: str
    session_id: str  # Return session_id so frontend can track it
    sources: Optional[List[str]] = None
    detected_technology: Optional[str] = None
    detected_domain: Optional[str] = None
    figures: Optional[List[ChatFigure]] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    message: str


# ==================== DOCUMENT LIBRARY SCHEMAS ====================

class DocumentInfo(BaseModel):
    document_id: str
    file_name: str
    status: Optional[str] = None
    chunk_count: Optional[int] = 0
    technology: Optional[str] = None
    domain: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pdf_path: Optional[str] = None


# ==================== USAGE / DASHBOARD ====================


class UsageSummary(BaseModel):
    chat_completions: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Optional[float] = None


class UsageEventItem(BaseModel):
    id: Optional[int] = None
    session_id: Optional[str] = None
    query_preview: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    cost_usd: Optional[float] = None
    created_at: str = ""


class DashboardUsageResponse(BaseModel):
    days: int
    summary: UsageSummary
    recent: List[UsageEventItem]

