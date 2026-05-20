"""Chat session API routes - create, list, get, delete sessions and messages."""
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from app.config import settings
from app.models.schemas import (
    CreateSessionRequest,
    ChatSessionResponse,
    ChatMessageResponse,
    SessionMessagesResponse,
)
from app.services.chat_service import get_chat_service

router = APIRouter(prefix="/sessions", tags=["sessions"])

class SessionDocumentsRequest(BaseModel):
    file_ids: List[str] = []


def _session_to_response(session: dict, message_count: int = 0) -> ChatSessionResponse:
    """Convert session dict to response model."""
    return ChatSessionResponse(
        id=session["id"],
        title=session["title"],
        created_at=session["created_at"],
        last_message_at=session["last_message_at"],
        message_count=message_count,
    )


@router.post("/", response_model=ChatSessionResponse)
async def create_session(request: Optional[CreateSessionRequest] = Body(None)):
    """Create a new chat session."""
    title = "New Chat"
    if request is not None and request.title:
        title = request.title
    chat_service = get_chat_service()
    session = chat_service.create_session(title=title)
    return _session_to_response(session)


@router.get("/", response_model=list)
async def list_sessions():
    """List all chat sessions, newest first."""
    chat_service = get_chat_service()
    sessions = chat_service.get_all_sessions()
    result = []
    for s in sessions:
        count = chat_service.get_message_count(s["id"])
        result.append(_session_to_response(s, message_count=count))
    return result


@router.delete("/")
async def delete_all_sessions():
    """Delete all chat sessions and their messages."""
    chat_service = get_chat_service()
    deleted_count = await chat_service.delete_all_sessions_async()
    return {"message": "All sessions deleted", "deleted_count": deleted_count}


@router.get("/{session_id}", response_model=ChatSessionResponse)
async def get_session(session_id: str):
    """Get a session by ID."""
    chat_service = get_chat_service()
    session = chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    count = chat_service.get_message_count(session_id)
    return _session_to_response(session, message_count=count)


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its messages."""
    chat_service = get_chat_service()
    deleted = chat_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}


@router.patch("/{session_id}/title")
async def update_session_title(session_id: str, title: str = Body(..., embed=True)):
    """Update a session's title. Request body: {\"title\": \"New title\"}."""
    chat_service = get_chat_service()
    updated = chat_service.update_session_title(session_id, title)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Title updated", "title": title}


@router.get("/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str):
    """Get all messages for a session."""
    chat_service = get_chat_service()
    if not chat_service.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    messages = chat_service.get_messages(session_id)
    return SessionMessagesResponse(
        session_id=session_id,
        messages=[
            ChatMessageResponse(
                id=m["id"],
                session_id=m["session_id"],
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"],
            )
            for m in messages
        ],
        total_count=len(messages),
    )


@router.get("/{session_id}/documents", response_model=List[str])
async def get_session_documents(session_id: str):
    """Get the document scope for this session."""
    chat_service = get_chat_service()
    if not chat_service.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_service.get_session_documents(session_id)


@router.put("/{session_id}/documents", response_model=List[str])
async def set_session_documents(session_id: str, request: SessionDocumentsRequest):
    """Replace the document scope for this session."""
    chat_service = get_chat_service()
    if not chat_service.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_service.set_session_documents(session_id, request.file_ids or [])
