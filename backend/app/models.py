from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

class Message(BaseModel):
    """Individual message in a conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    citations: Optional[List[dict]] = None

class Conversation(BaseModel):
    """Conversation with history and context"""
    id: str
    title: str
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    context: Optional[str] = None  # Accumulated context for follow-ups

class ChatRequest(BaseModel):
    """Request to ask a question"""
    query: str
    conversation_id: str

class ChatResponse(BaseModel):
    """Response from chat endpoint"""
    id: str
    message: str
    citations: Optional[List[dict]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ConversationListItem(BaseModel):
    """Lightweight conversation info for list"""
    id: str
    title: str
    updated_at: datetime
    preview: str  # First 100 chars of last message
