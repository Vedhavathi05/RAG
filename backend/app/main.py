"""
FastAPI Backend for RAG Chat Application
Provides REST API for chat, conversation management, and history
"""
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import traceback

print("[Main] Loading models...")

try:
    from app.models import ChatRequest, ChatResponse, Message
    print("[Main] ✓ Models imported")
except Exception as e:
    print(f"[Main] ✗ Failed to import models: {e}")
    traceback.print_exc()

try:
    from app.db import db
    print("[Main] ✓ Database initialized")
except Exception as e:
    print(f"[Main] ✗ Failed to initialize database: {e}")
    traceback.print_exc()

try:
    from app.rag_service import rag_service
    print("[Main] ✓ RAG service initialized")
except Exception as e:
    print(f"[Main] ✗ Failed to initialize RAG service: {e}")
    traceback.print_exc()

# Lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[FastAPI] Starting up...")
    yield
    # Shutdown
    print("[FastAPI] Shutting down...")

# Create FastAPI app
print("[Main] Creating FastAPI app...")
app = FastAPI(title="RAG Chat API", version="1.0.0", lifespan=lifespan)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://rag-chatbot-uqml.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[Main] ✓ FastAPI app configured with CORS")
import sys
sys.stdout.flush()

# =====================
# CONVERSATION ENDPOINTS
# =====================

@app.post("/api/conversations/create")
def create_conversation(title: str = "New Conversation"):
    """Create a new conversation"""
    try:
        conv = db.create_conversation(title)
        return {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "messages": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
def list_conversations():
    """List all conversations"""
    try:
        conversations = db.list_conversations()
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """Get a specific conversation with full history"""
    try:
        conv = db.get_conversation(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "context": conv.context,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "citations": msg.citations
                }
                for msg in conv.messages
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        success = db.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# CHAT ENDPOINTS
# =====================

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Send a message and get a response
    Handles context from previous messages for follow-ups
    """
    try:
        conv = db.get_conversation(request.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Add user message
        db.add_message(request.conversation_id, "user", request.query)
        
        # Get RAG answer with context
        rag_result = rag_service.answer(request.query, context=conv.context)
        
        if rag_result.get('error'):
            raise HTTPException(status_code=500, detail=rag_result['answer'])
        
        # Add assistant response
        db.add_message(
            request.conversation_id,
            "assistant",
            rag_result['answer'],
            citations=rag_result['citations']
        )
        
        # Update context for follow-up questions
        updated_context = conv.context + f"\n\nQ: {request.query}\nA: {rag_result['answer']}"
        if len(updated_context) > 2000:  # Limit context size
            updated_context = updated_context[-2000:]
        db.update_context(request.conversation_id, updated_context)
        
        return ChatResponse(
            id=request.conversation_id,
            message=rag_result['answer'],
            citations=rag_result['citations']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================
# HEALTH CHECK
# =====================

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

# =====================
# ROOT ENDPOINT
# =====================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "RAG Chat API",
        "version": "1.0.0",
        "docs": "/docs"
    }

print("[Main] ✓ All endpoints registered")
print("[Main] ✓ FastAPI app ready!")
import sys
sys.stdout.flush()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
