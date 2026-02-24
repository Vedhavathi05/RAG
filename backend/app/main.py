"""
FastAPI Backend for RAG Chat Application
Provides REST API for chat, conversation management, and history
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import traceback
import sys

print("\n[Main] Loading models...")

# --------------------------------------------------
# Imports with startup diagnostics
# --------------------------------------------------
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

sys.stdout.flush()

# --------------------------------------------------
# Lifespan
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[FastAPI] Starting up...")
    yield
    print("[FastAPI] Shutting down...")


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
print("[Main] Creating FastAPI app...")
app = FastAPI(title="RAG Chat API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[Main] ✓ FastAPI app configured with CORS")
sys.stdout.flush()

# ==================================================
# CONVERSATION ENDPOINTS
# ==================================================

@app.post("/api/conversations/create")
def create_conversation(title: str = "New Conversation"):
    try:
        print("[API] Creating conversation")
        conv = db.create_conversation(title)

        return {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "messages": []
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations")
def list_conversations():
    try:
        print("[API] Listing conversations")
        conversations = db.list_conversations()
        return {"conversations": conversations}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    try:
        print(f"[API] Fetch conversation {conversation_id}")

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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    try:
        print(f"[API] Delete conversation {conversation_id}")

        success = db.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"status": "deleted"}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================================================
# CHAT ENDPOINT (HEAVY DEBUG VERSION)
# ==================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    print("\n================ CHAT REQUEST ================")
    print("[CHAT] conversation_id:", request.conversation_id)
    print("[CHAT] query:", request.query)
    sys.stdout.flush()

    try:
        # ---------------------------------------------
        # Load conversation
        # ---------------------------------------------
        print("[CHAT] Loading conversation...")
        conv = db.get_conversation(request.conversation_id)

        if not conv:
            print("[CHAT ERROR] Conversation not found")
            raise HTTPException(status_code=404, detail="Conversation not found")

        print("[CHAT] Conversation loaded")

        # ---------------------------------------------
        # Save user message
        # ---------------------------------------------
        print("[CHAT] Saving user message...")
        db.add_message(request.conversation_id, "user", request.query)

        # ---------------------------------------------
        # CALL RAG
        # ---------------------------------------------
        print("[CHAT] Calling RAG service...")
        sys.stdout.flush()

        rag_result = rag_service.answer(
            request.query,
            context=conv.context
        )

        print("[CHAT] RAG result keys:", list(rag_result.keys()))
        sys.stdout.flush()

        # DO NOT CRASH API ON RAG FAILURE
        if rag_result.get("error"):
            print("[CHAT WARNING] RAG returned error:")
            print(rag_result["answer"])

        # ---------------------------------------------
        # Save assistant message
        # ---------------------------------------------
        print("[CHAT] Saving assistant response...")
        db.add_message(
            request.conversation_id,
            "assistant",
            rag_result.get("answer", ""),
            citations=rag_result.get("citations", [])
        )

        # ---------------------------------------------
        # Update context safely
        # ---------------------------------------------
        old_context = conv.context or ""

        updated_context = (
            old_context
            + f"\n\nQ: {request.query}\nA: {rag_result.get('answer','')}"
        )

        if len(updated_context) > 2000:
            updated_context = updated_context[-2000:]

        db.update_context(request.conversation_id, updated_context)

        print("[CHAT] Response complete")
        print("================================================\n")
        sys.stdout.flush()

        return ChatResponse(
            id=request.conversation_id,
            message=rag_result.get("answer", ""),
            citations=rag_result.get("citations", [])
        )

    except HTTPException:
        raise

    except Exception as e:
        print("\n[CHAT FATAL ERROR]")
        traceback.print_exc()
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=str(e))


# ==================================================
# HEALTH
# ==================================================

@app.get("/api/health")
def health_check():
    print("[API] Health check OK")
    return {"status": "ok"}


# ==================================================
# ROOT
# ==================================================

@app.get("/")
def root():
    return {
        "name": "RAG Chat API",
        "version": "1.0.0",
        "docs": "/docs"
    }


print("[Main] ✓ All endpoints registered")
print("[Main] ✓ FastAPI app ready!")
sys.stdout.flush()

# --------------------------------------------------
# Local Run
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)