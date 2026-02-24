import json
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict

from app.models import Conversation, Message


class ConversationDB:
    """
    Simple JSON-based conversation storage.

    NOTE:
    Render free tier uses ephemeral storage.
    Conversations reset on redeploy/restart.
    """

    # ---------------------------------------------------
    # Init
    # ---------------------------------------------------
    def __init__(self, db_path: Optional[str] = None):

        # Resolve absolute backend path (cloud-safe)
        if db_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "conversations")

        self.db_path = db_path
        os.makedirs(self.db_path, exist_ok=True)

    # ---------------------------------------------------
    # Helpers
    # ---------------------------------------------------
    def _get_conversation_file(self, conversation_id: str) -> str:
        return os.path.join(self.db_path, f"{conversation_id}.json")

    # ---------------------------------------------------
    # Create
    # ---------------------------------------------------
    def create_conversation(self, title: str = "New Conversation") -> Conversation:
        conversation_id = str(uuid.uuid4())

        conversation = Conversation(
            id=conversation_id,
            title=title,
            messages=[],
            context=""
        )

        self.save_conversation(conversation)
        return conversation

    # ---------------------------------------------------
    # Save
    # ---------------------------------------------------
    def save_conversation(self, conversation: Conversation):

        path = self._get_conversation_file(conversation.id)

        data = {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "context": conversation.context,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "citations": msg.citations,
                }
                for msg in conversation.messages
            ],
        }

        # Atomic write (prevents corruption)
        tmp_path = path + ".tmp"

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        os.replace(tmp_path, path)

    # ---------------------------------------------------
    # Load
    # ---------------------------------------------------
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:

        path = self._get_conversation_file(conversation_id)

        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        messages = [
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
                citations=m.get("citations"),
            )
            for m in data.get("messages", [])
        ]

        return Conversation(
            id=data["id"],
            title=data["title"],
            messages=messages,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            context=data.get("context", ""),
        )

    # ---------------------------------------------------
    # List
    # ---------------------------------------------------
    def list_conversations(self) -> List[Dict]:

        conversations = []

        if not os.path.exists(self.db_path):
            return conversations

        for filename in os.listdir(self.db_path):

            if not filename.endswith(".json"):
                continue

            conversation_id = filename[:-5]
            conv = self.get_conversation(conversation_id)

            if not conv:
                continue

            preview = ""

            if conv.messages:
                last_msg = next(
                    (m for m in reversed(conv.messages)
                     if m.role == "assistant"),
                    None,
                )

                if last_msg:
                    preview = last_msg.content[:100]

            conversations.append({
                "id": conv.id,
                "title": conv.title,
                "updated_at": conv.updated_at.isoformat(),
                "preview": preview,
            })

        conversations.sort(
            key=lambda x: x["updated_at"],
            reverse=True
        )

        return conversations

    # ---------------------------------------------------
    # Add message
    # ---------------------------------------------------
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: Optional[List] = None,
    ):

        conv = self.get_conversation(conversation_id)
        if not conv:
            return None

        message = Message(
            role=role,
            content=content,
            citations=citations,
        )

        conv.messages.append(message)
        self.save_conversation(conv)

        return message

    # ---------------------------------------------------
    # Context update
    # ---------------------------------------------------
    def update_context(self, conversation_id: str, context: str):

        conv = self.get_conversation(conversation_id)

        if conv:
            conv.context = context
            self.save_conversation(conv)

    # ---------------------------------------------------
    # Delete
    # ---------------------------------------------------
    def delete_conversation(self, conversation_id: str) -> bool:

        path = self._get_conversation_file(conversation_id)

        if os.path.exists(path):
            os.remove(path)
            return True

        return False


# ---------------------------------------------------
# Global instance
# ---------------------------------------------------
db = ConversationDB()