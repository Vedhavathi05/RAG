"""
RAG Service - grounded QA using retrieved chunks only
(MISTRAL-7B-INSTRUCT DEBUG VERSION FOR RENDER)
"""

import sys
import os
import re
import time
import traceback
import requests
from difflib import SequenceMatcher
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from groq import Groq

print("\n[RAG] Initializing RAG service...")

# ---------------------------------------------------
# Environment
# ---------------------------------------------------
load_dotenv(override=False)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_TOKEN = os.getenv("HF_API_TOKEN")

print("[RAG] HF token present:", bool(HF_TOKEN))

if not HF_TOKEN:
    raise RuntimeError(
        "HF_API_TOKEN not set. Add it in Render Environment Variables."
    )

# ---------------------------------------------------
# Add project root
# ---------------------------------------------------
BACKEND_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, "../.."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[RAG] Importing retriever...")
from retriever.retriever import retrieve
print("[RAG] ✓ Retriever imported")


# ===================================================
# HF API CLIENT
# ===================================================
# ===================================================
# HF API CLIENT
# ===================================================

class HFLLM:

    def __init__(self):
        print("[LLM] Initializing Groq client...")

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        print("[LLM] Client ready")

    def __call__(self, prompt, max_new_tokens=160):

        print("[LLM] Sending generation request...")
        sys.stdout.flush()

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=max_new_tokens,
            )

            return [{
                "generated_text":
                response.choices[0].message.content.strip()
            }]

        except Exception:
            print("[LLM EXCEPTION]")
            traceback.print_exc()
            return [{"generated_text": ""}]
        
def clean_text(text: str):
    text = " ".join(text.split())
    text = re.sub(r'^[A-Z\s]{3,}[-—:]\s*', '', text)
    text = re.sub(r'^GLOSSARY\s*', '', text, flags=re.I)
    text = re.sub(r'Heading:\s*', '', text, flags=re.I)
    return text.strip()


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def remove_redundancy(text, threshold=0.8):
    sentences = re.split(r'(?<=[.!?]) +', text)
    clean = []
    for s in sentences:
        if not any(similar(s, c) > threshold for c in clean):
            clean.append(s)
    return " ".join(clean)


def finish_sentence(text):
    text = text.strip()
    if not text:
        return ""
    if text.endswith((".", "!", "?")):
        return text
    return text + "."


def safe_preview(text, limit=200):
    text = clean_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last = cut.rfind(".")
    if last > 50:
        return cut[: last + 1]
    return cut + "..."


# ===================================================
# CONTEXT BUILDER
# ===================================================
def build_context(chunks, max_chars=1400):

    context_parts = []
    size = 0

    for c in chunks:
        text = clean_text(c["text"])

        if size + len(text) > max_chars:
            break

        context_parts.append(text)
        size += len(text)

    return "\n".join(context_parts)


# ===================================================
# PROMPT
# ===================================================
def build_prompt(query, context):
    return f"""<s>[INST]
Answer the question using ONLY the context below.
If the answer is not present in the context, say you do not know.

Context:
{context}

Question: {query}
[/INST]"""


# ===================================================
# RAG SERVICE
# ===================================================
class RAGService:

    def __init__(self):
        print("[RAG] Creating service...")
        self.llm = HFLLM()

    def answer(self, query: str, context: str = ""):

        print("\n[RAG] ===============================")
        print("[RAG] Query:", query)

        try:
            # ---------------- RETRIEVE ----------------
            chunks = retrieve(query)
            print("[RAG] Retrieved chunks:", len(chunks))

            # ✅ HANDLE NO CONTEXT (NEW)
            if not chunks or chunks[0]["chunk_id"] == "no_context":
                return {
                    "answer": "This question is outside the indexed knowledge base.",
                    "citations": [],
                    "original_query": query,
                }

            # ---------------- BUILD CONTEXT ----------------
            context_text = build_context(chunks)

            # ---------------- PROMPT ----------------
            prompt = build_prompt(query, context_text)

            # ---------------- GENERATION ----------------
            output = self.llm(prompt)[0]["generated_text"].strip()

            answer = finish_sentence(remove_redundancy(output))

            # ✅ NO FALLBACK SUMMARY (REMOVED)

            citations = [
                {
                    "rank": i + 1,
                    "source": c.get("source", "Unknown"),
                    "position": c.get("position", 0),
                    "score": c.get("score", 0),
                    "text": safe_preview(c.get("text", "")),
                }
                for i, c in enumerate(chunks)
            ]

            return {
                "answer": answer,
                "citations": citations,
                "original_query": query,
            }

        except Exception as e:
            traceback.print_exc()

            return {
                "answer": f"Error processing query: {str(e)}",
                "citations": [],
                "original_query": query,
                "error": True,
            }


# ---------------------------------------------------
# GLOBAL INSTANCE
# ---------------------------------------------------
rag_service = RAGService()
print("[RAG] ✓ Service ready\n")