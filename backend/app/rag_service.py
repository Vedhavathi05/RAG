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
class HFLLM:

    def __init__(self):
        self.url = (
            "https://api-inference.huggingface.co/models/"
            "mistralai/Mistral-7B-Instruct-v0.2"
        )

        self.headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }

        self.max_retries = 5
        print("[HF] Client ready")

    def __call__(self, prompt, max_new_tokens=160):

        print("[HF] Sending generation request...")
        sys.stdout.flush()

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.1,
                "do_sample": False,
                "return_full_text": False,
            },
        }

        for attempt in range(self.max_retries):

            try:
                print(f"[HF] Attempt {attempt+1}")

                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )

                print("[HF] Status:", response.status_code)

                try:
                    result = response.json()
                except Exception:
                    print("[HF] Non-JSON response, retrying...")
                    time.sleep(3)
                    continue

                # cold start
                if isinstance(result, dict) and "error" in result:
                    err = result["error"].lower()
                    print("[HF ERROR]", err)

                    if "loading" in err:
                        wait = 5 + attempt * 2
                        print(f"[HF] Model loading — waiting {wait}s")
                        time.sleep(wait)
                        continue

                    return [{"generated_text": ""}]

                if isinstance(result, list) and result:
                    print("[HF] Generation success")
                    return [{
                        "generated_text":
                        result[0].get("generated_text", "")
                    }]

            except Exception:
                print("[HF EXCEPTION]")
                traceback.print_exc()
                time.sleep(3)

        print("[HF] Max retries exceeded")
        return [{"generated_text": ""}]


# ===================================================
# TEXT HELPERS
# ===================================================
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

    print("[RAG] Building context...")
    context_parts = []
    size = 0

    for c in chunks:
        text = clean_text(c["text"])

        if size + len(text) > max_chars:
            break

        context_parts.append(text)
        size += len(text)

    print("[RAG] Context size:", size)
    return "\n".join(context_parts)


# ===================================================
# PROMPT
# ===================================================
def build_prompt(query, context):
    return f"""<s>[INST]
Answer using ONLY the context.

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
        sys.stdout.flush()

        try:
            # ---------------- RETRIEVE ----------------
            print("[RAG] Retrieving chunks...")
            chunks = retrieve(query)

            print("[RAG] Retrieved chunks:", len(chunks))

            if not chunks:
                print("[RAG] No chunks found")
                return {
                    "answer": "No relevant information found.",
                    "citations": [],
                    "original_query": query,
                }

            # ---------------- BUILD CONTEXT ----------------
            context_text = build_context(chunks)

            # ---------------- PROMPT ----------------
            print("[RAG] Building prompt...")
            prompt = build_prompt(query, context_text)

            # ---------------- GENERATION ----------------
            print("[RAG] Calling LLM...")
            output = self.llm(prompt)[0]["generated_text"].strip()

            print("[RAG] Raw output:", output[:120])

            answer = finish_sentence(remove_redundancy(output))

            if not answer:
                print("[RAG] Empty generation fallback used")
                answer = safe_preview(chunks[0]["text"], 200)

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

            print("[RAG] Answer ready")
            print("[RAG] ===============================\n")

            return {
                "answer": answer,
                "citations": citations,
                "original_query": query,
            }

        except Exception as e:
            print("\n[RAG FATAL ERROR]")
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