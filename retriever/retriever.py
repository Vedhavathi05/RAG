"""
Retriever
Vector retrieval + API reranking
STABLE — compatible with rag_service.py
"""

import os
import chromadb
import traceback
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(CURRENT_DIR, ".."))
)

CHROMA_PATH = os.path.join(PROJECT_ROOT, "rag_index", "chroma_db")

# ---------------------------------------------------
# HF EMBEDDINGS
# ---------------------------------------------------
HF_TOKEN = os.getenv("HF_API_TOKEN")

embed_client = InferenceClient(
    model="BAAI/bge-base-en-v1.5",
    token=HF_TOKEN,
)

# ---------------------------------------------------
# Embedder
# ---------------------------------------------------
class APIEmbedder:

    def encode(self, texts):
        try:
            emb = embed_client.feature_extraction(texts)

            # normalize embedding shape
            if isinstance(texts, str):
                emb = [emb]

            if isinstance(emb[0][0], list):
                emb = [e[0] for e in emb]

            return emb

        except Exception:
            traceback.print_exc()
            return [[0.0] * 768]


embedder = APIEmbedder()

# ---------------------------------------------------
# ChromaDB
# ---------------------------------------------------
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("medical_guidelines")


# ---------------------------------------------------
# Utils
# ---------------------------------------------------
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def deduplicate_chunks(results):

    seen = set()
    unique = []

    for item in results:
        key = item["text"][:150]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


# ---------------------------------------------------
# Retrieval
# ---------------------------------------------------
def retrieve(query, k=5):

    try:
        bge_query = (
            "Represent this sentence for searching relevant passages: "
            + query.strip()
        )

        query_vec = embedder.encode([bge_query])[0]

        candidate_k = max(50, k * 10)

        result = collection.query(
            query_embeddings=[query_vec],
            n_results=candidate_k,
            include=["documents", "metadatas", "embeddings"]
        )

        docs = result["documents"][0]
        metas = result["metadatas"][0]
        embeddings = result["embeddings"][0]
        ids = result["ids"][0]

        output = []

        for doc, cid, emb, meta in zip(docs, ids, embeddings, metas):

            meta = meta or {}

            score = cosine(query_vec, emb)

            output.append({
                "rank": 0,
                "chunk_id": cid,
                "text": doc or "",
                "score": float(score),
                "source": meta.get("source", "Unknown"),
                "position": int(meta.get("position", 0)),
            })

        output = deduplicate_chunks(output)
        output.sort(key=lambda x: x["score"], reverse=True)

        for i, item in enumerate(output):
            item["rank"] = i + 1

        return output[:k]

    except Exception:
        traceback.print_exc()

        return [{
            "rank": 1,
            "chunk_id": "error",
            "text": "Retrieval failed.",
            "score": 0.0,
            "source": "System",
            "position": 0,
        }]