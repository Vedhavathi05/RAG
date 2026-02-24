#!/usr/bin/env python
"""
Start script for the RAG backend server
Render / Railway deployment-safe
"""

import os
import sys
import traceback


def main():
    try:
        print("\n🚀 Starting RAG Chat Backend...")
        print("Loading modules...\n")

        # ---------------------------------------------------
        # Deployment safety settings
        # ---------------------------------------------------
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        # ---------------------------------------------------
        # Ensure backend folder is import root
        # ---------------------------------------------------
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, CURRENT_DIR)
        sys.path.insert(0, os.path.dirname(CURRENT_DIR))

        # ---------------------------------------------------
        # Import uvicorn AFTER env setup
        # ---------------------------------------------------
        import uvicorn

        print("✅ Uvicorn loaded")
        print("📚 API Docs: /docs")
        print("⏳ Models load on first request\n")

        sys.stdout.flush()

        # Render provides PORT automatically
        port = int(os.environ.get("PORT", 8000))

        # FastAPI app location
        APP_MODULE = "app.main:app"

        print(f"🌐 Starting server on port {port}")
        print(f"📦 App module: {APP_MODULE}\n")

        uvicorn.run(
            APP_MODULE,
            host="0.0.0.0",
            port=port,
            log_level="info",
            reload=False,
            access_log=True,
        )

    except Exception as e:
        print("\n❌ ERROR: Failed to start backend!")
        print(f"\nError: {type(e).__name__}: {str(e)}")
        print("\nFull traceback:\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()