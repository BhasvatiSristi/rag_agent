"""
ingestion/ingest_pipeline.py
Run this ONCE to ingest all PDFs into ChromaDB.

Usage:
    python -m ingestion.ingest_pipeline
    python -m ingestion.ingest_pipeline --data-dir ./data/raw
"""

import argparse
import time
import sys
from pathlib import Path

# Make sure project root is on path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.loader import load_all_pdfs
from ingestion.chunker import chunk_documents
from retrieval.vectorstore import add_chunks, collection_size
from retrieval.bm25 import build_bm25_index


def run_ingestion(data_dir: str):
    start = time.time()
    print("=" * 50)
    print("🚀 Starting curriculum ingestion pipeline")
    print("=" * 50)

    # Step 1: Load PDFs
    print("\n📄 Step 1: Loading PDFs...")
    documents = load_all_pdfs(data_dir)

    # Step 2: Chunk
    print("\n✂️  Step 2: Chunking documents...")
    chunks = chunk_documents(documents)

    # Step 3: Embed + Store
    print("\n🗄️  Step 3: Embedding and storing in ChromaDB...")
    add_chunks(chunks)

    # Step 4: BM25 Index
    print("\n🔎 Step 4: Building BM25 index...")
    build_bm25_index(chunks)

    elapsed = time.time() - start
    print("\n" + "=" * 50)
    print(f"✅ Ingestion complete in {elapsed:.1f}s")
    print(f"   Total chunks in DB: {collection_size()}")
    print("=" * 50)
    print("\nYou can now start the API:")
    print("  uvicorn backend.main:app --reload --port 8000\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest curriculum PDFs into ChromaDB")
    parser.add_argument(
        "--data-dir",
        default="./data/raw",
        help="Directory containing PDF files (default: ./data/raw)",
    )
    args = parser.parse_args()
    run_ingestion(args.data_dir)
    sys.exit(0)
