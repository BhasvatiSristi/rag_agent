"""
test_query.py
Test the RAG pipeline from the terminal WITHOUT starting the API.
Runs the full retrieval + generation chain directly.

Usage:
    python test_query.py
    python test_query.py --question "What subjects are in semester 3?"
    python test_query.py --top-k 3
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from retrieval.vectorstore import query_chunks, collection_size
from generation.llm import generate_answer


SAMPLE_QUESTIONS = [
    "What subjects are in semester 3?",
    "How many credits does Data Structures have?",
    "Which course teaches machine learning?",
    "What electives are offered in semester 6?",
    "What is the prerequisite for Theory of Machines and Design?",
    "How many total credits are needed to complete the B.Tech program?",
    "What programming languages are taught in the first year?",
]


def run_query(question: str, top_k: int = 5, verbose: bool = False):
    print("\n" + "=" * 60)
    print(f"❓ Question: {question}")
    print("=" * 60)

    if collection_size() == 0:
        print("❌ ChromaDB is empty. Run ingest_pipeline.py first.")
        return

    # Retrieve
    chunks = query_chunks(question, top_k=top_k)

    if verbose:
        print(f"\n📚 Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  [{i}] Source: {chunk['source']} | Page: {chunk['page']} | Score: {chunk['score']}")
            print(f"      {chunk['text'][:200]}...")

    # Generate
    print("\n🤖 Generating answer...\n")
    answer = generate_answer(question, chunks)
    print(f"💬 Answer:\n{answer}")

    print(f"\n📎 Sources used:")
    seen = set()
    for chunk in chunks:
        key = (chunk['source'], chunk['page'])
        if key not in seen:
            seen.add(key)
            print(f"   • {chunk['source']} — Page {chunk['page']} (score: {chunk['score']})")


def interactive_mode(top_k: int):
    print("\n" + "=" * 60)
    print("🎓 Curriculum RAG — Interactive Mode")
    print("   Type your question and press Enter.")
    print("   Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    while True:
        try:
            question = input("\n❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        run_query(question, top_k=top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the RAG pipeline")
    parser.add_argument("--question", "-q", help="Single question to ask")
    parser.add_argument("--top-k", "-k", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true", help="Show retrieved chunks")
    parser.add_argument("--sample", "-s", action="store_true", help="Run all sample questions")
    args = parser.parse_args()

    if args.sample:
        for q in SAMPLE_QUESTIONS:
            run_query(q, top_k=args.top_k, verbose=args.verbose)
    elif args.question:
        run_query(args.question, top_k=args.top_k, verbose=args.verbose)
    else:
        interactive_mode(top_k=args.top_k)
    
    sys.exit(0)
