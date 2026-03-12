"""
Debug script to compare retrieval and answers between methods.
Usage: python debug_comparison.py --question "Your question here"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from retrieval.vectorstore import query_chunks
from generation.llm import generate_answer

def test_question(question: str, top_k: int = 5):
    print("\n" + "=" * 70)
    print(f"🔍 Testing: {question}")
    print("=" * 70)
    
    # Step 1: Retrieve chunks
    print(f"\n1️⃣  RETRIEVAL (top_k={top_k}):")
    print("-" * 70)
    chunks = query_chunks(question, top_k=top_k)
    
    if not chunks:
        print("❌ No chunks retrieved!")
        return
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n   [{i}] Score: {chunk['score']:.4f} | {chunk['source']} (Page {chunk['page']})")
        print(f"       Text preview: {chunk['text'][:150]}...")
    
    # Step 2: Generate answer
    print(f"\n2️⃣  GENERATION:")
    print("-" * 70)
    answer = generate_answer(question, chunks)
    print(f"\n{answer}")
    
    print("\n" + "=" * 70)
    print("✅ Test complete")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug retrieval and generation")
    parser.add_argument("--question", "-q", required=True, help="Question to test")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()
    
    test_question(args.question, top_k=args.top_k)
    sys.exit(0)
