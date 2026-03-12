"""
debug_retrieval.py
Deep dive into what chunks are retrieved and how the LLM processes them.

Usage:
    python debug_retrieval.py --question "What courses are in semester 3 for mechanical branch?"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from retrieval.vectorstore import query_chunks
from generation.llm import build_prompt, SYSTEM_PROMPT

def debug_question(question: str, top_k: int = 5):
    print("\n" + "=" * 80)
    print(f"❓ QUESTION: {question}")
    print("=" * 80)
    
    # Get chunks
    chunks = query_chunks(question, top_k=top_k)
    
    if not chunks:
        print("❌ No chunks retrieved!")
        return
    
    print(f"\n📊 Retrieved {len(chunks)} chunks:\n")
    
    # Show each chunk in detail
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'='*80}")
        print(f"[CHUNK {i}] Source: {chunk['source']} | Page: {chunk['page']} | Score: {chunk['score']:.4f}")
        print(f"{'='*80}")
        print(chunk['text'][:500])  # Show first 500 chars
        if len(chunk['text']) > 500:
            print("... [truncated]")
    
    # Show what the LLM will see
    print(f"\n{'='*80}")
    print("💬 WHAT THE LLM WILL RECEIVE:")
    print(f"{'='*80}\n")
    
    prompt = build_prompt(question, chunks)
    print(prompt)
    
    print("\n" + "=" * 80)
    print("SYSTEM PROMPT:")
    print("=" * 80)
    print(SYSTEM_PROMPT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug retrieval and LLM input")
    parser.add_argument("--question", "-q", required=True, help="Question to debug")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()
    
    debug_question(args.question, top_k=args.top_k)
