"""Quick check of what's in the database."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from retrieval.vectorstore import query_chunks

# Test a semester 1 query
chunks = query_chunks('semester 1 courses', top_k=3)
print('=== SEMESTER 1 QUERY ===')
for i, c in enumerate(chunks, 1):
    print(f'\n[{i}] {c["source"]} | Page {c["page"]} | Score {c["score"]}')
    print(c['text'][:250])

# Test a semester 3 query
chunks = query_chunks('semester 3 courses', top_k=3)
print('\n\n=== SEMESTER 3 QUERY ===')
for i, c in enumerate(chunks, 1):
    print(f'\n[{i}] {c["source"]} | Page {c["page"]} | Score {c["score"]}')
    print(c['text'][:250])
