"""Check if semester 3 data exists in database."""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(__file__))

from retrieval.vectorstore import query_chunks

# Try very specific semester 3 queries
queries = [
    'semester 3 courses',
    'semester 3',
    'third semester',
]

for q in queries:
    print(f'\n=== Query: "{q}" ===')
    chunks = query_chunks(q, top_k=3)
    for i, c in enumerate(chunks, 1):
        print(f'[{i}] {c["source"]} | Page {c["page"]} | Score {c["score"]}')
        # Find what semester is in the text
        semesters = set(re.findall(r'Semester (\d)', c['text']))
        print(f'    Contains semesters: {semesters if semesters else "No semester labels"}')
        print(f'    Preview: {c["text"][:200]}...\n')
