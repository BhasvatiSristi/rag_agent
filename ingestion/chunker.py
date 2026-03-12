"""
ingestion/chunker.py
Splits page documents into overlapping chunks (~700 tokens, 100 overlap).
Uses tiktoken for accurate token counting.
"""

import tiktoken
from typing import List, Dict
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


# Use cl100k_base tokenizer (compatible with most modern LLMs)
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def _split_by_semester(text: str) -> List[str]:
    """
    Split text by semester boundaries first.
    Returns list of (semester_label, semester_text) tuples.
    """
    import re
    
    # Split by Semester N patterns
    parts = re.split(r'(Semester \d+)', text)
    
    semester_blocks = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            semester_label = parts[i]
            semester_content = parts[i + 1]
            if semester_content.strip():
                semester_blocks.append(f"{semester_label}\n{semester_content}")
    
    return semester_blocks if semester_blocks else [text]


def _split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split a text string into overlapping token-based chunks.
    Strategy: First split by semester boundaries, then split each semester by tokens.
    """
    # First, split by semester to avoid mixing semesters in chunks
    semester_blocks = _split_by_semester(text)
    
    all_chunks = []
    
    for block in semester_blocks:
        # For each semester block, split by lines
        segments = [s.strip() for s in block.split("\n") if s.strip()]
        
        chunks = []
        current_tokens = []
        current_text_parts = []

        for segment in segments:
            seg_tokens = _tokenizer.encode(segment)

            # If adding this segment exceeds chunk_size, flush current chunk
            if len(current_tokens) + len(seg_tokens) > chunk_size and current_text_parts:
                chunks.append("\n".join(current_text_parts))

                # Keep overlap: retain last N tokens worth of content
                overlap_parts = []
                overlap_token_count = 0
                for part in reversed(current_text_parts):
                    part_tokens = _tokenizer.encode(part)
                    if overlap_token_count + len(part_tokens) <= overlap:
                        overlap_parts.insert(0, part)
                        overlap_token_count += len(part_tokens)
                    else:
                        break

                current_text_parts = overlap_parts
                current_tokens = _tokenizer.encode("\n".join(current_text_parts))

            current_text_parts.append(segment)
            current_tokens.extend(seg_tokens)

        # Flush remaining
        if current_text_parts:
            chunks.append("\n".join(current_text_parts))
        
        all_chunks.extend(chunks)

    return all_chunks if all_chunks else [text]


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Take page-level documents and split into smaller chunks.

    Each chunk:
    {
        "text": "...",
        "source": "file.pdf",
        "page": 1,
        "chunk_id": "file.pdf_p1_c0"
    }
    """
    all_chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        page = doc["page"]

        chunks = _split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page,
                "chunk_id": f"{source}_p{page}_c{i}",
            })

    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks
