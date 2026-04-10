"""
ingestion/chunker.py
Splits page documents into overlapping chunks (~700 tokens, 100 overlap).
Uses tiktoken for accurate token counting.
"""

import re
import tiktoken
from typing import List, Dict
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


# Use cl100k_base tokenizer (compatible with most modern LLMs)
_tokenizer = tiktoken.get_encoding("cl100k_base")

def _split_by_semester(text: str) -> List[str]:
    parts = re.split(r"(Semester \d+)", text)
    if len(parts) < 3:
        return [text]

    semester_blocks = []
    for label, content in zip(parts[1::2], parts[2::2]):
        if content.strip():
            semester_blocks.append(f"{label}\n{content}")

    return semester_blocks or [text]


def _split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    semester_blocks = _split_by_semester(text)
    all_chunks = []

    for block in semester_blocks:
        segments = [segment.strip() for segment in block.split("\n") if segment.strip()]
        current_text_parts = []
        current_token_count = 0

        for segment in segments:
            seg_tokens = _tokenizer.encode(segment)

            if current_text_parts and current_token_count + len(seg_tokens) > chunk_size:
                all_chunks.append("\n".join(current_text_parts))

                overlap_parts = []
                overlap_token_count = 0
                for part in reversed(current_text_parts):
                    part_tokens = _tokenizer.encode(part)
                    if overlap_token_count + len(part_tokens) > overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_token_count += len(part_tokens)

                current_text_parts = overlap_parts
                current_token_count = len(_tokenizer.encode("\n".join(current_text_parts))) if current_text_parts else 0

            current_text_parts.append(segment)
            current_token_count += len(seg_tokens)

        if current_text_parts:
            all_chunks.append("\n".join(current_text_parts))

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
