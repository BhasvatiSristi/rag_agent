"""
generation/generator.py
Prompt building and Mistral LLM call for grounded curriculum answers.
"""

import os
import requests
from typing import List, Dict

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

NO_ANSWER_MSG = "I could not find supporting information in the curriculum documents."

SYSTEM_PROMPT = """You are a helpful assistant for a college curriculum information system at IIITDM Kancheepuram.

The context you receive contains curriculum data extracted from PDF tables. Table rows look like:
  S.No: 1 | Course Code: MA1000 | Course Name: Calculus | Category: BSC | L: 3 | T: 1 | P: 0 | C: 4

RULES:
1. Answer ONLY using the information in the CONTEXT provided.
2. Do NOT invent course names, codes, or credits not present in the context.
3. The context may contain data from multiple semesters or branches. Read carefully and match exactly the semester/branch the user asked about.
4. When listing courses, always include: Course Name, Course Code, and Credits (C value).
5. Present course lists as a clean numbered or bulleted list.
6. Do NOT say you cannot find information if the data IS present in the context.
7. Only if the specific data is genuinely absent, respond with exactly: \"I could not find supporting information in the curriculum documents.\"
"""


def build_prompt(question: str, context_chunks: List[Dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\\n{chunk['text']}"
        )
    context_str = "\\n\\n---\\n\\n".join(context_parts)

    return f"""CONTEXT (curriculum data extracted from PDF tables):
{context_str}

---

QUESTION: {question}

Read the context carefully, identify the relevant semester/branch, and answer clearly:"""


def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    if not context_chunks:
        return NO_ANSWER_MSG

    if not MISTRAL_API_KEY:
        return "Error: MISTRAL_API_KEY not set. Add it to your .env file."

    prompt = build_prompt(question, context_chunks)

    try:
        response = requests.post(
            MISTRAL_URL,
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"Error contacting Mistral API: {str(e)}"
