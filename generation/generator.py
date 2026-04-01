"""
generation/generator.py
Prompt building and LLM API calls for grounded curriculum answers.
"""

import os
import time
import requests
from typing import List, Dict

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hf").strip().lower()

HF_API_TOKEN = (
    os.getenv("HF_API_TOKEN", "")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    or os.getenv("HUGGINGFACE_API_TOKEN", "")
    or os.getenv("huggingfacehub_api_token", "")
)
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_MAX_RETRIES = int(os.getenv("HF_MAX_RETRIES", "4"))
HF_RETRY_BASE_SECONDS = float(os.getenv("HF_RETRY_BASE_SECONDS", "2"))
HF_FALLBACK_TO_MISTRAL = os.getenv("HF_FALLBACK_TO_MISTRAL", "true").strip().lower() == "true"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MAX_RETRIES = int(os.getenv("MISTRAL_MAX_RETRIES", "4"))
MISTRAL_RETRY_BASE_SECONDS = float(os.getenv("MISTRAL_RETRY_BASE_SECONDS", "2"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "4"))
GROQ_RETRY_BASE_SECONDS = float(os.getenv("GROQ_RETRY_BASE_SECONDS", "2"))

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
7. Only if the specific data is genuinely absent, respond with exactly: "I could not find supporting information in the curriculum documents."
"""


def build_prompt(question: str, context_chunks: List[Dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    return f"""CONTEXT (curriculum data extracted from PDF tables):
{context_str}

---

QUESTION: {question}

Read the context carefully, identify the relevant semester/branch, and answer clearly:"""


def _generate_with_hf(prompt: str) -> str:
    if not HF_API_TOKEN:
        return "Error: HF_API_TOKEN or HUGGINGFACEHUB_API_TOKEN not set. Add it to your .env file."

    hf_url = f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "parameters": {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "return_full_text": False,
        },
    }

    last_error = "unknown error"
    attempts = max(1, HF_MAX_RETRIES)

    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(hf_url, headers=headers, json=payload, timeout=60)

            if response.status_code in (429, 503):
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_s = float(retry_after)
                    except ValueError:
                        wait_s = HF_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                else:
                    wait_s = HF_RETRY_BASE_SECONDS * (2 ** (attempt - 1))

                last_error = f"temporary limit ({response.status_code})"
                if attempt < attempts:
                    time.sleep(wait_s)
                    continue

            if 500 <= response.status_code < 600 and attempt < attempts:
                last_error = f"server error {response.status_code}"
                wait_s = HF_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(wait_s)
                continue

            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and data and isinstance(data[0], dict):
                text = data[0].get("generated_text", "").strip()
                return text if text else NO_ANSWER_MSG

            if isinstance(data, dict) and "generated_text" in data:
                text = str(data.get("generated_text", "")).strip()
                return text if text else NO_ANSWER_MSG

            if isinstance(data, dict) and data.get("error"):
                return f"Error contacting Hugging Face API: {data['error']}"

            return NO_ANSWER_MSG

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < attempts:
                wait_s = HF_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(wait_s)
                continue

    return f"Error contacting Hugging Face API: {last_error}"


def _generate_with_mistral(prompt: str) -> str:
    if not MISTRAL_API_KEY:
        return "Error: MISTRAL_API_KEY not set. Add it to your .env file."

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    last_error = "unknown error"
    attempts = max(1, MISTRAL_MAX_RETRIES)

    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                MISTRAL_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_s = float(retry_after)
                    except ValueError:
                        wait_s = MISTRAL_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                else:
                    wait_s = MISTRAL_RETRY_BASE_SECONDS * (2 ** (attempt - 1))

                last_error = "rate limited (429)"
                if attempt < attempts:
                    time.sleep(wait_s)
                    continue

                return (
                    "Error contacting Mistral API: rate limit exceeded (429). "
                    "Try again in a minute, or lower request frequency."
                )

            if 500 <= response.status_code < 600 and attempt < attempts:
                last_error = f"server error {response.status_code}"
                wait_s = MISTRAL_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(wait_s)
                continue

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < attempts:
                wait_s = MISTRAL_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(wait_s)
                continue

    return f"Error contacting Mistral API: {last_error}"


def _generate_with_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set. Add it to your .env file."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    last_error = "unknown error"
    attempts = max(1, GROQ_MAX_RETRIES)

    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                GROQ_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_s = float(retry_after)
                    except ValueError:
                        wait_s = GROQ_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                else:
                    wait_s = GROQ_RETRY_BASE_SECONDS * (2 ** (attempt - 1))

                last_error = "rate limited (429)"
                if attempt < attempts:
                    time.sleep(wait_s)
                    continue

                return (
                    "Error contacting Groq API: rate limit exceeded (429). "
                    "Try again in a minute."
                )

            if 500 <= response.status_code < 600 and attempt < attempts:
                last_error = f"server error {response.status_code}"
                wait_s = GROQ_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(wait_s)
                continue

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt < attempts:
                wait_s = GROQ_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                time.sleep(wait_s)
                continue

    return f"Error contacting Groq API: {last_error}"


def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    if not context_chunks:
        return NO_ANSWER_MSG

    prompt = build_prompt(question, context_chunks)

    if LLM_PROVIDER == "groq":
        return _generate_with_groq(prompt)

    if LLM_PROVIDER == "mistral":
        return _generate_with_mistral(prompt)

    hf_result = _generate_with_hf(prompt)
    if (
        hf_result.startswith("Error contacting Hugging Face API")
        and HF_FALLBACK_TO_MISTRAL
        and MISTRAL_API_KEY
    ):
        return _generate_with_mistral(prompt)

    return hf_result
