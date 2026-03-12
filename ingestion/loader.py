"""
ingestion/loader.py
Loads PDF files and extracts text + tables.
Tables are converted to structured readable sentences for better embedding.
"""

import pdfplumber
from pathlib import Path
from typing import List, Dict


def _table_to_text(table: List[List[str]], source: str, page_num: int) -> str:
    """
    Convert a pdfplumber table (list of rows) into structured readable text.

    Example output:
        Semester: 3 | Course Code: CS301 | Course Name: Data Structures | Credits: 4
    """
    if not table or len(table) < 2:
        return ""

    # First row is assumed to be the header
    headers = [str(h).strip() if h else "" for h in table[0]]

    rows_text = []
    for row in table[1:]:
        if not any(cell for cell in row):  # skip fully empty rows
            continue
        parts = []
        for header, cell in zip(headers, row):
            cell_val = str(cell).strip() if cell else ""
            if cell_val:
                parts.append(f"{header}: {cell_val}")
        if parts:
            rows_text.append(" | ".join(parts))

    return "\n".join(rows_text)


def load_pdf(file_path: str) -> List[Dict]:
    """
    Load a PDF and return a list of page-level documents.

    Each document is a dict:
    {
        "text": "...",       # full text of the page (prose + tables)
        "source": "file.pdf",
        "page": 1
    }
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    documents = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract text with tables properly formatted
            # pdfplumber.extract_text() handles both prose and tables automatically
            raw_text = page.extract_text()
            
            if raw_text and raw_text.strip():
                documents.append({
                    "text": raw_text.strip(),
                    "source": path.name,
                    "page": page_num,
                })

    print(f"  → Loaded {len(documents)} pages from '{path.name}'")
    return documents


def load_all_pdfs(data_dir: str) -> List[Dict]:
    """
    Load all PDFs from a directory.
    Returns combined list of page documents.
    """
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {data_dir}")

    all_docs = []
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        docs = load_pdf(str(pdf_file))
        all_docs.extend(docs)

    print(f"\n✅ Total pages loaded: {len(all_docs)}")
    return all_docs
