"""
ingestion/loader.py
Loads PDF files and extracts text + tables.
Tables are converted to structured readable sentences for better embedding.
"""

import pdfplumber
from pathlib import Path
from typing import List, Dict

def load_pdf(file_path: str) -> List[Dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Guard against Git LFS pointer files being mistaken for PDFs.
    if path.stat().st_size < 1024:
        head = path.read_text(encoding="utf-8", errors="ignore")
        if "git-lfs.github.com/spec/v1" in head:
            raise ValueError(
                f"{path.name} is a Git LFS pointer, not a real PDF. Run 'git lfs pull' to fetch binary files."
            )

    documents = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
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
