"""
Purpose:

* Loads curriculum PDF files and extracts page-level text content.

Inputs:

* A single PDF file path or a directory containing multiple PDF files.

Outputs:

* A list of document dictionaries with text, source filename, and page number.

Used in:

* Called by the ingestion pipeline before chunking and indexing.
"""

import pdfplumber
from pathlib import Path
from typing import List, Dict

def load_pdf(file_path: str) -> List[Dict]:
    """
    Load one PDF and extract text from each page.

    Parameters:

    * file_path (str): path to the PDF file.

    Returns:

    * List[Dict]: page-level records with text, source, and page.

    Steps:

    1. Check that the file exists.
    2. Detect and block Git LFS pointer files.
    3. Open the PDF with pdfplumber.
    4. Read text page by page.
    5. Keep non-empty pages and return them.
    """
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
    Load all PDF files from one directory.

    Parameters:

    * data_dir (str): directory path that contains PDF files.

    Returns:

    * List[Dict]: combined page-level records from all PDFs.

    Steps:

    1. Find all .pdf files in the given directory.
    2. Raise an error if no PDFs are found.
    3. Load each PDF using load_pdf.
    4. Merge all page records into one list.
    5. Return the combined result.
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
