"""
Purpose:

* Provides BM25 keyword-based retrieval with a JSON-backed index.

Inputs:

* Chunk dictionaries during index build and question text during search.

Outputs:

* Ranked chunk matches based on keyword relevance.

Used in:

* Called during ingestion to build index.
* Called during query time for sparse retrieval and hybrid fusion.
"""

import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from config.settings import BM25_INDEX_PATH

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
BM25_K1 = 1.5
BM25_B = 0.75
_INDEX_CACHE: Optional["BM25Index"] = None
_INDEX_CACHE_MTIME: float = -1.0


class BM25Index:
    """
    In-memory BM25 index built from chunk documents.

    Parameters:

    * documents (List[Dict]): chunk documents with text and metadata.

    Returns:

    * BM25Index: ready-to-query sparse retrieval object.

    Steps:

    1. Tokenize each document.
    2. Store term frequencies per document.
    3. Compute document frequencies across corpus.
    4. Keep average document length for BM25 scoring.
    """
    def __init__(self, documents: List[Dict]):
        """
        Build BM25 internal statistics from documents.

        Parameters:

        * documents (List[Dict]): chunk documents with text field.

        Returns:

        * None

        Steps:

        1. Save input documents.
        2. Tokenize each document text.
        3. Compute document lengths and average length.
        4. Build term-frequency and document-frequency maps.
        """
        self.documents = documents
        self.doc_tokens: List[List[str]] = [self._tokenize(d["text"]) for d in documents]
        self.doc_len: List[int] = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0

        self.term_df: Dict[str, int] = defaultdict(int)
        self.term_tf: List[Counter] = []
        for tokens in self.doc_tokens:
            tf = Counter(tokens)
            self.term_tf.append(tf)
            for term in tf:
                self.term_df[term] += 1

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Convert raw text into lowercase alphanumeric tokens.

        Parameters:

        * text (str): raw input text.

        Returns:

        * List[str]: token list used for BM25 calculations.

        Steps:

        1. Lowercase the text.
        2. Extract tokens using regex.
        3. Return token list.
        """
        return _TOKEN_RE.findall(text.lower())

    def _idf(self, term: str) -> float:
        """
        Compute inverse document frequency for a term.

        Parameters:

        * term (str): token to score.

        Returns:

        * float: IDF value, or 0 when term is unseen.

        Steps:

        1. Read total document count.
        2. Read document frequency for the term.
        3. Return smoothed BM25 IDF value.
        """
        n = len(self.documents)
        df = self.term_df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query(self, question: str, top_k: int = 5, source_file: Optional[str] = None) -> List[Dict]:
        """
        Search indexed documents using BM25 keyword scoring.

        Parameters:

        * question (str): user query text.
        * top_k (int): maximum number of results.
        * source_file (Optional[str]): optional source filename filter.

        Returns:

        * List[Dict]: ranked result chunks with score metadata.

        Steps:

        1. Tokenize the question.
        2. Loop through indexed documents.
        3. Skip documents outside source filter.
        4. Compute BM25 score term by term.
        5. Sort by score and return top results.
        """
        if not self.documents:
            return []

        q_terms = self._tokenize(question)
        if not q_terms:
            return []

        k1 = 1.5
        b = 0.75
        scored = []

        for i, doc in enumerate(self.documents):
            if source_file and doc.get("source") != source_file:
                continue

            score = 0.0
            tf = self.term_tf[i]
            dl = self.doc_len[i] if self.doc_len[i] else 1
            norm = BM25_K1 * (1 - BM25_B + BM25_B * (dl / self.avg_doc_len)) if self.avg_doc_len else BM25_K1

            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term)
                score += idf * ((f * (BM25_K1 + 1)) / (f + norm))

            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, doc in scored[:top_k]:
            out.append({
                "text": doc["text"],
                "source": doc.get("source", "unknown"),
                "page": doc.get("page", 0),
                "score": round(float(score), 4),
            })
        return out


def _index_path() -> Path:
    """
    Get the BM25 index path and ensure parent directory exists.

    Parameters:

    * None

    Returns:

    * Path: filesystem path for BM25 index JSON file.

    Steps:

    1. Build path from configuration.
    2. Create parent folders if missing.
    3. Return the path.
    """
    path = Path(BM25_INDEX_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_bm25_index(chunks: List[Dict]) -> None:
    """
    Build and save BM25 index data from chunk list.

    Parameters:

    * chunks (List[Dict]): chunk dictionaries from ingestion.

    Returns:

    * None

    Steps:

    1. Resolve output JSON path.
    2. Keep only serializable chunk fields.
    3. Write documents payload to disk.
    """
    path = _index_path()
    serializable_docs = [
        {"text": c["text"], "source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]}
        for c in chunks
    ]
    path.write_text(json.dumps({"documents": serializable_docs}, ensure_ascii=True), encoding="utf-8")
    print(f"  -> Stored BM25 index at {path}")


def _load_documents() -> List[Dict]:
    """
    Load indexed documents from BM25 JSON file.

    Parameters:

    * None

    Returns:

    * List[Dict]: document list, or empty list on failure.

    Steps:

    1. Resolve BM25 index path.
    2. Return empty list if file does not exist.
    3. Parse JSON payload safely.
    4. Return documents field if available.
    """
    path = _index_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("documents", [])
    except Exception:
        return []


def _get_cached_index() -> Optional[BM25Index]:
    """
    Return cached BM25 index and reload only when file changes.

    Parameters:

    * None

    Returns:

    * Optional[BM25Index]: cached index or None when unavailable.

    Steps:

    1. Check whether index file exists.
    2. Compare file modification time with cached time.
    3. Reuse cache when unchanged.
    4. Reload documents and rebuild cache when changed.
    """
    global _INDEX_CACHE, _INDEX_CACHE_MTIME

    path = _index_path()
    if not path.exists():
        _INDEX_CACHE = None
        _INDEX_CACHE_MTIME = -1.0
        return None

    mtime = path.stat().st_mtime
    if _INDEX_CACHE is not None and _INDEX_CACHE_MTIME == mtime:
        return _INDEX_CACHE

    documents = _load_documents()
    if not documents:
        _INDEX_CACHE = None
        _INDEX_CACHE_MTIME = mtime
        return None

    _INDEX_CACHE = BM25Index(documents)
    _INDEX_CACHE_MTIME = mtime
    return _INDEX_CACHE


def query_bm25(question: str, top_k: int = 5, source_file: Optional[str] = None) -> List[Dict]:
    """
    Query BM25 index and return sparse retrieval results.

    Parameters:

    * question (str): user query text.
    * top_k (int): max number of results.
    * source_file (Optional[str]): optional filename filter.

    Returns:

    * List[Dict]: BM25-ranked chunk results.

    Steps:

    1. Load cached index if available.
    2. Return empty list if index is missing.
    3. Run BM25 query with provided parameters.
    """
    index = _get_cached_index()
    if index is None:
        return []
    return index.query(question, top_k=top_k, source_file=source_file)
