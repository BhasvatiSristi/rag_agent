"""
retrieval/bm25.py
Lightweight BM25 keyword retrieval with JSON persistence.
"""

import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from config.settings import BM25_INDEX_PATH

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class BM25Index:
    def __init__(self, documents: List[Dict]):
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
        return _TOKEN_RE.findall(text.lower())

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self.term_df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def query(self, question: str, top_k: int = 5, source_file: Optional[str] = None) -> List[Dict]:
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
            norm = k1 * (1 - b + b * (dl / self.avg_doc_len)) if self.avg_doc_len else k1

            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf(term)
                score += idf * ((f * (k1 + 1)) / (f + norm))

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
    path = Path(BM25_INDEX_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_bm25_index(chunks: List[Dict]) -> None:
    path = _index_path()
    serializable_docs = [
        {"text": c["text"], "source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]}
        for c in chunks
    ]
    path.write_text(json.dumps({"documents": serializable_docs}, ensure_ascii=True), encoding="utf-8")
    print(f"  -> Stored BM25 index at {path}")


def _load_documents() -> List[Dict]:
    path = _index_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("documents", [])
    except Exception:
        return []


def query_bm25(question: str, top_k: int = 5, source_file: Optional[str] = None) -> List[Dict]:
    documents = _load_documents()
    if not documents:
        return []
    return BM25Index(documents).query(question, top_k=top_k, source_file=source_file)
