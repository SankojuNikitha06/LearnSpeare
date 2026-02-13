# rag.py
"""
RAG: load FAISS index + chunk metadata, embed queries with Gemini,
retrieve top-k chunks, build context for prompts.
"""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv

# Load .env from project root (same as app.py)
load_dotenv(Path(__file__).resolve().parent / ".env")

# Optional: use google.genai for embeddings (google-genai package)
try:
    from google import genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    genai = None


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


class RAGIndex:
    """
    Loads:
      - FAISS index from index/faiss.index
      - chunk metadata from index/chunks.jsonl

    Retrieval:
      - embeds query using Gemini embeddings model
      - searches FAISS (cosine similarity via normalized vectors + IndexFlatIP)
      - returns top-k chunks + citations
    """

    def __init__(
        self,
        index_dir: str = "index",
        embed_model: str = None,
        api_key: str = None,
    ):
        if not _GENAI_AVAILABLE:
            raise ImportError("google-genai is required for RAG. Install with: pip install google-genai")

        self.index_dir = index_dir
        self.embed_model = embed_model or os.getenv("EMBED_MODEL", "gemini-embedding-001")
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("RAG requires GEMINI_API_KEY or GOOGLE_API_KEY in .env")
        self.client = genai.Client(api_key=api_key)

        faiss_path = os.path.join(index_dir, "faiss.index")
        meta_path = os.path.join(index_dir, "chunks.jsonl")

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self.index = faiss.read_index(faiss_path)
        self.meta = self._load_meta(meta_path)

        if self.index.ntotal != len(self.meta):
            print(
                f"[RAG] Warning: FAISS vectors={self.index.ntotal} but meta rows={len(self.meta)}",
                flush=True,
            )

    def _load_meta(self, meta_path: str) -> List[Dict]:
        out = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    @lru_cache(maxsize=2048)
    def _embed_query_cached(self, q: str) -> Tuple[float, ...]:
        res = self.client.models.embed_content(
            model=self.embed_model,
            contents=q,
        )
        # google-genai: result has .embeddings (list); each has .values or is the vector
        emb = res.embeddings
        if not emb:
            raise RuntimeError("Empty embeddings response")
        first = emb[0]
        if hasattr(first, "values"):
            vec = np.array(first.values, dtype="float32")
        elif isinstance(first, (list, tuple)):
            vec = np.array(first, dtype="float32")
        else:
            vec = np.array(getattr(first, "embedding", first), dtype="float32")
        return tuple(vec.tolist())

    def embed_query(self, q: str) -> np.ndarray:
        v = np.array(self._embed_query_cached(q), dtype="float32")[None, :]
        return _l2_normalize(v)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        v = self.embed_query(query)
        scores, ids = self.index.search(v, k)

        results = []
        for rank, idx in enumerate(ids[0].tolist()):
            if idx < 0:
                continue
            if idx >= len(self.meta):
                continue
            m = self.meta[idx]
            results.append(
                {
                    "rank": rank + 1,
                    "score": float(scores[0][rank]),
                    "source": m.get("source", ""),
                    "title": m.get("title", ""),
                    "text": m.get("text", ""),
                }
            )
        return results

    def build_context(
        self,
        query: str,
        k: int = 5,
        max_chars: int = 6000,
    ) -> Tuple[str, List[Dict]]:
        """
        Returns:
          context_text: formatted snippets for prompt
          citations: list for UI response
        """
        hits = self.retrieve(query, k=k)

        ctx_lines = []
        used = 0
        for h in hits:
            snippet = h["text"].strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200].rstrip() + "…"

            block = f"[{h['rank']}] source={h['source']} | title={h['title']}\n{snippet}\n"
            if used + len(block) > max_chars:
                break
            ctx_lines.append(block)
            used += len(block)

        context_text = "\n".join(ctx_lines).strip()
        return context_text, hits
