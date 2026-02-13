#!/usr/bin/env python3
"""
ingest.py — Build a FAISS index for LearnSphere RAG from scikit-learn HTML docs (Sphinx site),
plus optional .md/.txt/.pdf files.

✅ What it does:
1) Walk docs folder (HTML build)
2) Extract main content from each HTML page
3) Chunk text
4) Embed chunks using Gemini embeddings
5) Store vectors in FAISS (cosine similarity via L2-normalized vectors + inner product)
6) Store chunk metadata (source path + text) in JSONL

Run:
  python ingest.py --docs_root knowledge_base/sklearn_html

Outputs:
  index/faiss.index
  index/chunks.jsonl
  index/manifest.json

Env:
  GEMINI_API_KEY=...
  EMBED_MODEL=gemini-embedding-001  (optional override)
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv, find_dotenv

# Optional deps (HTML + PDF)
from bs4 import BeautifulSoup  # pip install beautifulsoup4 lxml
import fitz  # pip install pymupdf

from google import genai  # pip install google-genai


# ----------------------------
# Config / Defaults
# ----------------------------

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

# Recommended default for scikit-learn docs: keep index small + high-signal
DEFAULT_ALLOWED_DIRS = {"modules", "tutorial", "api", "auto_examples", "datasets"}
DEFAULT_ALLOWED_ROOT_FILES = {
    "supervised_learning.html",
    "unsupervised_learning.html",
    "model_selection.html",
    "data_transforms.html",
    "common_pitfalls.html",
    "getting_started.html",
    "glossary.html",
    "user_guide.html",
    "datasets.html",
}

SKIP_DIRS = {"_static", "_images", "_downloads", "_sources", "binder", "lite", "notebooks"}
SKIP_FILES = {
    "search.html",
    "genindex.html",
    "py-modindex.html",
    "index.html",  # root landing page (often nav-heavy)
    "contents.html",  # mostly nav table of contents
}
SKIP_EXTS = {
    ".js",
    ".css",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".gif",
    ".webp",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".map",
    ".inv",
}

# Chunking defaults (char-based; simple & robust)
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

# Embedding batching
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_RETRIES = 6


# ----------------------------
# Helpers
# ----------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def clean_text(s: str) -> str:
    # Normalize whitespace; keep newlines for readability
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks = []
    i = 0
    step = chunk_size - overlap
    while i < len(text):
        c = text[i : i + chunk_size].strip()
        if c:
            chunks.append(c)
        i += step
    return chunks


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


# ----------------------------
# Readers (HTML / PDF / Text)
# ----------------------------

@dataclass
class DocPage:
    source_path: str
    title: str
    text: str


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)


def read_html(path: str) -> Tuple[str, str]:
    """
    Returns (title, extracted_text)
    Designed for Sphinx/pydata-sphinx-theme docs (like scikit-learn).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        # scikit-learn titles often contain "— scikit-learn ..."
        title = re.sub(r"\s+—\s+scikit-learn.*$", "", title, flags=re.IGNORECASE).strip()

    # Prefer “main content” containers used by Sphinx themes
    main = (
        soup.select_one("div.bd-content") or
        soup.select_one('div[role="main"]') or
        soup.select_one("div.document") or
        soup.select_one("div.body") or
        soup.body
    )
    if not main:
        return title, ""

    # Remove nav/sidebars/scripts/styles and other noisy layout elements
    for sel in [
        "nav", "header", "footer", "script", "style", "aside", "form",
        ".bd-sidebar", ".bd-toc", ".toc", ".sphinxsidebar", ".sidebar",
        ".navbar", ".header-article", ".prev-next-area", ".pagination",
        ".related", ".wy-nav-side", ".wy-side-scroll", ".wy-nav-content"
    ]:
        for tag in main.select(sel):
            tag.decompose()

    text = main.get_text("\n", strip=True)
    text = clean_text(text)
    return title, text


# ----------------------------
# File filtering for sklearn HTML build
# ----------------------------

def should_skip(path: str, docs_root: str,
                allowed_dirs: Optional[set],
                allowed_root_files: Optional[set],
                strict_allowlist: bool) -> bool:
    rel = os.path.relpath(path, docs_root)
    parts = rel.split(os.sep)
    base = os.path.basename(path)
    ext = os.path.splitext(base)[1].lower()

    if ext in SKIP_EXTS:
        return True
    if base in SKIP_FILES:
        return True
    if any(p in SKIP_DIRS for p in parts):
        return True

    if not strict_allowlist:
        return False

    # If strict allowlist: only index key directories + selected root html
    if len(parts) >= 2:
        top = parts[0]
        if allowed_dirs and top not in allowed_dirs:
            return True
    else:
        # root-level files
        if allowed_root_files and base not in allowed_root_files:
            return True

    return False


# ----------------------------
# Gemini embeddings
# ----------------------------

def init_genai_client() -> genai.Client:
    load_dotenv(find_dotenv())
    # google-genai reads GEMINI_API_KEY from env automatically
    return genai.Client()


def embed_texts(client: genai.Client,
                texts: List[str],
                model: str,
                batch_size: int,
                max_retries: int) -> np.ndarray:
    """
    Embed list of texts into float32 numpy array [N, dim]
    Uses batching + retries with exponential backoff.
    """
    vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        attempt = 0
        while True:
            try:
                res = client.models.embed_content(model=model, contents=batch)

                # google-genai response object: res.embeddings -> list of embeddings, each has .values
                embs = getattr(res, "embeddings", None)
                if embs is None and isinstance(res, dict):
                    embs = res.get("embeddings")

                if embs is None:
                    raise RuntimeError("embed_content returned no embeddings field")

                for e in embs:
                    vals = getattr(e, "values", None)
                    if vals is None and isinstance(e, dict):
                        vals = e.get("values")
                    if vals is None:
                        raise RuntimeError("embedding item has no values")
                    vectors.append(vals)

                break  # success
            except Exception as ex:
                attempt += 1
                if attempt > max_retries:
                    raise
                sleep_s = min(30.0, (2 ** (attempt - 1)) + random.random())
                log(f"[embed] retry {attempt}/{max_retries} after error: {ex}  (sleep {sleep_s:.1f}s)")
                time.sleep(sleep_s)

    arr = np.array(vectors, dtype="float32")
    return arr


# ----------------------------
# Main Ingest Pipeline
# ----------------------------

@dataclass
class ChunkMeta:
    source: str           # relative path
    title: str
    chunk_id: int
    text: str


def walk_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            files.append(os.path.join(dirpath, fn))
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_root", required=True,
                    help="Root folder containing scikit-learn HTML docs (the folder with modules/, tutorial/, etc.)")
    ap.add_argument("--out_dir", default="index", help="Output folder for faiss.index and chunks.jsonl")
    ap.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="Gemini embedding model id")
    ap.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    ap.add_argument("--top_k_dirs", default=",".join(sorted(DEFAULT_ALLOWED_DIRS)),
                    help="Comma-separated dirs under docs_root to index (only if strict_allowlist=1)")
    ap.add_argument("--root_files", default=",".join(sorted(DEFAULT_ALLOWED_ROOT_FILES)),
                    help="Comma-separated root-level HTML files to index (only if strict_allowlist=1)")
    ap.add_argument("--strict_allowlist", type=int, default=1,
                    help="1 = only index important dirs/pages; 0 = index all html pages except junk")
    ap.add_argument("--include_pdfs", type=int, default=0, help="Also ingest PDFs if present")
    ap.add_argument("--include_md_txt", type=int, default=0, help="Also ingest .md/.txt if present")
    ap.add_argument("--max_chunks", type=int, default=0,
                    help="Optional cap for quick testing (0 = no cap)")
    args = ap.parse_args()

    docs_root = os.path.abspath(args.docs_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    allowed_dirs = {s.strip() for s in args.top_k_dirs.split(",") if s.strip()}
    allowed_root_files = {s.strip() for s in args.root_files.split(",") if s.strip()}
    strict_allowlist = bool(args.strict_allowlist)

    # 1) Collect candidate files
    files = walk_files(docs_root)
    files.sort()

    # Filter + read
    pages: List[DocPage] = []
    for f in files:
        if should_skip(f, docs_root, allowed_dirs, allowed_root_files, strict_allowlist):
            continue

        ext = os.path.splitext(f)[1].lower()

        if ext in [".html", ".htm"]:
            title, text = read_html(f)
            if not text:
                continue
            pages.append(DocPage(source_path=f, title=title, text=text))

        elif args.include_md_txt and ext in [".md", ".txt"]:
            text = read_text_file(f)
            text = clean_text(text)
            if text:
                pages.append(DocPage(source_path=f, title=os.path.basename(f), text=text))

        elif args.include_pdfs and ext == ".pdf":
            text = read_pdf(f)
            text = clean_text(text)
            if text:
                pages.append(DocPage(source_path=f, title=os.path.basename(f), text=text))

        # else skip

    if not pages:
        log("No pages found after filtering. Check --docs_root and allowlist settings.")
        return 2

    log(f"Found {len(pages)} pages/files to ingest (strict_allowlist={int(strict_allowlist)}).")

    # 2) Chunk + dedupe
    metas: List[ChunkMeta] = []
    chunk_texts: List[str] = []
    seen_hashes = set()
    chunk_id = 0

    for p in pages:
        chunks = chunk_text(p.text, args.chunk_size, args.chunk_overlap)
        rel = os.path.relpath(p.source_path, docs_root)

        for c in chunks:
            h = sha1_text(c)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            metas.append(ChunkMeta(
                source=rel,
                title=p.title,
                chunk_id=chunk_id,
                text=c
            ))
            chunk_texts.append(c)
            chunk_id += 1

            if args.max_chunks and len(chunk_texts) >= args.max_chunks:
                break
        if args.max_chunks and len(chunk_texts) >= args.max_chunks:
            break

    if not chunk_texts:
        log("No chunks produced. Consider lowering filters or increasing chunk_size.")
        return 3

    log(f"Prepared {len(chunk_texts)} unique chunks.")

    # 3) Embed
    client = init_genai_client()
    t0 = time.time()
    vecs = embed_texts(
        client=client,
        texts=chunk_texts,
        model=args.embed_model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
    )
    dt = time.time() - t0
    log(f"Embedded {vecs.shape[0]} chunks in {dt:.1f}s  (dim={vecs.shape[1]}).")

    # 4) Normalize + FAISS index (cosine similarity)
    vecs = l2_normalize(vecs)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # 5) Write outputs
    faiss_path = os.path.join(out_dir, "faiss.index")
    meta_path = os.path.join(out_dir, "chunks.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")

    faiss.write_index(index, faiss_path)

    with open(meta_path, "w", encoding="utf-8") as out:
        for m in metas:
            out.write(json.dumps({
                "id": m.chunk_id,
                "source": m.source,
                "title": m.title,
                "text": m.text
            }, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "docs_root": docs_root,
        "out_dir": out_dir,
        "embed_model": args.embed_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "batch_size": args.batch_size,
        "strict_allowlist": strict_allowlist,
        "allowed_dirs": sorted(list(allowed_dirs)),
        "allowed_root_files": sorted(list(allowed_root_files)),
        "num_pages": len(pages),
        "num_chunks": len(metas),
        "faiss_index": "faiss.index",
        "meta_file": "chunks.jsonl",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log("✅ Done")
    log(f"FAISS:    {faiss_path}")
    log(f"Metadata: {meta_path}")
    log(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
