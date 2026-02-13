# LearnSphere — Project Overview

This document gives a high-level overview of the **LearnSphere** project: architecture, features, and main components.

---

## 1. What Is LearnSphere?

**LearnSphere** is a **multi-modal Machine Learning learning assistant** built with Flask and Google Gemini. It helps users learn ML through:

- **Text explanations** — Structured concept explanations (markdown) tailored to beginner / intermediate / advanced, with optional RAG from scikit-learn docs.
- **Code generation** — Runnable ML code (e.g. sklearn) with dependency detection and file download.
- **Audio lessons** — Narrated ML lessons (script + gTTS MP3) for on-the-go learning.
- **Visual learning aids** — ML concept diagrams (flowcharts, architecture) generated via Gemini image API.

All content is **personalized by learning level** (Beginner / Intermediate / Advanced) and optionally **grounded with RAG** (scikit-learn documentation) for text mode.

---

## 2. Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3, Flask 3.x |
| **AI / LLM** | Google Gemini (google-generativeai, google-genai) |
| **Text-to-speech** | gTTS |
| **RAG** | FAISS (faiss-cpu), Gemini embeddings, scikit-learn HTML docs |
| **Database** | SQLite (history) |
| **Frontend** | Jinja2 templates, vanilla JS, CSS (no framework) |
| **Build / ingest** | ingest.py (BeautifulSoup, PyMuPDF optional) |

Key dependencies: `Flask`, `google-generativeai`, `google-genai`, `gTTS`, `Pillow`, `markdown`, `Pygments`, `numpy`, `faiss-cpu`, `python-dotenv`, `requests`, `Werkzeug`.

---

## 3. Project Structure

```
learnsphere-updated/
├── app.py                 # Flask app: routes, API endpoints, streaming
├── history.py             # SQLite history (add / get / delete / clear)
├── rag.py                 # RAG: FAISS index + Gemini embeddings, build_context()
├── ingest.py              # Build FAISS index from scikit-learn HTML (and optional docs)
├── requirements.txt       # Python dependencies
├── .env.example           # Example env (GEMINI_API_KEY, FLASK_SECRET_KEY, etc.)
├── README.md              # Setup and run instructions
├── PROJECT_OVERVIEW.md    # This document
│
├── index/                 # RAG index (created by ingest.py)
│   ├── faiss.index        # FAISS vector index
│   ├── chunks.jsonl       # Chunk metadata (source, title, text)
│   └── manifest.json      # Ingest manifest
│
├── knowledge_base/        # Source docs for RAG (e.g. scikit-learn HTML)
│   └── sckitlearn/        # HTML + notebooks (optional)
│
├── storage/               # Runtime artifacts (created at run time)
│   ├── history.db         # SQLite history DB
│   ├── generated_code/    # Saved code files for download
│   ├── generated_audio/   # gTTS MP3 files
│   └── generated_images/  # Gemini-generated diagram images
│
├── templates/             # Jinja2 HTML
│   ├── base.html          # Layout, nav, onboarding modal
│   ├── index.html         # Home + feature cards
│   ├── text_explanation.html
│   ├── code_generation.html
│   ├── audio_learning.html
│   ├── image_visualization.html
│   ├── settings.html
│   ├── history.html
│   ├── about.html
│   ├── 404.html
│   └── 500.html
│
├── static/
│   ├── css/style.css      # Global styles (variables, layout, components)
│   ├── images/            # e.g. learnsphere-icon.png
│   └── js/
│       ├── main.js        # Shared (e.g. nav toggle)
│       ├── profile.js     # Profile / API key / level
│       ├── onboarding.js  # First-time setup modal
│       ├── settings.js
│       ├── text_explanation.js
│       ├── code_generation.js
│       ├── audio_learning.js
│       ├── image_visualization.js
│       └── history.js
│
└── utils/
    ├── __init__.py
    ├── genai_utils.py     # Gemini: text, code, audio script, image prompts, image gen, validation
    ├── code_executor.py   # Dependency detection (AST), save code to file
    ├── audio_utils.py     # gTTS text-to-audio, delete old audio files
    └── image_utils.py     # Pillow diagram drawing, image handling
```

---

## 4. Application Flow

### 4.1 Startup

1. **app.py** loads `.env`, initializes Flask, creates `storage/` dirs (`generated_code`, `generated_audio`, `generated_images`).
2. **history.py** is initialized with `storage/history.db`; table and index are created if missing.
3. **RAG** (optional): if `index/faiss.index` and `index/chunks.jsonl` exist, `rag.RAGIndex` is loaded; otherwise RAG is disabled and text mode works without retrieval.

### 4.2 Pages (GET)

- `/` — Home (feature cards).
- `/text-explanation` — Text explanation UI.
- `/code-generation` — Code generation UI.
- `/audio-learning` — Audio lesson UI.
- `/image-visualization` — Visual diagrams UI.
- `/settings` — API key, level, focus, RAG toggle.
- `/history` — List/delete history by type (text, code, audio, images).
- `/about` — About page.

All pages extend `base.html` (navbar, footer, onboarding modal). Profile (level, API key, RAG) is stored in the browser (e.g. localStorage) and sent in API request bodies.

### 4.3 API (POST / GET / DELETE)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/generate-text` | POST | One-shot text explanation (optional RAG). |
| `/api/generate-text-stream` | POST | Stream text explanation (SSE). |
| `/api/render-markdown` | POST | Convert markdown to HTML. |
| `/api/generate-code` | POST | One-shot code generation. |
| `/api/generate-code-stream` | POST | Stream code (SSE), then return code + deps + download URL. |
| `/api/generate-audio` | POST | Generate script + gTTS MP3; return script, audio URL, download URL. |
| `/api/generate-images` | POST | Generate multiple diagram images (batch). |
| `/api/generate-images-stream` | POST | Stream images one-by-one (SSE). |
| `/api/history` | GET | Get user history (optional type filter). |
| `/api/history` | POST | Append one history record (e.g. after image stream). |
| `/api/history/<id>` | DELETE | Delete one history record. |
| `/api/history/clear` | POST | Clear all history for user. |
| `/api/model-info` | GET | Return model info for UI. |

Static/download routes:

- `GET /generated/audio/<path>` — Serve generated MP3.
- `GET /generated/images/<path>` — Serve generated images.
- `GET /download/code/<path>` — Download code file.
- `GET /download/audio/<path>` — Download audio file.

---

## 5. Core Modules

### 5.1 app.py

- **Routes**: Page renders and API handlers above.
- **Profile/level**: Reads `profile.level`, `profile.rag_enabled`, `user_id` from request JSON; builds `level_instructions` for prompts.
- **RAG**: When RAG is enabled and index is loaded, calls `RAG.build_context(topic, k=5, max_chars=6000)` and injects context + citations into text prompts.
- **Streaming**: Text and code use Server-Sent Events (SSE); image stream yields one event per image.
- **History**: After successful generation, calls `history_mod.add_record(user_id, type, topic, payload)` (and for streams, on “done”).
- **Validation**: Text/code/image topics are validated via `genai_utils` (e.g. ML-only); non-ML gets a friendly message or redirect.

### 5.2 utils/genai_utils.py

- **Gemini**: Uses `google.generativeai` (and optionally image/embedding APIs). Configures API key (from argument or `GEMINI_API_KEY`).
- **Text**: `generate_text_explanation`, `generate_text_explanation_stream` — depth (brief/standard/detailed/comprehensive), level instructions, optional RAG context.
- **Code**: `generate_code_example`, `generate_code_example_stream`, `_extract_code` — language (Python, etc.), depth, level.
- **Audio**: `generate_audio_script` — script for gTTS; can return a non-ML marker.
- **Images**: `get_image_generation_prompts`, `generate_images_via_gemini_api`, `generate_one_image_via_gemini_api` — prompts from topic/level, then image generation.
- **Validation**: `validate_text_topic`, `validate_code_topic`, `validate_image_topic` — ensure topic is ML-related.
- **Model info**: `get_model_info()` for Settings/UI.

### 5.3 utils/code_executor.py

- **detect_dependencies(code)** — AST-based (with regex fallback) to find imports and map them to pip package names (e.g. `sklearn` → `scikit-learn`).
- **save_code_to_file(code, topic, out_dir, language)** — Saves to `generated_code/` with a safe filename; returns filename for download URL.

### 5.4 utils/audio_utils.py

- **text_to_audio(text, topic, out_dir)** — Cleans markdown for TTS, uses gTTS to produce MP3, saves under `generated_audio/`, returns filename.
- **delete_old_audio_files(out_dir, max_age_hours)** — Cleanup to avoid disk bloat.

### 5.5 utils/image_utils.py

- **generate_graphical_diagrams(step_lists, topic, out_dir)** — Pillow-based diagram drawing (e.g. flowcharts). Used when Gemini image API is not used or as fallback.
- Image handling and base64/IO for Gemini-generated images.

### 5.6 history.py

- **init_history(db_path)** — Create SQLite DB and `history` table (id, user_id, type, topic, created_at, payload).
- **add_record(user_id, type, topic, payload)** — Append one record (payload = JSON).
- **get_history(user_id, type_filter, limit)** — List records (newest first).
- **delete_record(user_id, record_id)** — Delete one.
- **clear_all(user_id)** — Delete all for user.

Types: `text`, `code`, `audio`, `images`.

### 5.7 rag.py

- **RAGIndex(index_dir, embed_model, api_key)** — Loads `index/faiss.index` and `index/chunks.jsonl`; uses `google.genai` for embeddings (e.g. `gemini-embedding-001`).
- **embed_query(q)** — Embed query string (L2-normalized for cosine via inner product).
- **retrieve(query, k)** — FAISS search; returns list of dicts (rank, score, source, title, text).
- **build_context(query, k, max_chars)** — Retrieves top-k, truncates long snippets, formats context string and citations for the prompt.

### 5.8 ingest.py

- Builds the RAG index from a docs root (e.g. scikit-learn HTML).
- Parses HTML (BeautifulSoup), optionally other formats; chunks text; embeds with Gemini; writes FAISS index + `chunks.jsonl` + `manifest.json`.
- Run: `python ingest.py --docs_root knowledge_base/sklearn_html` (or similar). Requires `GEMINI_API_KEY` and `google-genai`.

---

## 6. Configuration and Environment

- **.env** (copy from `.env.example`):
  - `GEMINI_API_KEY` — Required for Gemini (and RAG embeddings if using RAG).
  - `FLASK_SECRET_KEY` — Session/security.
  - `FLASK_DEBUG` — Debug mode (default True).
  - `PORT` — Server port (default 5000).
  - Optional: `EMBED_MODEL`, `HF_TOKEN` (if extending image pipeline).

- **UI**: API key and profile (level, focus, RAG) can be set in Settings and are stored in the browser; API key can also be set server-side only via `.env`.

---

## 7. Data and Storage

- **History**: SQLite at `storage/history.db`; one row per generation (user_id, type, topic, created_at, payload).
- **Code**: Files in `storage/generated_code/`; served and downloadable via `/download/code/`.
- **Audio**: MP3s in `storage/generated_audio/`; served and downloadable; old files pruned (e.g. 24h).
- **Images**: Generated images in `storage/generated_images/`; served via `/generated/images/`.
- **RAG**: Read-only `index/faiss.index` and `index/chunks.jsonl`; no user data stored in index.

---

## 8. Frontend Summary

- **Templates**: Jinja2; `base.html` provides layout, nav, footer, onboarding modal; each feature has its own template and optional script block.
- **CSS**: Single `static/css/style.css` — variables (e.g. palette), layout, cards, forms, buttons, history, modals, toasts, responsive rules.
- **JS**: Per-page scripts call the API (often with streaming), update DOM, handle profile/user_id and (where used) RAG. Onboarding and profile persistence are in `onboarding.js` and `profile.js`.

---

## 9. Error Handling

- **404 / 500**: Handled by Flask error handlers; render `404.html` and `500.html`.
- **API errors**: Return JSON `{ "ok": false, "error": "..." }` with appropriate HTTP status (e.g. 400, 500).
- **Validation**: Non-ML topics get friendly messages or “message_only” responses instead of generation.

---

## 10. Quick Reference

| Task | Command / Location |
|------|---------------------|
| Run app | `python app.py` → http://127.0.0.1:5000 |
| Install deps | `pip install -r requirements.txt` |
| Configure | Copy `.env.example` to `.env`, set `GEMINI_API_KEY` |
| Build RAG index | `python ingest.py --docs_root knowledge_base/...` |
| History DB | `storage/history.db` |
| Generated files | `storage/generated_code/`, `generated_audio/`, `generated_images/` |

This overview should be enough to navigate the repo, extend features, or onboard new contributors. For setup and run steps, see **README.md**.
