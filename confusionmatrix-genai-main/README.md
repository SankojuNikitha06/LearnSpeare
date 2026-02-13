# LearnSphere (Flask + Gemini) — Generative AI ML Learning System

LearnSphere is a multi‑modal ML learning assistant:
- Text explanations (Beginner/Intermediate/Advanced/Comprehensive)
- Code generation (Python examples + dependency detection + download)
- Audio lessons (script + gTTS MP3)
- Visual learning aids (diagram prompts + placeholder diagram images)

> This implementation follows the reference architecture / modalities described in the provided project reference PDF.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell

pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`.

## 2) Run

```bash
python app.py
```

Open http://127.0.0.1:5000

## Notes
- The UI stores your Gemini API key in browser `localStorage` by default and sends it in request bodies.
  You can also set `GEMINI_API_KEY` in `.env` to keep it server-side only.
- Audio generation uses gTTS which requires internet access.
