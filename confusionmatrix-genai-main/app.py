from __future__ import annotations

import json
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Generator

from dotenv import load_dotenv

# Load .env from app directory so it works regardless of current working directory
load_dotenv(Path(__file__).resolve().parent / ".env")

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

import markdown as md

from utils.genai_utils import (
    generate_text_explanation,
    generate_text_explanation_stream,
    generate_code_example,
    generate_code_example_stream,
    _extract_code,
    generate_audio_script,
    validate_text_topic,
    validate_code_topic,
    validate_image_topic,
    get_image_generation_prompts,
    generate_images_via_gemini_api,
    generate_one_image_via_gemini_api,
    get_model_info,
)
from utils.code_executor import detect_dependencies, save_code_to_file
from utils.audio_utils import text_to_audio, delete_old_audio_files

import history as history_mod

# RAG: optional FAISS + scikit-learn docs retrieval
RAG = None
try:
    from rag import RAGIndex
    _index_dir = str(Path(__file__).resolve().parent / "index")
    RAG = RAGIndex(index_dir=_index_dir)
    print("[RAG] Loaded index OK", flush=True)
except Exception as e:
    print(f"[RAG] Disabled: {e}", flush=True)
    RAG = None

# -----------------------------
# App setup
# -----------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("learnsphere")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret_change_me")

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
CODE_DIR = STORAGE_DIR / "generated_code"
AUDIO_DIR = STORAGE_DIR / "generated_audio"

# Shown when user requests audio for a non-ML topic (no audio generated)
AUDIO_NON_ML_MESSAGE = (
    "Welcome to LearnSphere! We're dedicated to helping you master machine learning concepts. "
    "While we appreciate your curiosity, our focus is purely on ML education. "
    "Perhaps you'd like to explore topics like 'how neural networks learn' or 'understanding classification metrics'?"
)
IMG_DIR = STORAGE_DIR / "generated_images"

for d in (CODE_DIR, AUDIO_DIR, IMG_DIR):
    d.mkdir(parents=True, exist_ok=True)

HISTORY_DB = str(STORAGE_DIR / "history.db")
history_mod.init_history(HISTORY_DB)

# -----------------------------
# Pages
# -----------------------------
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/text-explanation")
def text_explanation_page():
    return render_template("text_explanation.html")


@app.get("/code-generation")
def code_generation_page():
    return render_template("code_generation.html")


@app.get("/audio-learning")
def audio_learning_page():
    return render_template("audio_learning.html")


@app.get("/image-visualization")
def image_visualization_page():
    return render_template("image_visualization.html")


@app.get("/settings")
def settings_page():
    return render_template("settings.html")


@app.get("/about")
def about_page():
    return render_template("about.html")


@app.get("/history")
def history_page():
    return render_template("history.html")


# -----------------------------
# Helpers: language enforced English; personalize by level
# -----------------------------
def build_level_instructions(level: str) -> str:
    """Return a short instruction block for prompt personalization by user level."""
    lev = (level or "intermediate").strip().lower()
    if lev == "beginner":
        return (
            "Personalize for beginner: use simple words, intuition, and analogies; "
            "avoid formulas unless necessary; include 1 example."
        )
    if lev == "advanced":
        return (
            "Personalize for advanced: be concise but deep; include math/derivation where appropriate; "
            "cover edge cases and pitfalls; include best practices."
        )
    return (
        "Personalize for intermediate: balance intuition and light math; "
        "include key terms; include 1–2 examples."
    )


def _profile_level(data: dict) -> str:
    """Read profile.level from request JSON; default 'intermediate'. Language is always English."""
    profile = data.get("profile") or {}
    level = (profile.get("level") or "intermediate").strip().lower()
    if level not in ("beginner", "intermediate", "advanced"):
        return "intermediate"
    return level


def _user_id(data: dict) -> str:
    """Read user_id from request JSON (for history). Empty string if missing."""
    return (data.get("user_id") or "").strip()


def _is_text_redirect(content_md: str) -> bool:
    """True if this is the short 'ask an ML question' redirect, not a real explanation. Don't save these to history."""
    if not content_md or len(content_md.strip()) > 700:
        return False
    s = content_md.strip().lower()
    redirect_phrases = (
        "try asking about",
        "please ask a question related to machine learning",
        "learnsphere is designed to help",
        "ask a question related to ml",
    )
    return any(p in s for p in redirect_phrases)


# -----------------------------
# API
# -----------------------------
@app.post("/api/generate-text")
def api_generate_text():
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    depth = (data.get("depth") or "standard").strip()
    api_key = (data.get("api_key") or "").strip() or None
    level = _profile_level(data)
    level_instructions = build_level_instructions(level)

    if not topic:
        return jsonify({"ok": False, "error": "Topic is required."}), 400

    try:
        context = ""
        citations = []
        rag_enabled = (data.get("profile") or {}).get("rag_enabled", False)
        if RAG is not None and rag_enabled:
            context, citations = RAG.build_context(topic, k=5, max_chars=6000)

        text_md = generate_text_explanation(api_key, topic, depth, context=context or None, level_instructions=level_instructions)
        text_html = md.markdown(text_md, extensions=["fenced_code", "tables"])
        resp = {
            "ok": True,
            "topic": topic,
            "depth": depth,
            "content_md": text_md,
            "content_html": text_html,
            "citations": citations,
            "rag_used": bool(context),
            "level": level,
        }
        uid = _user_id(data)
        if uid and not _is_text_redirect(text_md):
            history_mod.add_record(uid, "text", topic, {
                "content_md": text_md,
                "content_html": text_html,
                "depth": depth,
                "level": level,
                "rag_used": bool(context),
                "citations": citations,
            })
        return jsonify(resp)
    except Exception as e:
        logger.exception("generate-text failed")
        return jsonify({"ok": False, "error": str(e)}), 500


def _stream_text_events(
    api_key: str | None,
    topic: str,
    depth: str,
    context: str,
    citations: list,
    level: str,
    level_instructions: str,
    user_id: str,
) -> Generator[str, None, None]:
    """Yield SSE events: 'chunk' with text, then 'done' with full content_md and metadata."""
    full_md: List[str] = []
    try:
        for chunk in generate_text_explanation_stream(
            api_key, topic, depth, context=context or None, level_instructions=level_instructions
        ):
            full_md.append(chunk)
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
    except Exception as e:
        logger.exception("generate-text-stream failed")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return
    content_md = "".join(full_md)
    if user_id and not _is_text_redirect(content_md):
        history_mod.add_record(user_id, "text", topic, {
            "content_md": content_md,
            "content_html": md.markdown(content_md, extensions=["fenced_code", "tables"]),
            "depth": depth,
            "level": level,
            "rag_used": bool(context),
            "citations": citations,
        })
    yield f"data: {json.dumps({'type': 'done', 'content_md': content_md, 'content_html': md.markdown(content_md, extensions=['fenced_code', 'tables']), 'citations': citations, 'rag_used': bool(context), 'level': level})}\n\n"


@app.post("/api/generate-text-stream")
def api_generate_text_stream():
    """Stream text explanation as SSE (chunk events, then done with full content)."""
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    depth = (data.get("depth") or "standard").strip()
    api_key = (data.get("api_key") or "").strip() or None
    level = _profile_level(data)
    level_instructions = build_level_instructions(level)
    user_id = _user_id(data)

    if not topic:
        return jsonify({"ok": False, "error": "Topic is required."}), 400

    try:
        status, message = validate_text_topic(api_key, topic)
        if status != "ok":
            msg = (message or "").strip() or "LearnSphere is for ML learning—try \"backpropagation\", \"decision trees\", or \"bias-variance tradeoff\"."
            return Response(
                f"data: {json.dumps({'type': 'message_only', 'message': msg, 'level': level})}\n\n"
                f"data: {json.dumps({'type': 'done'})}\n\n",
                content_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
    except Exception as e:
        logger.exception("validate-text-topic failed")
        return jsonify({"ok": False, "error": str(e)}), 500

    context = ""
    citations = []
    rag_enabled = (data.get("profile") or {}).get("rag_enabled", False)
    if RAG is not None and rag_enabled:
        context, citations = RAG.build_context(topic, k=5, max_chars=6000)

    return Response(
        stream_with_context(_stream_text_events(api_key, topic, depth, context, citations, level, level_instructions, user_id)),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/render-markdown")
def api_render_markdown():
    """Convert markdown to HTML (e.g. for final display after streaming)."""
    data = request.get_json(force=True) or {}
    content_md = data.get("content_md") or ""
    content_html = md.markdown(content_md, extensions=["fenced_code", "tables"])
    return jsonify({"ok": True, "content_html": content_html})


def _stream_code_events(
    api_key: str | None,
    topic: str,
    depth: str,
    language: str,
    level_instructions: str,
    level: str,
    user_id: str,
) -> Generator[str, None, None]:
    """Yield SSE: 'chunk' with text, then 'done' with code, is_message, deps, download_url, run_instructions, level."""
    full_raw: List[str] = []
    try:
        for chunk in generate_code_example_stream(
            api_key, topic, depth, language=language, level_instructions=level_instructions
        ):
            full_raw.append(chunk)
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
    except Exception as e:
        logger.exception("generate-code-stream failed")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return
    raw_text = "".join(full_raw)
    code = _extract_code(raw_text)
    is_message = not _looks_like_code(code)
    deps = []
    filename = None
    download_url = None
    run_instructions = ""
    if is_message:
        run_instructions = ""
    else:
        if language.strip().lower() == "python":
            deps = detect_dependencies(code)
        filename = save_code_to_file(code, topic, str(CODE_DIR), language=language)
        download_url = f"/download/code/{filename}"
        run_instructions = "\n".join(_run_instructions_for_language(language, filename, deps))
    if user_id and not is_message:
        history_mod.add_record(user_id, "code", topic, {
            "code": code,
            "language": language,
            "depth": depth,
            "filename": filename or "",
            "download_url": download_url or "",
            "level": level,
        })
    yield f"data: {json.dumps({'type': 'done', 'code': code, 'is_message': is_message, 'dependencies': deps, 'filename': filename or '', 'download_url': download_url or '', 'run_instructions': run_instructions, 'level': level, 'language': language})}\n\n"


def _run_instructions_for_language(language: str, filename: str, deps: list) -> list:
    lang = (language or "python").strip().lower()
    if lang == "java":
        return [
            "Compile and run (Java):",
            "  1) javac " + filename,
            "  2) java " + filename.replace(".java", ""),
        ]
    if lang == "javascript":
        return [
            "Run (Node.js): node " + filename,
            "Or open in browser and run in console.",
        ]
    if lang == "cpp":
        return [
            "Compile and run (C++):",
            "  1) g++ -o out " + filename,
            "  2) ./out",
        ]
    # Python
    dep_str = " ".join(deps) if deps else "<no extra deps detected>"
    return [
        "Option A: Run locally",
        "  1) python -m venv .venv && source .venv/bin/activate",
        "  2) pip install " + dep_str,
        "  3) python " + filename,
        "",
        "Option B: Google Colab",
        "  1) Upload the file to Colab",
        "  2) Install deps in a cell if needed: !pip install <deps>",
        "  3) Run the script",
    ]


def _looks_like_code(text: str) -> bool:
    """Heuristic: response is actual code, not a joke/message."""
    if not text or len(text.strip()) < 20:
        return False
    t = text.strip().lower()
    code_marks = ("def ", "class ", "import ", "function ", "void ", "public ", "private ", "{", "=>", "<?php", "#include")
    return any(m in t for m in code_marks)


@app.post("/api/generate-code")
def api_generate_code():
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    depth = (data.get("depth") or "detailed").strip()
    language = (data.get("language") or "python").strip()
    api_key = (data.get("api_key") or "").strip() or None
    level = _profile_level(data)
    level_instructions = build_level_instructions(level)

    if not topic:
        return jsonify({"ok": False, "error": "Topic is required."}), 400

    try:
        code = generate_code_example(api_key, topic, depth, language=language, level_instructions=level_instructions)
        is_message = not _looks_like_code(code)
        deps = []
        filename = None
        download_url = None
        run_instructions = ""

        if is_message:
            run_instructions = ""
        else:
            if language.strip().lower() == "python":
                deps = detect_dependencies(code)
            filename = save_code_to_file(code, topic, str(CODE_DIR), language=language)
            download_url = f"/download/code/{filename}"
            run_instructions = "\n".join(_run_instructions_for_language(language, filename, deps))

        resp = {
            "ok": True,
            "topic": topic,
            "depth": depth,
            "language": language,
            "code": code,
            "is_message": is_message,
            "dependencies": deps,
            "filename": filename or "",
            "download_url": download_url or "",
            "run_instructions": run_instructions,
            "level": level,
        }
        uid = _user_id(data)
        if uid and not is_message:
            history_mod.add_record(uid, "code", topic, {
                "code": code,
                "language": language,
                "depth": depth,
                "filename": filename or "",
                "download_url": download_url or "",
                "level": level,
            })
        return jsonify(resp)
    except Exception as e:
        logger.exception("generate-code failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/generate-code-stream")
def api_generate_code_stream():
    """Stream code generation as SSE (chunk events, then done with full code and metadata)."""
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    depth = (data.get("depth") or "detailed").strip()
    language = (data.get("language") or "python").strip()
    api_key = (data.get("api_key") or "").strip() or None
    level = _profile_level(data)
    level_instructions = build_level_instructions(level)
    user_id = _user_id(data)

    if not topic:
        return jsonify({"ok": False, "error": "Topic is required."}), 400

    try:
        status, message = validate_code_topic(api_key, topic)
        if status != "ok":
            msg = (message or "").strip() or "LearnSphere is for ML code—try \"logistic regression in Python\", \"decision tree with sklearn\", or \"k-means clustering\"."
            return Response(
                f"data: {json.dumps({'type': 'message_only', 'message': msg, 'level': level})}\n\n"
                f"data: {json.dumps({'type': 'done'})}\n\n",
                content_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
    except Exception as e:
        logger.exception("validate-code-topic failed")
        return jsonify({"ok": False, "error": str(e)}), 500

    return Response(
        stream_with_context(_stream_code_events(api_key, topic, depth, language, level_instructions, level, user_id)),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate-audio")
def api_generate_audio():
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    length = (data.get("length") or "brief").strip()
    api_key = (data.get("api_key") or "").strip() or None
    level = _profile_level(data)
    level_instructions = build_level_instructions(level)

    if not topic:
        return jsonify({"ok": False, "error": "Topic is required."}), 400

    try:
        status, _ = validate_image_topic(api_key, topic)
        if status != "ok":
            return jsonify({
                "ok": True,
                "message_only": True,
                "message": AUDIO_NON_ML_MESSAGE,
                "level": level,
            })

        script = generate_audio_script(api_key, topic, length, level_instructions=level_instructions)
        if script.strip().upper().startswith("NON_ML_TOPIC"):
            return jsonify({
                "ok": True,
                "message_only": True,
                "message": AUDIO_NON_ML_MESSAGE,
                "level": level,
            })

        # keep storage tidy
        delete_old_audio_files(str(AUDIO_DIR), max_age_hours=24)

        filename = text_to_audio(script, topic, str(AUDIO_DIR))
        audio_url = f"/generated/audio/{filename}"
        download_url = f"/download/audio/{filename}"

        resp = {
            "ok": True,
            "topic": topic,
            "length": length,
            "script": script,
            "audio_url": audio_url,
            "download_url": download_url,
            "level": level,
        }
        uid = _user_id(data)
        if uid:
            history_mod.add_record(uid, "audio", topic, {
                "script": script,
                "audio_url": audio_url,
                "download_url": download_url,
                "length": length,
                "level": level,
            })
        return jsonify(resp)
    except Exception as e:
        logger.exception("generate-audio failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/generate-images")
def api_generate_images():
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    api_key = (data.get("api_key") or "").strip() or None
    level = _profile_level(data)
    level_instructions = build_level_instructions(level)

    if not topic:
        return jsonify({"ok": False, "error": "Topic is required."}), 400

    try:
        status, message = validate_image_topic(api_key, topic)
        if status != "ok":
            return jsonify({
                "ok": True,
                "message_only": True,
                "message": message or "",
                "prompts": [],
                "image_urls": [],
                "level": level,
            })

        prompts = get_image_generation_prompts(api_key, topic, level_instructions=level_instructions)
        topic_slug = re.sub(r"[^\w\-]", "_", topic)[:40] or "diagram"
        filenames = generate_images_via_gemini_api(
            api_key, prompts, str(IMG_DIR), topic_slug=topic_slug
        )
        image_urls = [f"/generated/images/{fn}" for fn in filenames]
        resp = {
            "ok": True,
            "topic": topic,
            "prompts": prompts,
            "image_urls": image_urls,
            "level": level,
        }
        uid = _user_id(data)
        if uid:
            history_mod.add_record(uid, "images", topic, {
                "image_urls": image_urls,
                "prompts": prompts,
                "level": level,
            })
        return jsonify(resp)
    except Exception as e:
        logger.exception("generate-images failed")
        return jsonify({"ok": False, "error": str(e)}), 500


def _stream_image_events(
    api_key: str | None,
    topic_slug: str,
    prompts: List[str],
    level: str,
) -> Generator[str, None, None]:
    """Yield SSE events: one 'image' per generated image, then 'done'."""
    for i, prompt in enumerate(prompts):
        try:
            filename = generate_one_image_via_gemini_api(
                api_key, prompt, str(IMG_DIR), topic_slug=topic_slug, index=i
            )
            url = f"/generated/images/{filename}"
            yield f"data: {json.dumps({'type': 'image', 'url': url, 'prompt': prompt, 'index': i + 1})}\n\n"
        except Exception as e:
            logger.exception("stream image %s failed", i + 1)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'index': i + 1})}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'level': level})}\n\n"


@app.post("/api/generate-images-stream")
def api_generate_images_stream():
    """Stream images one-by-one as they're ready (SSE). Reduces perceived wait."""
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    api_key = (data.get("api_key") or "").strip() or None
    level = _profile_level(data)
    level_instructions = build_level_instructions(level)

    if not topic:
        return jsonify({"ok": False, "error": "Topic is required."}), 400

    try:
        status, message = validate_image_topic(api_key, topic)
        if status != "ok":
            msg = (message or "").strip() or "This isn’t an ML concept I can draw. Try \"decision trees\", \"gradient descent\", or \"neural network layers\"."
            return Response(
                f"data: {json.dumps({'type': 'message_only', 'message': msg, 'level': level})}\n\n"
                f"data: {json.dumps({'type': 'done'})}\n\n",
                content_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        prompts = get_image_generation_prompts(api_key, topic, level_instructions=level_instructions)
        topic_slug = re.sub(r"[^\w\-]", "_", topic)[:40] or "diagram"

        return Response(
            stream_with_context(_stream_image_events(api_key, topic_slug, prompts, level)),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        logger.exception("generate-images-stream failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/history")
def api_get_history():
    """Return user's history (text, code, audio, images). Query: user_id, type (optional)."""
    user_id = (request.args.get("user_id") or "").strip()
    type_filter = (request.args.get("type") or "").strip() or None
    if not user_id:
        return jsonify({"ok": False, "error": "user_id required"}), 400
    try:
        items = history_mod.get_history(user_id, type_filter=type_filter, limit=100)
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        logger.exception("get-history failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/history")
def api_post_history():
    """Append one history record (e.g. after image stream). Body: user_id, type, topic, payload."""
    data = request.get_json(force=True) or {}
    user_id = _user_id(data)
    type_ = (data.get("type") or "").strip()
    topic = (data.get("topic") or "").strip()
    payload = data.get("payload")
    if not user_id or not type_ or not topic:
        return jsonify({"ok": False, "error": "user_id, type, and topic required"}), 400
    if type_ not in ("text", "code", "audio", "images"):
        return jsonify({"ok": False, "error": "type must be text, code, audio, or images"}), 400
    if not isinstance(payload, dict):
        payload = {}
    try:
        history_mod.add_record(user_id, type_, topic, payload)
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("post-history failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.delete("/api/history/<int:record_id>")
def api_delete_history_item(record_id: int):
    """Delete one history record for the current user. Query: user_id."""
    user_id = (request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"ok": False, "error": "user_id required"}), 400
    try:
        deleted = history_mod.delete_record(user_id, record_id)
        return jsonify({"ok": True, "deleted": deleted})
    except Exception as e:
        logger.exception("delete-history-item failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/history/clear")
def api_clear_history():
    """Delete all history for the current user. Body: user_id."""
    data = request.get_json(force=True) or {}
    user_id = (data.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"ok": False, "error": "user_id required"}), 400
    try:
        count = history_mod.clear_all(user_id)
        return jsonify({"ok": True, "deleted": count})
    except Exception as e:
        logger.exception("clear-history failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/model-info")
def api_model_info():
    try:
        return jsonify({"ok": True, "models": get_model_info()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# -----------------------------
# Static serving for generated artifacts
# -----------------------------
@app.get("/generated/audio/<path:filename>")
def serve_generated_audio(filename: str):
    return send_from_directory(AUDIO_DIR, filename)


@app.get("/generated/images/<path:filename>")
def serve_generated_images(filename: str):
    return send_from_directory(IMG_DIR, filename)


@app.get("/download/code/<path:filename>")
def download_code(filename: str):
    return send_from_directory(CODE_DIR, filename, as_attachment=True)


@app.get("/download/audio/<path:filename>")
def download_audio(filename: str):
    return send_from_directory(AUDIO_DIR, filename, as_attachment=True)


# -----------------------------
# Errors
# -----------------------------
@app.errorhandler(404)
def not_found(_):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(_):
    return render_template("500.html"), 500


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug)
