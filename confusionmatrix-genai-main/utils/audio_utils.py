from __future__ import annotations

import re
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from gtts import gTTS

logger = logging.getLogger(__name__)


def _script_for_tts(text: str) -> str:
    """Remove markdown/formatting so TTS doesn't read '*', '_', etc. aloud."""
    if not text:
        return ""
    # Remove asterisks and underscores used for emphasis (*word* or **word** or _word_)
    s = re.sub(r"\*+([^*]*)\*+", r"\1", text)
    s = re.sub(r"_+([^_]*)_+", r"\1", s)
    # Remove any remaining stray * or _ that might be read as "asterisk" / "underscore"
    s = s.replace("*", "").replace("_", " ")
    return s


def text_to_audio(text: str, topic: str, out_dir: str) -> str:
    """Convert text to MP3 using gTTS and return the filename."""
    if not text or not text.strip():
        raise ValueError("Audio script is empty; cannot generate audio.")

    clean_text = _script_for_tts(text)

    safe_topic = "".join(ch for ch in (topic or "topic") if ch.isalnum() or ch in ("-", "_", " ")).strip()
    safe_topic = (safe_topic[:30] or "topic").replace(" ", "_")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"audio_{safe_topic}_{ts}.mp3"

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / filename

    tts = gTTS(text=clean_text, lang="en", slow=False)
    tts.save(str(path))

    logger.info("Generated audio: %s", path)
    return filename


def delete_old_audio_files(out_dir: str, max_age_hours: int = 24) -> int:
    """Delete audio files older than max_age_hours; returns number deleted."""
    p = Path(out_dir)
    if not p.exists():
        return 0

    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    deleted = 0

    for f in p.glob("*.mp3"):
        try:
            mtime = datetime.utcfromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                f.unlink(missing_ok=True)
                deleted += 1
        except Exception as e:
            logger.warning("Failed to delete old audio file %s: %s", f, e)

    return deleted
