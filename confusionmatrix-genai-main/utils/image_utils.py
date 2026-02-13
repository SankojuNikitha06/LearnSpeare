from __future__ import annotations

import base64
import io
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def _safe_topic(topic: str) -> str:
    safe = "".join(ch for ch in (topic or "topic") if ch.isalnum() or ch in ("-", "_", " ")).strip()
    return (safe[:30] or "topic").replace(" ", "_")


def _get_fonts() -> Tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    try:
        return (
            ImageFont.truetype("DejaVuSans.ttf", 22),
            ImageFont.truetype("DejaVuSans.ttf", 16),
        )
    except Exception:
        default = ImageFont.load_default()
        return default, default


def _draw_rounded_rect(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], radius: int, fill: Tuple[int, int, int], outline: Tuple[int, int, int], width: int = 2) -> None:
    """Draw a rounded rectangle (Pillow 8+ has draw.rounded_rectangle)."""
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)


def generate_graphical_diagrams(step_lists: List[List[str]], topic: str, out_dir: str) -> List[str]:
    """Draw diagram-style visuals: rounded nodes, soft colors, graph-like layout (no plain boxes)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe = _safe_topic(topic)
    font, font_small = _get_fonts()

    # Softer palette
    bg = (240, 244, 248)
    node_fill = (255, 255, 255)
    node_outline = (99, 102, 241)  # accent
    text_color = (30, 41, 59)
    arrow_color = (99, 102, 241)
    title_color = (71, 85, 105)

    filenames = []
    for i, steps in enumerate(step_lists[:6], start=1):
        img = Image.new("RGB", (1024, 576), bg)
        draw = ImageDraw.Draw(img)

        draw.text((30, 20), f"LearnSphere · Diagram {i}", fill=text_color, font=font)
        draw.text((30, 48), topic[:60] + ("..." if len(topic) > 60 else ""), fill=title_color, font=font_small)

        img_w, img_h = 1024, 576
        title_bottom = 88
        bottom_pad = 24
        available_h = img_h - title_bottom - bottom_pad
        n_steps = len(steps)
        slot_h = available_h / n_steps if n_steps else 0
        box_h = max(40, min(60, int(slot_h * 0.65)))
        gap = max(6, int(slot_h - box_h) // 2)
        arrow_h = gap
        box_w = 540
        cx = img_w // 2
        y = title_bottom
        radius = 12

        for j, label in enumerate(steps):
            x1 = cx - box_w // 2
            x2 = cx + box_w // 2
            y1, y2 = y, y + box_h
            _draw_rounded_rect(draw, (x1, y1, x2, y2), radius, node_fill, node_outline, width=2)
            lines = _wrap_text(label, 40)[:2]
            line_h = 20
            start_y = y + (box_h - len(lines) * line_h) // 2
            for k, ln in enumerate(lines):
                txt = ln[:50]
                tx = max(x1 + 14, cx - len(txt) * 4)
                draw.text((tx, start_y + k * line_h), txt, fill=text_color, font=font_small)
            y = y2 + gap
            if j < len(steps) - 1:
                ax = cx
                ay1, ay2 = y, y + arrow_h
                draw.line([(ax, ay1), (ax, ay2)], fill=arrow_color, width=2)
                ah = 10
                draw.polygon([(ax, ay2), (ax - ah, ay2 - ah), (ax + ah, ay2 - ah)], fill=arrow_color)
                y = ay2

        filename = f"img_{safe}_{ts}_{i}.png"
        path = Path(out_dir) / filename
        img.save(path, format="PNG")
        filenames.append(filename)

    logger.info("Generated %d graphical diagram images", len(filenames))
    return filenames


def generate_flowchart_diagrams(step_lists: List[List[str]], topic: str, out_dir: str, style: str = "flowchart") -> List[str]:
    """Draw one flowchart per list of steps. style='graphical' uses rounded nodes and softer colors."""
    if (style or "flowchart").strip().lower() == "graphical":
        return generate_graphical_diagrams(step_lists, topic, out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe = _safe_topic(topic)
    font, font_small = _get_fonts()

    filenames = []
    for i, steps in enumerate(step_lists[:6], start=1):
        img = Image.new("RGB", (1024, 576), (248, 250, 252))
        draw = ImageDraw.Draw(img)

        # Title
        draw.text((30, 20), f"LearnSphere · Diagram {i}", fill=(30, 41, 59), font=font)
        draw.text((30, 48), topic[:60] + ("..." if len(topic) > 60 else ""), fill=(71, 85, 105), font=font_small)

        # Vertical flowchart: fit all steps in canvas (no cropping)
        img_h = 576
        img_w = 1024
        title_bottom = 88
        bottom_pad = 24
        available_h = img_h - title_bottom - bottom_pad
        n_steps = len(steps)
        slot_h = available_h / n_steps if n_steps else 0
        box_h = max(36, min(56, int(slot_h * 0.68)))
        arrow_h = max(10, min(20, int((slot_h - box_h) * 0.5)))
        gap = max(4, int(slot_h - box_h - arrow_h))

        box_w = 560
        cx = img_w // 2
        y = title_bottom

        for j, label in enumerate(steps):
            # Box
            x1 = cx - box_w // 2
            x2 = cx + box_w // 2
            y1, y2 = y, y + box_h
            draw.rectangle([x1, y1, x2, y2], outline=(30, 41, 59), width=2, fill=(255, 255, 255))
            # Step text (wrap to one line if possible, else two)
            lines = _wrap_text(label, 42)[:2]
            line_h = 20
            start_y = y + (box_h - len(lines) * line_h) // 2
            for k, ln in enumerate(lines):
                txt = ln[:52]
                # Approximate center: 8px per character
                tx = max(x1 + 12, cx - len(txt) * 4)
                draw.text((tx, start_y + k * line_h), txt, fill=(30, 41, 59), font=font_small)
            y = y2 + gap
            # Arrow to next
            if j < len(steps) - 1:
                ax = cx
                ay1, ay2 = y, y + arrow_h
                draw.line([(ax, ay1), (ax, ay2)], fill=(30, 41, 59), width=2)
                ah = 10
                draw.polygon([(ax, ay2), (ax - ah, ay2 - ah), (ax + ah, ay2 - ah)], fill=(30, 41, 59))
                y = ay2

        filename = f"img_{safe}_{ts}_{i}.png"
        path = Path(out_dir) / filename
        img.save(path, format="PNG")
        filenames.append(filename)

    logger.info("Generated %d flowchart images", len(filenames))
    return filenames


def generate_placeholder_diagrams(prompts: List[str], topic: str, out_dir: str) -> List[str]:
    """Create simple placeholder 'diagram' PNGs (legacy). Prefer generate_flowchart_diagrams for topic-aware flowcharts."""
    step_lists = [[p] if isinstance(p, str) else p for p in prompts]
    # If prompts look like "Step1 → Step2" (from new flow), split back to steps
    for i, p in enumerate(step_lists):
        if isinstance(p, list) and len(p) == 1 and " → " in p[0]:
            step_lists[i] = [s.strip() for s in p[0].split(" → ") if s.strip()]
    return generate_flowchart_diagrams(step_lists, topic, out_dir)


def _wrap_text(text: str, width: int) -> List[str]:
    words = (text or "").split()
    lines = []
    cur = []
    cur_len = 0
    for w in words:
        extra = len(w) + (1 if cur else 0)
        if cur_len + extra > width:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += extra
    if cur:
        lines.append(" ".join(cur))
    return lines
