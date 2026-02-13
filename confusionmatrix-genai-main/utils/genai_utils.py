from __future__ import annotations

import base64
import os
import re
import time
import logging
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import requests
import google.generativeai as genai

logger = logging.getLogger(__name__)


# A conservative default for educational content.
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


DEPTH_TO_STYLE = {
    "brief": "Brief, high-level explanation (5-10 bullet points).",
    "standard": "Clear explanation with a few examples and intuition.",
    "detailed": "Detailed explanation with equations (when needed) and worked examples.",
    "comprehensive": "Comprehensive: learning objectives, intuition, step-by-step, pitfalls, and applications.",
}


DEFAULT_TEXT_MODEL = "gemini-2.5-flash"
DEFAULT_CODE_MODEL = "gemini-2.5-flash"
DEFAULT_IMAGE_PROMPT_MODEL = "gemini-2.5-flash"
DEFAULT_AUDIO_SCRIPT_MODEL = "gemini-2.5-flash"
GEMINI_IMAGE_GEN_MODEL = "gemini-2.5-flash-image"


@dataclass
class GenAIResult:
    text_md: str = ""
    code_py: str = ""
    audio_script: str = ""
    image_prompts: List[str] = None


def _get_api_key(api_key: Optional[str]) -> str:
    """Return the API key to use (client or .env). Raises if missing."""
    key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise ValueError("Missing Gemini API key. Set it in Settings (browser) or in .env as GEMINI_API_KEY.")
    return key


def _configure(api_key: Optional[str]) -> None:
    genai.configure(api_key=_get_api_key(api_key))


def _call_model(model_name: str, prompt: str, *,
                temperature: float = 0.6,
                top_p: float = 0.9,
                max_output_tokens: int = 2048,
                retries: int = 3) -> str:
    """Call Gemini with simple retry logic."""
    model = genai.GenerativeModel(model_name)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = model.generate_content(
                prompt,
                safety_settings=SAFETY_SETTINGS,
                generation_config={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_output_tokens,
                },
            )
            # google-generativeai returns .text sometimes None if blocked.
            text = (resp.text or "").strip()
            if not text:
                raise RuntimeError("Empty response from model (may be blocked by safety filters).")
            return text
        except Exception as e:
            last_err = e
            sleep_s = 1.5 * (2 ** attempt)
            logger.warning("Gemini call failed (attempt %d/%d): %s; retrying in %.1fs",
                           attempt + 1, retries, e, sleep_s)
            time.sleep(sleep_s)
    raise RuntimeError(f"Gemini call failed after {retries} attempts: {last_err}")


_CODE_FENCE_RE = re.compile(r"```\w*\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_OPEN_RE = re.compile(r"```\w*\s*(.*)", re.DOTALL | re.IGNORECASE)


def _extract_code(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Truncated response (no closing ```): take everything after opening ```
    m_open = _CODE_OPEN_RE.search(text)
    if m_open:
        return m_open.group(1).strip()
    return text.strip()


def generate_text_explanation(
    api_key: Optional[str],
    topic: str,
    depth: str,
    context: Optional[str] = None,
    level_instructions: Optional[str] = None,
) -> str:
    _configure(api_key)
    depth_key = (depth or "standard").strip().lower()
    style = DEPTH_TO_STYLE.get(depth_key, DEPTH_TO_STYLE["standard"])

    rag_instructions = ""
    if context and context.strip():
        rag_instructions = (
            "Use the CONTEXT below (scikit-learn / ML docs snippets) to ground your answer. "
            "When you use information from the context, cite it with the number in brackets, e.g. [1], [2]. "
            "If something is not in the context, you can still explain from your knowledge but do not invent doc details. "
            "CONTEXT:\n" + context.strip() + "\n\n"
        )
    level_block = f"- Level: {level_instructions}\n" if level_instructions else ""

    prompt = f"""You are LearnSphere, a clear and thoughtful tutor for machine learning (ML) education. This platform is focused on helping users learn ML concepts. Language: English only.
{rag_instructions}
{level_block}
Topic or question: {topic}

Important:
- Do NOT start with a greeting or introduction (no "Hello!", "I'm LearnSphere", "Let's dive in", etc.). Start directly with the first heading or the first paragraph of the explanation.
- If the topic is an ML concept (algorithms, models, training, data, neural networks, decision trees, gradient descent, bias-variance, etc.), explain it in a structured way with {style}. Use clean headings, 2-3 examples or analogies, and end with: (1) common pitfalls or takeaways, (2) a short quiz (3 questions), (3) related ML topics to explore. Output Markdown.
- If the topic is NOT about ML (e.g. general or philosophical questions like "who am I", "what is happiness", or other non-ML subjects), do NOT give a long explanation. Instead, in 2-4 short sentences: politely say that LearnSphere is designed for machine learning education, and suggest they try an ML topic. Give one example line, e.g. "Try asking about 'how decision trees work', 'bias-variance tradeoff', or 'backpropagation in neural networks'." Output only that short redirect in Markdown; no quiz or long content.

Output format: Markdown. Start immediately with content, no preamble."""

    return _call_model(os.getenv("GEMINI_TEXT_MODEL", DEFAULT_TEXT_MODEL), prompt,
                       temperature=0.5, max_output_tokens=8192)


def generate_text_explanation_stream(
    api_key: Optional[str],
    topic: str,
    depth: str,
    context: Optional[str] = None,
    level_instructions: Optional[str] = None,
) -> Generator[str, None, None]:
    """Yield text chunks as the model generates. Same prompt as generate_text_explanation."""
    _configure(api_key)
    depth_key = (depth or "standard").strip().lower()
    style = DEPTH_TO_STYLE.get(depth_key, DEPTH_TO_STYLE["standard"])
    rag_instructions = ""
    if context and context.strip():
        rag_instructions = (
            "Use the CONTEXT below (scikit-learn / ML docs snippets) to ground your answer. "
            "When you use information from the context, cite it with the number in brackets, e.g. [1], [2]. "
            "If something is not in the context, you can still explain from your knowledge but do not invent doc details. "
            "CONTEXT:\n" + context.strip() + "\n\n"
        )
    level_block = f"- Level: {level_instructions}\n" if level_instructions else ""
    prompt = f"""You are LearnSphere, a clear and thoughtful tutor for machine learning (ML) education. This platform is focused on helping users learn ML concepts. Language: English only.
{rag_instructions}
{level_block}
Topic or question: {topic}

Important:
- Do NOT start with a greeting or introduction (no "Hello!", "I'm LearnSphere", "Let's dive in", etc.). Start directly with the first heading or the first paragraph of the explanation.
- If the topic is an ML concept (algorithms, models, training, data, neural networks, decision trees, gradient descent, bias-variance, etc.), explain it in a structured way with {style}. Use clean headings, 2-3 examples or analogies, and end with: (1) common pitfalls or takeaways, (2) a short quiz (3 questions), (3) related ML topics to explore. Output Markdown.
- If the topic is NOT about ML (e.g. general or philosophical questions like "who am I", "what is happiness", or other non-ML subjects), do NOT give a long explanation. Instead, in 2-4 short sentences: politely say that LearnSphere is designed for machine learning education, and suggest they try an ML topic. Give one example line, e.g. "Try asking about 'how decision trees work', 'bias-variance tradeoff', or 'backpropagation in neural networks'." Output only that short redirect in Markdown; no quiz or long content.

Output format: Markdown. Start immediately with content, no preamble."""
    model = genai.GenerativeModel(os.getenv("GEMINI_TEXT_MODEL", DEFAULT_TEXT_MODEL))
    resp = model.generate_content(
        prompt,
        safety_settings=SAFETY_SETTINGS,
        generation_config={
            "temperature": 0.5,
            "top_p": 0.9,
            "max_output_tokens": 8192,
        },
        stream=True,
    )
    for chunk in resp:
        if chunk.text:
            yield chunk.text


CODE_LANGUAGES = {
    "python": ("Python", "py"),
    "java": ("Java", "java"),
    "javascript": ("JavaScript", "js"),
    "cpp": ("C++", "cpp"),
}


def generate_code_example(
    api_key: Optional[str],
    topic: str,
    depth: str,
    language: str = "python",
    level_instructions: Optional[str] = None,
) -> str:
    _configure(api_key)
    depth_key = (depth or "detailed").strip().lower()
    lang_key = (language or "python").strip().lower()
    lang_name = CODE_LANGUAGES.get(lang_key, ("Python", "py"))[0]

    # Two levels: minimal or detailed (standard -> minimal for backward compat)
    if depth_key in ("minimal", "standard"):
        depth_instruction = "Keep it minimal: clear working code and a short demo. No long comments or docstrings."
    else:
        depth_instruction = "Add inline comments on key steps and a small runnable demo. One short docstring/comment at the top is enough."

    level_block = f"Level: {level_instructions}. " if level_instructions else ""

    if lang_key == "python":
        rules = """- Output ONLY one code block (```python ... ```). No other prose.
- Must run as-is (avoid placeholders). Prefer standard library + common ML libs.
- Include main() and if __name__ == '__main__' guard.
- Include a small demo (e.g. simple prints or a quick example)."""
    elif lang_key == "java":
        rules = """- Output ONLY one code block (```java ... ```). No other prose.
- Single file with public class; use main(String[] args) as entry point.
- Use standard Java only (no external libs required).
- Include a small demo in main (e.g. build a tree and print)."""
    elif lang_key == "javascript":
        rules = """- Output ONLY one code block (```javascript ... ```). No other prose.
- Self-contained script; runnable in Node or browser console.
- Include a small demo at the bottom (e.g. console.log)."""
    elif lang_key == "cpp":
        rules = """- Output ONLY one code block (```cpp ... ```). No other prose.
- Single file with main() and necessary includes.
- Include a small demo in main."""
    else:
        rules = f"- Output ONLY one code block (```{lang_key} ... ```). No other prose. Include a small demo."

    prompt = f"""You are LearnSphere, an ML learning platform. We help users learn through code—focus on machine learning (ML) topics (e.g. models, data prep, training, sklearn/tensorflow, algorithms).

First, check:
1) Is "{topic}" an ML-related coding task or a clear technical task we can demonstrate with code? If NOT (e.g. "who am I", "what is love", life advice, philosophy, or clearly non-ML), do NOT generate code. Output ONLY a short, friendly message: say LearnSphere is for ML learning and suggest they try an ML topic (e.g. "Try 'linear regression in Python', 'decision tree with sklearn', or 'simple neural network'."). Put it in a single ``` block.
2) Is the request clear and specific enough? If it's vague (e.g. "give me code", "ML program", "something for classification"), do NOT guess. Output ONLY a short message asking them to clarify with an ML example (e.g. "Which ML concept? Try 'logistic regression in Python' or 'k-means clustering with sklearn'."). Put the message in a single ``` block.

If it IS a valid ML or technical coding topic AND specific enough, generate a single self-contained {lang_name} program for:

{topic}

{level_block}Language: English only.
Rules:
{rules}
Detail level: {depth_instruction}
Prefer ML-related code (data, models, training, evaluation) when the topic allows.
"""
    text = _call_model(os.getenv("GEMINI_CODE_MODEL", DEFAULT_CODE_MODEL), prompt,
                       temperature=0.3, max_output_tokens=2048)
    return _extract_code(text)


def generate_code_example_stream(
    api_key: Optional[str],
    topic: str,
    depth: str,
    language: str = "python",
    level_instructions: Optional[str] = None,
) -> Generator[str, None, None]:
    """Yield raw response chunks. Same prompt as generate_code_example. Caller must run _extract_code on full text when done."""
    _configure(api_key)
    depth_key = (depth or "detailed").strip().lower()
    lang_key = (language or "python").strip().lower()
    lang_name = CODE_LANGUAGES.get(lang_key, ("Python", "py"))[0]
    if depth_key in ("minimal", "standard"):
        depth_instruction = "Keep it minimal: clear working code and a short demo. No long comments or docstrings."
    else:
        depth_instruction = "Add inline comments on key steps and a small runnable demo. One short docstring/comment at the top is enough."
    level_block = f"Level: {level_instructions}. " if level_instructions else ""
    if lang_key == "python":
        rules = """- Output ONLY one code block (```python ... ```). No other prose.
- Must run as-is (avoid placeholders). Prefer standard library + common ML libs.
- Include main() and if __name__ == '__main__' guard.
- Include a small demo (e.g. simple prints or a quick example)."""
    elif lang_key == "java":
        rules = """- Output ONLY one code block (```java ... ```). No other prose.
- Single file with public class; use main(String[] args) as entry point.
- Use standard Java only (no external libs required).
- Include a small demo in main (e.g. build a tree and print)."""
    elif lang_key == "javascript":
        rules = """- Output ONLY one code block (```javascript ... ```). No other prose.
- Self-contained script; runnable in Node or browser console.
- Include a small demo at the bottom (e.g. console.log)."""
    elif lang_key == "cpp":
        rules = """- Output ONLY one code block (```cpp ... ```). No other prose.
- Single file with main() and necessary includes.
- Include a small demo in main."""
    else:
        rules = f"- Output ONLY one code block (```{lang_key} ... ```). No other prose. Include a small demo."
    prompt = f"""You are LearnSphere, an ML learning platform. We help users learn through code—focus on machine learning (ML) topics (e.g. models, data prep, training, sklearn/tensorflow, algorithms).

First, check:
1) Is "{topic}" an ML-related coding task or a clear technical task we can demonstrate with code? If NOT (e.g. "who am I", "what is love", life advice, philosophy, or clearly non-ML), do NOT generate code. Output ONLY a short, friendly message: say LearnSphere is for ML learning and suggest they try an ML topic (e.g. "Try 'linear regression in Python', 'decision tree with sklearn', or 'simple neural network'."). Put it in a single ``` block.
2) Is the request clear and specific enough? If it's vague (e.g. "give me code", "ML program", "something for classification"), do NOT guess. Output ONLY a short message asking them to clarify with an ML example (e.g. "Which ML concept? Try 'logistic regression in Python' or 'k-means clustering with sklearn'."). Put the message in a single ``` block.

If it IS a valid ML or technical coding topic AND specific enough, generate a single self-contained {lang_name} program for:

{topic}

{level_block}Language: English only.
Rules:
{rules}
Detail level: {depth_instruction}
Prefer ML-related code (data, models, training, evaluation) when the topic allows.
"""
    model = genai.GenerativeModel(os.getenv("GEMINI_CODE_MODEL", DEFAULT_CODE_MODEL))
    resp = model.generate_content(
        prompt,
        safety_settings=SAFETY_SETTINGS,
        generation_config={
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 2048,
        },
        stream=True,
    )
    for chunk in resp:
        if chunk.text:
            yield chunk.text


def generate_audio_script(
    api_key: Optional[str],
    topic: str,
    length: str,
    level_instructions: Optional[str] = None,
) -> str:
    _configure(api_key)
    length_key = (length or "brief").strip().lower()
    target = {
        "brief": "1-2 minutes when read aloud. Use 150-300 words MAXIMUM. Do not exceed 300 words.",
        "standard": "3-5 minutes when read aloud. Use 450-750 words MAXIMUM. Do not exceed 750 words.",
        "detailed": "6-8 minutes when read aloud. Use 900-1100 words MAXIMUM. Do not exceed 1100 words.",
    }.get(length_key, "3-5 minutes when read aloud. Use 450-750 words MAXIMUM. Do not exceed 750 words.")

    level_block = f"Level: {level_instructions}. Adapt vocabulary, pace, and depth to this level. " if level_instructions else ""

    prompt = f"""You are LearnSphere, an ML learning platform. Create an audio lesson script about machine learning concepts. Language: English only.
{level_block}
Topic or question: {topic}

Important:
- If the topic is an ML concept (algorithms, models, training, data, neural networks, etc.), create a warm, conversational lesson with the length and style below. Use the "ML instructor" tone.
- If the topic is NOT about ML (e.g. general or philosophical questions like "who am I", "what is happiness"), do NOT record a lesson. Instead, output exactly one line with only:
- NON_ML_TOPIC
Then stop. No other text.

Constraints for ML topics:
- Length: {target} (spoken pace ~150 words/minute). Stay within the word cap.
- Conversational, clear, no heavy jargon without explanation.
- Use occasional rhetorical questions to engage.
- Include 1 short recap and 3 key takeaways at the end.
- Avoid markdown headings; write as spoken narration.
- Output the COMPLETE script from start to finish. No cutting off mid-sentence.

Return only the full narration text, nothing else."""

    # Detailed (6-8 min) needs ~1200+ tokens; use 4096 so full script always fits
    max_tokens = 4096 if length_key == "detailed" else 2048
    return _call_model(os.getenv("GEMINI_AUDIO_MODEL", DEFAULT_AUDIO_SCRIPT_MODEL), prompt,
                       temperature=0.6, max_output_tokens=max_tokens)


def generate_image_prompts(api_key: Optional[str], topic: str, count: int = 3) -> List[str]:
    """Legacy: returns one prompt line per diagram (used for display). Use get_flowchart_steps_for_diagrams for actual diagram content."""
    steps_per_diagram = get_flowchart_steps_for_diagrams(api_key, topic, count)
    return [" → ".join(steps) for steps in steps_per_diagram]


def _parse_flowchart_blocks(raw: str, count: int) -> List[List[str]]:
    """Parse model output into blocks of step labels. Handles --- separators and strips numbers/bullets."""
    text = raw.strip()
    blocks = [b.strip() for b in re.split(r"\n\s*---\s*\n", text) if b.strip()]
    step_lists: List[List[str]] = []
    for block in blocks[: count * 2]:
        lines = []
        for ln in block.splitlines():
            ln = ln.strip()
            if not ln or ln.upper() in ("---", "DONE", "END"):
                continue
            ln = re.sub(r"^\s*[\d]+[.)]\s*", "", ln)
            ln = ln.strip(" -\t•")
            if ln and len(ln) < 80:
                lines.append(ln[:60])
        lines = lines[:8]
        if len(lines) >= 3:  # require at least 3 steps per flowchart
            step_lists.append(lines)
        elif len(lines) == 2 and step_lists:
            step_lists[-1].extend(lines)  # merge 2-step fragment into previous
        if len(step_lists) >= count:
            break
    # If we got one big block with many steps, split into multiple flowcharts
    if len(step_lists) == 1 and len(step_lists[0]) >= count * 2:
        steps = step_lists[0]
        chunk = max(3, (len(steps) + count - 1) // count)
        step_lists = [steps[i : i + chunk] for i in range(0, len(steps), chunk)][:count]
    return step_lists


def get_flowchart_steps_for_diagrams(api_key: Optional[str], topic: str, count: int = 3) -> List[List[str]]:
    """Return one list of step labels per diagram, so we can draw flowcharts that explain the topic visually."""
    _configure(api_key)
    prompt = f"""You are LearnSphere. You MUST create exactly {count} flowcharts. The user asked for {count} visuals, so output exactly {count} blocks of steps. Each block has 3 to 6 short step labels (one per line). After each block write a line with only --- (three hyphens). So you will have {count} blocks separated by ---.

Topic: {topic}

Rules:
- Block 1: MAIN algorithm or process steps in order (init, main loop, key ops, end). At least 3 steps.
- Block 2: Data structures, or another view (e.g. inputs/outputs). At least 3 steps.
- Block 3: Short example or key takeaway. At least 3 steps.
- If {count} >= 4: Block 4 can be complexity, common mistakes, or another angle. At least 3 steps.
- Output ONLY step labels and ---. No numbering, no extra text.

Example for "Dijkstra" with 3 blocks:
Initialize distances (source 0, others infinity)
Add source to priority queue
Extract node with minimum distance
Relax edges to unvisited neighbors
Mark visited and repeat until done
---
Graph as adjacency list
Distance and predecessor arrays
Min-priority queue
---
Start from node A
Update distances via neighbors
Shortest path to C found
---
"""

    raw = _call_model(os.getenv("GEMINI_IMAGE_PROMPT_MODEL", DEFAULT_IMAGE_PROMPT_MODEL), prompt,
                      temperature=0.5, max_output_tokens=800)
    step_lists = _parse_flowchart_blocks(raw, count)
    if not step_lists:
        step_lists = [["Overview", "Key idea", "Outcome"]]
    # If model returned only 1 block but user asked for more, split that block into count flowcharts (no duplicates)
    if len(step_lists) == 1 and count > 1 and len(step_lists[0]) >= count * 2:
        steps = step_lists[0]
        chunk = max(3, (len(steps) + count - 1) // count)
        step_lists = [steps[i : i + chunk] for i in range(0, len(steps), chunk)][:count]
    return step_lists[:count]


# Max images: at most 5.
MAX_IMAGE_PROMPTS = 3


_DEFAULT_JOKE = "LearnSphere focuses on ML concepts—try \"decision trees\", \"gradient descent\", or \"neural network layers\" for diagrams!"
_DEFAULT_CLARIFY = "Which ML concept would you like diagrams for? For example: 'decision trees', 'gradient descent', or 'backpropagation'."

_DEFAULT_TEXT_JOKE = "LearnSphere is for ML learning—try \"backpropagation\", \"decision trees\", or \"bias-variance tradeoff\" for an explanation."
_DEFAULT_TEXT_CLARIFY = "Which ML concept would you like explained? For example: 'how decision trees work', 'gradient descent', or 'neural network layers'."

_DEFAULT_CODE_JOKE = "LearnSphere is for ML code—try \"logistic regression in Python\", \"decision tree with sklearn\", or \"k-means clustering\"."
_DEFAULT_CODE_CLARIFY = "Which ML concept do you want code for? For example: 'linear regression in Python', 'k-means clustering with sklearn', or 'simple neural network'."


def validate_code_topic(api_key: Optional[str], topic: str) -> Tuple[str, Optional[str]]:
    """Check if the topic is ML-related for code generation. Returns ("ok", None), ("joke", message), or ("clarify", message)."""
    _configure(api_key)
    prompt = f"""You are LearnSphere, an ML learning platform. We only generate code for machine learning or clear technical tasks. The user entered: "{topic}"

Reply with exactly one of these three options (nothing else):

1) If this is an ML-related coding task or a clear technical task we can demonstrate with code (e.g. logistic regression, decision trees, k-means, neural network, data prep, sklearn, training loops), or a CS concept used with ML (e.g. trees, graphs, sorting), reply with only: OK

2) If this is NOT an ML or technical coding topic (philosophy, general knowledge, greeting, off-topic), reply on two lines:
Line 1: JOKE
Line 2: One short, friendly sentence suggesting they try an ML code topic (e.g. "LearnSphere writes ML code—try 'logistic regression in Python' or 'decision tree with sklearn'.")

3) If it is ambiguous or too vague (e.g. "give me code", "ML program"), reply on two lines:
Line 1: CLARIFY
Line 2: One polite sentence asking them to specify an ML concept with code.

No quotes, no extra text."""

    raw_orig = _call_model(
        os.getenv("GEMINI_CODE_MODEL", DEFAULT_CODE_MODEL),
        prompt,
        temperature=0.3,
        max_output_tokens=128,
    )
    raw = raw_orig.strip()
    raw_upper = raw.upper()
    if raw_upper.startswith("OK") and (len(raw_upper) <= 2 or raw_upper[2:].strip() == ""):
        return ("ok", None)
    if raw_upper.startswith("JOKE"):
        rest = raw[4:].strip() if len(raw) > 4 else ""
        msg = " ".join(ln.strip() for ln in rest.split() if ln.strip()).strip() or rest.strip()
        if len(msg) >= 50 and msg[-1] in ".!?":
            pass
        else:
            msg = _DEFAULT_CODE_JOKE
        return ("joke", msg)
    if raw_upper.startswith("CLARIFY"):
        rest = raw[7:].strip() if len(raw) > 7 else ""
        msg = " ".join(ln.strip() for ln in rest.split() if ln.strip()).strip() or rest.strip()
        if len(msg) >= 40 and msg[-1] in ".!?":
            pass
        else:
            msg = _DEFAULT_CODE_CLARIFY
        return ("clarify", msg)
    if len(raw_orig.strip()) < 30 and not raw_upper.startswith("OK"):
        return ("joke", _DEFAULT_CODE_JOKE)
    return ("ok", None)


def validate_text_topic(api_key: Optional[str], topic: str) -> Tuple[str, Optional[str]]:
    """Check if the topic is ML-related for text explanations. Returns ("ok", None), ("joke", message), or ("clarify", message)."""
    _configure(api_key)
    prompt = f"""You are LearnSphere, an ML learning platform. We only give text explanations for machine learning concepts. The user entered: "{topic}"

Reply with exactly one of these three options (nothing else):

1) If this is a clear ML concept (algorithm, model, training, data, neural networks, decision trees, gradient descent, bias-variance, overfitting, k-means, etc.) or an overview question ("what is ML", "explain machine learning"), or a CS concept used with ML (trees, graphs, complexity), reply with only: OK

2) If this is NOT an ML concept (philosophy, general knowledge, greeting, off-topic), reply on two lines:
Line 1: JOKE
Line 2: One short, friendly sentence suggesting they try an ML topic (e.g. "LearnSphere explains ML—try 'how decision trees work' or 'bias-variance tradeoff'.")

3) If it is ambiguous or not clearly an ML concept, reply on two lines:
Line 1: CLARIFY
Line 2: One polite sentence asking them to specify an ML concept.

No quotes, no extra text."""

    raw_orig = _call_model(
        os.getenv("GEMINI_TEXT_MODEL", DEFAULT_TEXT_MODEL),
        prompt,
        temperature=0.3,
        max_output_tokens=128,
    )
    raw = raw_orig.strip()
    raw_upper = raw.upper()
    if raw_upper.startswith("OK") and (len(raw_upper) <= 2 or raw_upper[2:].strip() == ""):
        return ("ok", None)
    if raw_upper.startswith("JOKE"):
        rest = raw[4:].strip() if len(raw) > 4 else ""
        msg = " ".join(ln.strip() for ln in rest.split() if ln.strip()).strip() or rest.strip()
        if len(msg) >= 50 and msg[-1] in ".!?":
            pass
        else:
            msg = _DEFAULT_TEXT_JOKE
        return ("joke", msg)
    if raw_upper.startswith("CLARIFY"):
        rest = raw[7:].strip() if len(raw) > 7 else ""
        msg = " ".join(ln.strip() for ln in rest.split() if ln.strip()).strip() or rest.strip()
        if len(msg) >= 40 and msg[-1] in ".!?":
            pass
        else:
            msg = _DEFAULT_TEXT_CLARIFY
        return ("clarify", msg)
    if len(raw_orig.strip()) < 30 and not raw_upper.startswith("OK"):
        return ("joke", _DEFAULT_TEXT_JOKE)
    return ("ok", None)


def validate_image_topic(api_key: Optional[str], topic: str) -> Tuple[str, Optional[str]]:
    """Check if the topic is an ML-related concept for diagrams. Returns ("ok", None), ("joke", message), or ("clarify", message)."""
    _configure(api_key)
    prompt = f"""You are LearnSphere, an ML learning platform. We only generate diagrams for machine learning concepts. The user entered: "{topic}"

Reply with exactly one of these three options (nothing else):

1) If this is a clear machine learning concept (algorithm, model, training idea, data concept, or ML process we can explain—e.g. decision trees, neural networks, gradient descent, backpropagation, bias-variance, overfitting, k-means), OR if the user is asking for an overview of ML itself (e.g. "explain machine learning", "what is ML"), OR if it is a data structure or CS concept that is used in or alongside ML (e.g. binary search tree, BST, trees, graphs, heaps, sorting, search, complexity, recursion—these help understand ML and algorithms), reply with only: OK

2) If this is NOT an ML concept (philosophical question, general knowledge, greeting, or off-topic), reply on two lines:
Line 1: JOKE
Line 2: One short, light-hearted sentence suggesting they try an ML topic (e.g. "LearnSphere draws ML concepts—try 'how gradient descent works' or 'decision tree splitting'.")

3) If it is ambiguous or not clearly an ML concept, reply on two lines:
Line 1: CLARIFY
Line 2: One polite sentence asking them to specify an ML concept (e.g. "Which ML concept do you want diagrams for? Try 'neural network layers' or 'k-means clustering'.")

No quotes, no extra text."""

    raw_orig = _call_model(
        os.getenv("GEMINI_IMAGE_PROMPT_MODEL", DEFAULT_IMAGE_PROMPT_MODEL),
        prompt,
        temperature=0.3,
        max_output_tokens=128,
    )
    raw = raw_orig.strip()
    raw_upper = raw.upper()
    if raw_upper.startswith("OK") and (len(raw_upper) <= 2 or raw_upper[2:].strip() == ""):
        return ("ok", None)
    if raw_upper.startswith("JOKE"):
        rest = raw[4:].strip() if len(raw) > 4 else ""
        msg = " ".join(ln.strip() for ln in rest.split() if ln.strip()).strip() or rest.strip()
        # Use our default unless model gave a complete sentence (long enough and ends with . ! ?)
        if len(msg) >= 50 and msg[-1] in ".!?":
            pass  # use msg
        else:
            msg = _DEFAULT_JOKE
        return ("joke", msg)
    if raw_upper.startswith("CLARIFY"):
        rest = raw[7:].strip() if len(raw) > 7 else ""
        msg = " ".join(ln.strip() for ln in rest.split() if ln.strip()).strip() or rest.strip()
        if len(msg) >= 40 and msg[-1] in ".!?":
            pass
        else:
            msg = _DEFAULT_CLARIFY
        return ("clarify", msg)
    if len(raw_orig.strip()) < 30 and not raw_upper.startswith("OK"):
        return ("joke", _DEFAULT_JOKE)
    return ("ok", None)


def get_image_generation_prompts(
    api_key: Optional[str],
    topic: str,
    *,
    count: Optional[int] = None,
    level_instructions: Optional[str] = None,
) -> List[str]:
    """Ask the AI for image-generation prompts for the topic. Returns variable-length list (max MAX_IMAGE_PROMPTS).
    The count argument is ignored (kept for backward compatibility)."""
    _configure(api_key)
    level_block = f"Level: {level_instructions}. For beginner: ask for simple diagrams with minimal text and clear shapes. For advanced: allow denser, more technical diagrams. " if level_instructions else ""
    prompt = f"""You are LearnSphere, an ML learning platform. The user wants educational diagram images for this machine learning topic: {topic}
{level_block}Language: English only.

Output as many image-generation prompts as appropriate (cover process steps, key concepts, structure; one prompt per line). Include at least one flowchart (steps, boxes, decision flow). Focus on ML concepts: algorithms, models, data flow, training, etc.
Each line must be a single, clear prompt (one or two sentences) that would produce a helpful ML educational diagram. No numbering, no bullets, no extra text. Style: clean, educational, suitable for students learning ML at the given level.
Topic: {topic}
Output only the prompts, one per line."""

    raw = _call_model(
        os.getenv("GEMINI_IMAGE_PROMPT_MODEL", DEFAULT_IMAGE_PROMPT_MODEL),
        prompt,
        temperature=0.5,
        max_output_tokens=2048,
    )
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^\s*[\d]+[.)]\s*", "", ln).strip(" -\t•")
        if ln and len(ln) < 500:
            cleaned.append(ln[:400])
    if not cleaned:
        cleaned = [f"Educational diagram explaining: {topic}"]
    return cleaned[:MAX_IMAGE_PROMPTS]


def _generate_one_image(
    key: str,
    text_prompt: str,
    out_dir: str,
    topic_slug: str,
    index: int,
) -> str:
    """Generate a single image via Gemini API; save to out_dir. Returns filename."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_IMAGE_GEN_MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": key}
    full_prompt = (
        "Create a complete educational diagram that fits entirely in the image with nothing cut off. "
        + text_prompt
    )
    body = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": "3:4",
                "imageSize": "2K",
            },
        },
    }
    r = requests.post(url, json=body, headers=headers, timeout=120)
    r.raise_for_status()
    data = r.json()
    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts") or []
    image_data_b64: Optional[str] = None
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            image_data_b64 = inline.get("data")
            break
    if not image_data_b64:
        msg = (data.get("candidates") or [{}])[0].get("finishReason") or data.get("error", {}).get("message") or r.text
        raise RuntimeError(f"Gemini image model returned no image (prompt {index + 1}). {msg}")
    raw_bytes = base64.b64decode(image_data_b64)
    safe = re.sub(r"[^\w\-]", "_", topic_slug)[:40] or "diagram"
    filename = f"gemini_{safe}_{int(time.time())}_{index}.png"
    filepath = os.path.join(out_dir, filename)
    with open(filepath, "wb") as f:
        f.write(raw_bytes)
    return filename


def generate_images_via_gemini_api(
    api_key: Optional[str],
    prompts: List[str],
    out_dir: str,
    topic_slug: str = "diagram",
) -> List[str]:
    """Call Gemini image-generation API (REST) for each prompt; save PNGs to out_dir. Returns list of filenames."""
    key = _get_api_key(api_key)
    filenames: List[str] = []
    for i, text_prompt in enumerate(prompts):
        try:
            fn = _generate_one_image(key, text_prompt, out_dir, topic_slug, i)
            filenames.append(fn)
        except requests.RequestException as e:
            raise RuntimeError(f"Gemini image API request failed: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"Gemini image API invalid JSON: {e}") from e
    return filenames


def generate_one_image_via_gemini_api(
    api_key: Optional[str],
    prompt: str,
    out_dir: str,
    topic_slug: str = "diagram",
    index: int = 0,
) -> str:
    """Generate a single image; returns filename. Used for streaming one-by-one."""
    key = _get_api_key(api_key)
    return _generate_one_image(key, prompt, out_dir, topic_slug, index)


def get_model_info() -> Dict[str, str]:
    """Surface model choices in one place."""
    return {
        "text_model": os.getenv("GEMINI_TEXT_MODEL", DEFAULT_TEXT_MODEL),
        "code_model": os.getenv("GEMINI_CODE_MODEL", DEFAULT_CODE_MODEL),
        "audio_model": os.getenv("GEMINI_AUDIO_MODEL", DEFAULT_AUDIO_SCRIPT_MODEL),
        "image_prompt_model": os.getenv("GEMINI_IMAGE_PROMPT_MODEL", DEFAULT_IMAGE_PROMPT_MODEL),
        "image_gen_model": os.getenv("GEMINI_IMAGE_GEN_MODEL", GEMINI_IMAGE_GEN_MODEL),
    }
