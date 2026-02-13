from __future__ import annotations

import ast
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)

# Map common import names to pip packages
IMPORT_TO_PIP = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "statsmodels": "statsmodels",
}


def detect_dependencies(code: str) -> List[str]:
    """Detect likely third-party deps from imports using AST."""
    if not code or not code.strip():
        return []

    imports: Set[str] = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
    except SyntaxError:
        # If code isn't parseable, do a best-effort regex fallback
        for line in code.splitlines():
            line = line.strip()
            if line.startswith("import "):
                imports.add(line.split()[1].split(".")[0])
            elif line.startswith("from "):
                parts = line.split()
                if len(parts) >= 2:
                    imports.add(parts[1].split(".")[0])

    # Filter standard lib imports crudely: only keep known ML/DS libs + common third party
    deps = set()
    for mod in imports:
        pip_name = IMPORT_TO_PIP.get(mod)
        if pip_name:
            deps.add(pip_name)

    return sorted(deps)


def validate_code(code: str) -> None:
    """Raise if code does not compile."""
    compile(code, "<generated>", "exec")


def get_extension_for_language(language: str) -> str:
    ext = {"python": "py", "java": "java", "javascript": "js", "cpp": "cpp"}.get(
        (language or "python").strip().lower(), "py"
    )
    return ext


def save_code_to_file(code: str, topic: str, out_dir: str, language: str = "python") -> str:
    if not code or not code.strip():
        raise ValueError("Generated code is empty.")

    ext = get_extension_for_language(language)
    safe_topic = "".join(ch for ch in (topic or "topic") if ch.isalnum() or ch in ("-", "_", " ")).strip()
    safe_topic = (safe_topic[:30] or "topic").replace(" ", "_")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"code_{safe_topic}_{ts}.{ext}"

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / filename
    path.write_text(code, encoding="utf-8")

    logger.info("Saved code: %s", path)
    return filename
