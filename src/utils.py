"""Shared utilities: JSONL/JSON IO, config loading, image encoding."""
from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import yaml


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(:-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def sub(m):
            name, _, default = m.groups()
            return os.environ.get(name, default or "")
        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _expand_env(cfg)


def encode_image_b64(path: str | Path, max_side: int = 768) -> str:
    """Resize-then-base64-encode an image for vision API calls."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val
