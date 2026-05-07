"""Extract a 1-5 scalar rating from a model's raw text output.

Order of attempts:
    1. The line `Rating: <int>` (CoT format).
    2. A bare integer somewhere in the trailing line.
    3. The first integer 1-5 anywhere in the response.

Returns None if no rating can be parsed.
"""
from __future__ import annotations

import re

_RATING_LINE = re.compile(r"(?:^|\n)\s*Rating\s*:\s*([1-5])\s*(?:\.|$|\n)", re.IGNORECASE)
_BARE_INT   = re.compile(r"\b([1-5])\b")


def parse_rating(raw: str | None) -> float | None:
    if not raw:
        return None
    text = raw.strip()
    m = _RATING_LINE.search(text)
    if m:
        return float(m.group(1))
    # Look at the last non-empty line first (common CoT trailing pattern).
    for line in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        m = _BARE_INT.search(line)
        if m:
            return float(m.group(1))
    return None
