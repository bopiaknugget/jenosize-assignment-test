from __future__ import annotations

import re
from typing import Iterable, List


def normalize_whitespace(text: str) -> str:
    text = text or ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_basic_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(text)


def safe_join_lines(lines: Iterable[str]) -> str:
    clean: List[str] = []
    for line in lines:
        line = normalize_whitespace(line)
        if line:
            clean.append(line)
    return "\n".join(clean)


def extract_words(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def word_count(text: str) -> int:
    return len(extract_words(text))
