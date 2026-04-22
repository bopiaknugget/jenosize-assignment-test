from __future__ import annotations

from typing import List

from app.config import CONFIG
from app.utils.text import normalize_whitespace


class TextChunker:
    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        self.chunk_size = chunk_size or CONFIG.retrieval.chunk_size
        self.overlap = overlap or CONFIG.retrieval.chunk_overlap

    def chunk(self, text: str) -> List[str]:
        text = normalize_whitespace(text)
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        size = len(text)

        while start < size:
            end = min(start + self.chunk_size, size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= size:
                break
            start = max(end - self.overlap, 0)

        return chunks
