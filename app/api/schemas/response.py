from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    text: str
    score: float


class GenerateArticleResponse(BaseModel):
    article: str
    retrieved_chunks: List[RetrievedChunk]
    scores: Dict[str, float]
    generation_config: Dict[str, float | int]
