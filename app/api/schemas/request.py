from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class GenerateArticleRequest(BaseModel):
    topic_category: str = Field(..., examples=["Artificial Intelligence in Customer Experience"])
    industry: str = Field(..., examples=["Retail Banking"])
    target_audience: str = Field(..., examples=["Business Executives"])
    source_content: str = Field(..., examples=["Retail banks are increasingly using AI..."])
    seo_keywords: List[str] = Field(default_factory=list)
    article_length: str = Field(default="900-1200 words")
    top_k: int = Field(default=4, ge=1, le=10)
