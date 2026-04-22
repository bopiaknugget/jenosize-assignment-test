from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas.request import GenerateArticleRequest
from app.api.schemas.response import GenerateArticleResponse
from app.services.article_pipeline import ArticlePipeline

router = APIRouter()
pipeline = ArticlePipeline()


@router.post("/generate", response_model=GenerateArticleResponse)
def generate_article(payload: GenerateArticleRequest) -> GenerateArticleResponse:
    try:
        result = pipeline.run(
            topic_category=payload.topic_category,
            industry=payload.industry,
            target_audience=payload.target_audience,
            source_content=payload.source_content,
            seo_keywords=payload.seo_keywords,
            article_length=payload.article_length,
            top_k=payload.top_k,
        )
        return GenerateArticleResponse(**result)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc
