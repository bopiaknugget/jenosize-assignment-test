from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.articles import router as article_router

app = FastAPI(title="Jenosize Trend Article Generator", version="0.1.0")
app.include_router(article_router, prefix="/v1/articles", tags=["articles"])


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
