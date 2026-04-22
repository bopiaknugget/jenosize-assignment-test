from __future__ import annotations

from typing import List, Tuple

from app.config import CONFIG
from app.rag.chunker import TextChunker
from app.rag.embedder import TextEmbedder
from app.rag.indexer import FaissIndexer
from app.utils.text import strip_basic_html


class Retriever:
    def __init__(self) -> None:
        self.chunker = TextChunker()
        self.embedder = TextEmbedder()

    def clean_source(self, text: str) -> str:
        return strip_basic_html(text)

    def build_query(
        self,
        topic_category: str,
        industry: str,
        target_audience: str,
        seo_keywords: list[str],
    ) -> str:
        return (
            f"Topic: {topic_category}. "
            f"Industry: {industry}. "
            f"Audience: {target_audience}. "
            f"Keywords: {', '.join(seo_keywords)}. "
            "Focus on business trends, future implications, opportunities, and pain points."
        )

    def retrieve(
        self,
        source_text: str,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[str, float]]:
        top_k = top_k or CONFIG.retrieval.default_top_k
        source_text = self.clean_source(source_text)
        chunks = self.chunker.chunk(source_text)
        if not chunks:
            return []

        embeddings = self.embedder.encode_texts(chunks)
        indexer = FaissIndexer(embedding_dim=embeddings.shape[1])
        indexer.add(embeddings, chunks)
        query_embedding = self.embedder.encode_query(query)
        return indexer.search(query_embedding, top_k=top_k)
