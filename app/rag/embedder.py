from __future__ import annotations

from typing import Iterable

from sentence_transformers import SentenceTransformer

from app.config import CONFIG


class TextEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or CONFIG.retrieval.embedding_model_name
        self.model = SentenceTransformer(self.model_name)

    def encode_texts(self, texts: Iterable[str]):
        return self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_query(self, query: str):
        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
