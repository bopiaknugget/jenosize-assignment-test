from __future__ import annotations

from typing import List, Sequence, Tuple

import faiss
import numpy as np


class FaissIndexer:
    def __init__(self, embedding_dim: int) -> None:
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.payloads: List[str] = []

    def add(self, embeddings: np.ndarray, payloads: Sequence[str]) -> None:
        if len(embeddings) != len(payloads):
            raise ValueError("embeddings and payloads must have the same length")
        self.index.add(embeddings.astype("float32"))
        self.payloads.extend(payloads)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[str, float]]:
        query_embedding = np.array([query_embedding]).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.payloads[idx], float(score)))
        return results
