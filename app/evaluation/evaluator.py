from __future__ import annotations

from typing import Dict, List, Tuple

from app.evaluation.metrics import (
    groundedness_score,
    keyword_coverage,
    length_compliance,
    readability_score,
    structure_score,
)


class ArticleEvaluator:
    def evaluate(
        self,
        article: str,
        seo_keywords: List[str],
        retrieved_chunks: List[Tuple[str, float]],
    ) -> Dict[str, float]:
        scores = {
            "keyword_coverage": keyword_coverage(article, seo_keywords),
            "length_compliance": length_compliance(article),
            "structure_score": structure_score(article),
            "groundedness_score": groundedness_score(article, retrieved_chunks),
            "readability_score": readability_score(article),
        }
        scores["final_score"] = (
            0.25 * scores["keyword_coverage"]
            + 0.20 * scores["length_compliance"]
            + 0.25 * scores["structure_score"]
            + 0.20 * scores["groundedness_score"]
            + 0.10 * scores["readability_score"]
        )
        return scores
