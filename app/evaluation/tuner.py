from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 700
    retrieval_top_k: int = 4
    repetition_penalty: float = 1.1


class SimpleTuner:
    def tune(self, scores: Dict[str, float], config: GenerationConfig) -> GenerationConfig:
        tuned = GenerationConfig(
            temperature=config.temperature,
            top_p=config.top_p,
            max_new_tokens=config.max_new_tokens,
            retrieval_top_k=config.retrieval_top_k,
            repetition_penalty=config.repetition_penalty,
        )

        if scores.get("keyword_coverage", 1.0) < 0.7:
            tuned.temperature = max(0.5, tuned.temperature - 0.1)
        if scores.get("structure_score", 1.0) < 0.7:
            tuned.temperature = max(0.45, tuned.temperature - 0.1)
        if scores.get("groundedness_score", 1.0) < 0.6:
            tuned.retrieval_top_k = min(6, tuned.retrieval_top_k + 1)
        if scores.get("length_compliance", 1.0) < 0.8:
            tuned.max_new_tokens = min(900, tuned.max_new_tokens + 100)
        if scores.get("readability_score", 1.0) < 0.7:
            tuned.repetition_penalty = min(1.25, tuned.repetition_penalty + 0.05)
            tuned.temperature = max(0.45, tuned.temperature - 0.05)

        return tuned
