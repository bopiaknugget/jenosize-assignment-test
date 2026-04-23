from __future__ import annotations

from typing import Dict, List, Tuple

from app.config import CONFIG
from app.evaluation.evaluator import ArticleEvaluator
from app.evaluation.tuner import GenerationConfig, SimpleTuner
from app.rag.prompt_builder import SYSTEM_PROMPT, build_grounded_user_prompt
from app.rag.retriever import Retriever
from app.services.generator import ArticleGenerator


class ArticlePipeline:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.generator = ArticleGenerator()
        self.evaluator = ArticleEvaluator()
        self.tuner = SimpleTuner()

    def run(
        self,
        topic_category: str,
        industry: str,
        target_audience: str,
        source_content: str,
        seo_keywords: List[str],
        article_length: str,
        top_k: int,
    ) -> Dict[str, object]:
        query = self.retriever.build_query(
            topic_category=topic_category,
            industry=industry,
            target_audience=target_audience,
            seo_keywords=seo_keywords,
        )

        config = GenerationConfig(retrieval_top_k=top_k)
        best_article = ""
        best_scores: Dict[str, float] = {"final_score": 0.0}
        best_chunks: List[Tuple[str, float]] = []

        for _ in range(1): # Single iteration for now; can be increased for multiple retries
            chunks = self.retriever.retrieve(
                source_text=source_content,
                query=query,
                top_k=config.retrieval_top_k,
            )

            user_prompt = build_grounded_user_prompt(
                topic_category=topic_category,
                industry=industry,
                target_audience=target_audience,
                seo_keywords=seo_keywords,
                retrieved_chunks=chunks,
                article_length=article_length,
            )

            article = self.generator.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config=config,
            )

            scores = self.evaluator.evaluate(
                article=article,
                seo_keywords=seo_keywords,
                retrieved_chunks=chunks,
            )

            if scores["final_score"] > best_scores.get("final_score", 0.0):
                best_article = article
                best_scores = scores
                best_chunks = chunks

            if scores["final_score"] >= CONFIG.runtime.single_retry_score_threshold:
                break

            config = self.tuner.tune(scores, config)

        return {
            "article": best_article,
            "retrieved_chunks": [
                {"text": text, "score": score} for text, score in best_chunks
            ],
            "scores": best_scores,
            "generation_config": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_new_tokens": config.max_new_tokens,
                "retrieval_top_k": config.retrieval_top_k,
                "repetition_penalty": config.repetition_penalty,
            },
        }
