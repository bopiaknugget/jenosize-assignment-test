from __future__ import annotations

import argparse
from pathlib import Path

from app.config import CONFIG
from app.evaluation.tuner import GenerationConfig
from app.rag.prompt_builder import SYSTEM_PROMPT
from app.services.generator import ArticleGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test generation with either a base model or a fine-tuned adapter.")
    parser.add_argument(
        "--model-strategy",
        choices=["accuracy", "speed", "balance"],
        help="Shortcut strategy that resolves to a predefined Hugging Face model ID.",
    )
    parser.add_argument(
        "--base-model-name",
        help="Exact Hugging Face model ID. Overrides --model-strategy when both are provided.",
    )
    parser.add_argument(
        "--model-dir",
        help="Adapter directory to load first. Falls back to base model if adapter_config.json is not present.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Generation length for the smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_model_name = CONFIG.resolve_generation_model(args.base_model_name or args.model_strategy)
    model_dir = Path(args.model_dir) if args.model_dir else CONFIG.training.output_dir

    print(f"Using base model: {base_model_name}")
    print(f"Checking adapter dir: {model_dir}")

    generator = ArticleGenerator(model_dir=str(model_dir), base_model_name=base_model_name)
    config = GenerationConfig(max_new_tokens=args.max_new_tokens, retrieval_top_k=4)

    user_prompt = """
Generate a business trend article.

Topic Category: Artificial Intelligence in Customer Experience
Industry: Retail Banking
Target Audience: Business Executives
SEO Keywords: AI banking trends, future of customer experience, retail banking innovation
Desired Length: 900-1200 words

Retrieved Context:
[Source Chunk 1 | relevance=0.9500]
Retail banks are increasingly using AI to personalize customer interactions, reduce call center load, speed up onboarding, improve fraud detection, and offer proactive product recommendations.

[Source Chunk 2 | relevance=0.9200]
However, many banks still struggle to unify customer data across channels, creating fragmented experiences that weaken retention and cross-sell performance.

Requirements:
1. Write a compelling title.
2. Start with a strong executive hook.
3. Use section headings.
4. Explain business implications and future opportunities.
5. Naturally incorporate SEO keywords.
6. End with a forward-looking conclusion.
7. Do not invent unsupported facts.
""".strip()

    article = generator.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, config=config)
    print(article)


if __name__ == "__main__":
    main()
