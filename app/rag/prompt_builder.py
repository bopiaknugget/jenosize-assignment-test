from __future__ import annotations

from typing import List, Tuple


SYSTEM_PROMPT = (
    "You are a senior business content strategist. "
    "Write insightful, future-focused business articles for executive readers. "
    "Use the provided retrieved context as the main grounding source. "
    "Do not introduce unsupported claims beyond the retrieved content."
)


def build_grounded_user_prompt(
    topic_category: str,
    industry: str,
    target_audience: str,
    seo_keywords: List[str],
    retrieved_chunks: List[Tuple[str, float]],
    article_length: str,
) -> str:
    source_blocks = []
    for idx, (chunk, score) in enumerate(retrieved_chunks, start=1):
        source_blocks.append(f"[Source Chunk {idx} | relevance={score:.4f}]\n{chunk}")

    source_text = "\n\n".join(source_blocks) if source_blocks else "No retrieved context provided."

    return f"""
Generate a business trend / future ideas article.

Topic Category: {topic_category}
Industry: {industry}
Target Audience: {target_audience}
SEO Keywords: {", ".join(seo_keywords)}
Desired Length: {article_length}

Retrieved Context:
{source_text}

Requirements:
1. Write a compelling title.
2. Start with a strong executive hook.
3. Use section headings.
4. Explain business implications and future opportunities.
5. Naturally incorporate SEO keywords.
6. End with a forward-looking conclusion.
7. Do not invent unsupported facts.
""".strip()
