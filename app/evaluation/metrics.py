from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def keyword_coverage(article: str, seo_keywords: List[str]) -> float:
    if not seo_keywords:
        return 1.0
    article_norm = normalize_text(article)
    hits = 0
    for keyword in seo_keywords:
        keyword = normalize_text(keyword)
        if keyword and keyword in article_norm:
            hits += 1
    return hits / len(seo_keywords)


def length_compliance(article: str, min_words: int = 700, max_words: int = 1400) -> float:
    count = len(article.split())
    if min_words <= count <= max_words:
        return 1.0
    if count < min_words:
        return max(0.0, count / min_words)
    overflow = count - max_words
    penalty = min(1.0, overflow / max_words)
    return max(0.0, 1.0 - penalty)


def structure_score(article: str) -> float:
    article = article.strip()
    title = article.startswith("#")
    heading_count = len(re.findall(r"^#{1,3}\s+.+$", article, flags=re.MULTILINE))
    conclusion = any(token in article.lower() for token in ["conclusion", "looking ahead", "final thought"])
    has_paragraphs = len([x for x in article.split("\n\n") if x.strip()]) >= 3

    score = 0.0
    score += 0.25 if title else 0.0
    score += 0.35 if heading_count >= 2 else 0.15 if heading_count == 1 else 0.0
    score += 0.20 if has_paragraphs else 0.0
    score += 0.20 if conclusion else 0.0
    return min(score, 1.0)


def groundedness_score(article: str, retrieved_chunks: List[Tuple[str, float]]) -> float:
    if not retrieved_chunks:
        return 0.5
    article_tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", article.lower()))
    source_tokens = set()
    for chunk, _ in retrieved_chunks:
        source_tokens.update(re.findall(r"\b[a-zA-Z]{4,}\b", chunk.lower()))
    if not article_tokens or not source_tokens:
        return 0.5
    overlap = article_tokens.intersection(source_tokens)
    return min(1.0, len(overlap) / max(20, len(article_tokens)) * 4)


def readability_score(article: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", article) if s.strip()]
    if not sentences:
        return 0.0
    avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
    word_counts = Counter(re.findall(r"\b\w+\b", article.lower()))
    repeated_terms = sum(1 for _, count in word_counts.items() if count > 8)

    score = 1.0
    if avg_sentence_len > 30:
        score -= 0.25
    if avg_sentence_len < 6:
        score -= 0.15
    if repeated_terms > 10:
        score -= 0.25
    return max(0.0, min(score, 1.0))
