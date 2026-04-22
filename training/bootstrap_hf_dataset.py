from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import load_dataset

from app.utils.text import normalize_whitespace

DEFAULT_OUTPUT = Path('data/processed/article_training_source.csv')
MEDIUM_DATASET = 'Alaamer/medium-articles-posts-with-content'
REUTERS_DATASET = 'danidanou/Reuters_Financial_News'

STOPWORDS = {
    'about','after','again','against','also','among','because','being','between','could','their','there',
    'these','those','through','under','using','with','where','which','while','would','from','into','than',
    'have','has','had','this','that','they','them','your','ours','into','more','most','over','such','what',
    'when','will','been','than','then','very','much','many','some','each','only','just','than','make','made',
    'business','future','trend','trends','article','ideas','idea','executive','executives'
}

RELEVANCE_TERMS = {
    'ai','artificial intelligence','automation','digital','transformation','technology','innovation','consumer',
    'marketing','retail','banking','finance','financial','customer','platform','data','cloud','startup',
    'enterprise','strategy','operations','experience','supply chain','ecommerce','commerce','growth',
    'industry','mobile','payments','analytics','personalization','sustainability'
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Bootstrap a prototype training CSV from Hugging Face datasets for the Jenosize article generator.'
    )
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Path to output CSV.')
    parser.add_argument('--medium-limit', type=int, default=80, help='Maximum filtered samples from the Medium dataset.')
    parser.add_argument('--reuters-limit', type=int, default=40, help='Maximum filtered samples from the Reuters dataset.')
    parser.add_argument(
        '--target-audience',
        default='Business Executives',
        help='Default target audience label for generated training rows.',
    )
    parser.add_argument('--no-streaming', action='store_true', help='Disable streaming dataset reads.')
    return parser.parse_args()


def first_present(record: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = normalize_whitespace(str(value))
        if text and text.lower() != 'none':
            return text
    return ''


def normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_whitespace(str(v)).lower() for v in value if str(v).strip()]
    text = normalize_whitespace(str(value)).lower()
    if not text:
        return []
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
    parts = re.split(r'[|,;/]', text)
    return [p.strip() for p in parts if p.strip()]


def looks_relevant(title: str, content: str, tags: list[str]) -> bool:
    tag_text = ' '.join(tags)
    haystack = f'{title} {content[:1800]} {tag_text}'.lower()
    matches = sum(1 for term in RELEVANCE_TERMS if term in haystack)
    return matches >= 2


def infer_topic_category(title: str, content: str, tags: list[str]) -> str:
    tag_text = ' '.join(tags)
    text = f'{title} {content[:2000]} {tag_text}'.lower()
    if any(term in text for term in ['ai', 'artificial intelligence', 'automation', 'machine learning']):
        return 'AI & Automation'
    if any(term in text for term in ['digital transformation', 'cloud', 'platform', 'modernization']):
        return 'Digital Transformation'
    if any(term in text for term in ['customer', 'consumer', 'experience', 'personalization']):
        return 'Customer Experience'
    if any(term in text for term in ['startup', 'innovation', 'technology']):
        return 'Technology Trends'
    if any(term in text for term in ['sustainability', 'climate', 'green']):
        return 'Sustainability'
    return 'Business Trends'


def infer_industry(title: str, content: str, tags: list[str]) -> str:
    tag_text = ' '.join(tags)
    text = f'{title} {content[:2000]} {tag_text}'.lower()
    if any(term in text for term in ['bank', 'banking', 'fintech', 'finance', 'financial']):
        return 'Financial Services'
    if any(term in text for term in ['retail', 'ecommerce', 'commerce', 'shopping']):
        return 'Retail'
    if any(term in text for term in ['health', 'healthcare', 'pharma', 'medical']):
        return 'Healthcare'
    if any(term in text for term in ['manufacturing', 'supply chain', 'logistics']):
        return 'Manufacturing'
    if any(term in text for term in ['media', 'advertising', 'marketing']):
        return 'Marketing & Media'
    return 'Cross-Industry'


def build_seo_keywords(title: str, topic_category: str, industry: str, content: str) -> list[str]:
    tokens = re.findall(r'[A-Za-z][A-Za-z\-]{3,}', f'{title} {content[:300]}'.lower())
    counts = Counter(token for token in tokens if token not in STOPWORDS)
    lead_terms = [word for word, _ in counts.most_common(2)]
    keywords = [
        f'{topic_category.lower()} {industry.lower()}'.replace('&', 'and'),
        topic_category.lower().replace('&', 'and'),
        industry.lower(),
    ]
    keywords.extend(lead_terms)

    deduped: list[str] = []
    for kw in keywords:
        kw_clean = normalize_whitespace(kw).strip(' -')
        if len(kw_clean) < 4:
            continue
        if kw_clean not in deduped:
            deduped.append(kw_clean)
    return deduped[:4]


def build_row(title: str, article_body: str, source_content: str, tags: list[str], source_dataset: str, target_audience: str) -> dict[str, Any]:
    topic_category = infer_topic_category(title, article_body, tags)
    industry = infer_industry(title, article_body, tags)
    seo_keywords = build_seo_keywords(title, topic_category, industry, article_body)
    return {
        'topic_category': topic_category,
        'industry': industry,
        'target_audience': target_audience,
        'seo_keywords': str(seo_keywords),
        'source_content': source_content,
        'article_title': title,
        'article_body': article_body,
        'desired_length': '900-1200 words',
        'source_dataset': source_dataset,
    }


def iter_medium_rows(limit: int, target_audience: str, streaming: bool) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(MEDIUM_DATASET, split='train', streaming=streaming)
    count = 0
    seen_titles: set[str] = set()

    for record in dataset:
        title = first_present(record, ['title', 'Title', 'headline', 'Headline'])
        article_body = first_present(record, ['content', 'Content', 'text', 'article', 'body'])
        tags = normalize_tags(record.get('tags') or record.get('category') or record.get('categories'))

        if len(article_body) < 1200 or not title:
            continue
        if not looks_relevant(title, article_body, tags):
            continue
        normalized_title = title.lower()
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)

        source_content = article_body[:1200]
        yield build_row(title, article_body, source_content, tags, MEDIUM_DATASET, target_audience)
        count += 1
        if count >= limit:
            break


def iter_reuters_rows(limit: int, target_audience: str, streaming: bool) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(REUTERS_DATASET, split='train', streaming=streaming)
    count = 0
    seen_titles: set[str] = set()

    for record in dataset:
        title = first_present(record, ['Headline', 'headline', 'title', 'Title'])
        article_body = first_present(record, ['Article', 'article', 'content', 'text'])
        summary = first_present(record, ['Summary', 'summary', 'description'])

        if len(article_body) < 800 or not title:
            continue
        normalized_title = title.lower()
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)

        source_content = summary or article_body[:900]
        tags = ['finance', 'business', 'market']
        yield build_row(title, article_body, source_content, tags, REUTERS_DATASET, target_audience)
        count += 1
        if count >= limit:
            break


def main() -> None:
    args = parse_args()
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(iter_medium_rows(args.medium_limit, args.target_audience, not args.no_streaming))
    rows.extend(iter_reuters_rows(args.reuters_limit, args.target_audience, not args.no_streaming))

    if not rows:
        raise RuntimeError('No rows were collected from Hugging Face datasets. Try smaller limits or disable streaming.')

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=['article_title']).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f'Saved {len(df)} rows to {output_path}')
    print('Source datasets:')
    print(f'- {MEDIUM_DATASET}')
    print(f'- {REUTERS_DATASET}')


if __name__ == '__main__':
    main()
