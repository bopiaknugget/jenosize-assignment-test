from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.utils.text import normalize_whitespace


SYSTEM_PROMPT = (
    "You are a senior business content strategist. "
    "Write insightful, future-focused business articles for executive readers."
)


def parse_keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def build_user_prompt(row: Dict[str, Any]) -> str:
    return f"""
Generate a business trend / future ideas article using the following input.

Topic Category: {row['topic_category']}
Industry: {row['industry']}
Target Audience: {row['target_audience']}
SEO Keywords: {', '.join(row['seo_keywords'])}
Desired Length: {row.get('desired_length', '900-1200 words')}

Source Content:
{row['source_content']}

Requirements:
1. Write a compelling title.
2. Start with a strong executive hook.
3. Use section headings.
4. Explain business implications and future opportunities.
5. Naturally incorporate SEO keywords.
6. End with a forward-looking conclusion.
""".strip()


def build_assistant_output(row: Dict[str, Any]) -> str:
    title = str(row.get("article_title", "")).strip()
    article_body = normalize_whitespace(str(row["article_body"]))
    return f"# {title}\n\n{article_body}" if title else article_body


def convert_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(row)},
            {"role": "assistant", "content": build_assistant_output(row)},
        ]
    }


def load_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported source format: {path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert processed CSV/JSON/JSONL into train/val chat-style JSONL files.')
    parser.add_argument('--source-path', type=Path, default=Path('data/processed/article_training_source.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/training'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source_path
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(
            f'Missing source file: {source_path}. Run training/bootstrap_hf_dataset.py first or provide --source-path.'
        )

    df = load_source(source_path)
    required = {"topic_category", "industry", "target_audience", "source_content", "seo_keywords", "article_body"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["source_content", "article_body"]).copy()
    df["source_content"] = df["source_content"].astype(str).map(normalize_whitespace)
    df["article_body"] = df["article_body"].astype(str).map(normalize_whitespace)
    df["seo_keywords"] = df["seo_keywords"].map(parse_keywords)
    df = df[(df["source_content"].str.len() > 150) & (df["article_body"].str.len() > 400)].reset_index(drop=True)

    examples = [convert_row(row) for row in df.to_dict(orient="records")]
    split = int(len(examples) * 0.9)
    train_data = examples[:split]
    val_data = examples[split:]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as handle:
        for item in train_data:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as handle:
        for item in val_data:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(val_data)} validation samples to {val_path}")


if __name__ == "__main__":
    main()
