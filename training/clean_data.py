from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from app.utils.text import normalize_whitespace, strip_basic_html


DEFAULT_INPUT = Path("data/processed/article_training_source.csv")
DEFAULT_OUTPUT = Path("data/processed/article_training_source_clean.csv")

REQUIRED_COLUMNS = {
    "topic_category",
    "industry",
    "target_audience",
    "seo_keywords",
    "source_content",
    "article_title",
    "article_body",
    "desired_length",
    "source_dataset",
}

MOJIBAKE_MARKERS = (
    "\ufffd",
    "โ€",
    "ยท",
    "ย’",
    "ย“",
    "ย”",
    "â€",
    "â€™",
    "â€œ",
    "â€�",
    "Ã",
)

BAD_TITLE_PATTERNS = (
    r"\bdraft\b",
    r"\buntitled\b",
    r"\btest\b",
    r"\bregister\b",
    r"\binvitation\b",
    r"\bwebinar\b",
)

BOILERPLATE_SECTION_PATTERNS = (
    r"^\s*bibliography\s*$",
    r"^\s*references\s*$",
    r"^\s*works cited\s*$",
    r"^\s*learn more:?\s*$",
    r"^\s*author:\s+",
    r"^\s*follow me\b",
    r"^\s*to be continued",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean bootstrapped article-training CSV before chat-format conversion."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Bootstrapped source CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Cleaned output CSV.")
    parser.add_argument(
        "--rejected-output",
        type=Path,
        help="Optional CSV path for rejected rows with rejection reasons.",
    )
    parser.add_argument("--min-source-chars", type=int, default=250)
    parser.add_argument("--min-article-chars", type=int, default=800)
    parser.add_argument("--max-article-chars", type=int, default=8000)
    parser.add_argument("--max-urls", type=int, default=4)
    parser.add_argument(
        "--allow-non-english",
        action="store_true",
        help="Keep rows that do not look mostly English.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic shuffle.")
    parser.add_argument("--no-shuffle", action="store_true", help="Preserve input row order.")
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def count_urls(text: str) -> int:
    return len(re.findall(r"https?://|www\.", text or "", flags=re.IGNORECASE))


def has_mojibake(text: str) -> bool:
    return any(marker in (text or "") for marker in MOJIBAKE_MARKERS)


def looks_english(text: str) -> bool:
    letters = re.findall(r"[A-Za-z]", text or "")
    non_ascii_letters = re.findall(r"[^\W\d_]", text or "", flags=re.UNICODE)
    if not non_ascii_letters:
        return False
    return len(letters) / len(non_ascii_letters) >= 0.85


def is_bad_title(title: str) -> bool:
    normalized = normalize_whitespace(title).lower()
    return any(re.search(pattern, normalized) for pattern in BAD_TITLE_PATTERNS)


def strip_url_lines(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if re.fullmatch(r"(https?://|www\.).+", stripped, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines)


def strip_boilerplate_sections(text: str) -> str:
    kept_lines: list[str] = []
    for line in (text or "").splitlines():
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in BOILERPLATE_SECTION_PATTERNS):
            break
        kept_lines.append(line)
    return "\n".join(kept_lines)


def clean_text(value: Any) -> str:
    text = strip_basic_html(str(value or ""))
    text = strip_url_lines(text)
    text = strip_boilerplate_sections(text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    return normalize_whitespace(text)


def source_is_too_close_to_article(source_content: str, article_body: str) -> bool:
    source = normalize_whitespace(source_content).lower()
    article = normalize_whitespace(article_body).lower()
    if not source or not article:
        return True
    if source == article:
        return True
    return len(source) / max(len(article), 1) > 0.85


def clean_row(row: pd.Series, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    title = clean_text(row.get("article_title", ""))
    source_content = clean_text(row.get("source_content", ""))
    article_body = clean_text(row.get("article_body", ""))
    combined_text = f"{title}\n{source_content}\n{article_body}"

    if not title:
        return None, "missing_title"
    if is_bad_title(title):
        return None, "bad_title"
    if has_mojibake(combined_text):
        return None, "mojibake"
    if not args.allow_non_english and not looks_english(combined_text):
        return None, "non_english"
    if count_urls(combined_text) > args.max_urls:
        return None, "too_many_urls"
    if len(source_content) < args.min_source_chars:
        return None, "short_source"
    if len(article_body) < args.min_article_chars:
        return None, "short_article"
    if len(article_body) > args.max_article_chars:
        return None, "long_article"
    if source_is_too_close_to_article(source_content, article_body):
        return None, "source_too_close_to_article"

    cleaned = row.to_dict()
    cleaned["article_title"] = title
    cleaned["source_content"] = source_content
    cleaned["article_body"] = article_body
    cleaned["topic_category"] = normalize_whitespace(str(cleaned.get("topic_category", "")))
    cleaned["industry"] = normalize_whitespace(str(cleaned.get("industry", "")))
    cleaned["target_audience"] = normalize_whitespace(str(cleaned.get("target_audience", "")))
    cleaned["desired_length"] = normalize_whitespace(str(cleaned.get("desired_length", "900-1200 words")))
    cleaned["source_dataset"] = normalize_whitespace(str(cleaned.get("source_dataset", "")))
    return cleaned, None


def print_distribution(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns or df.empty:
        return
    print(f"{column}:")
    for value, count in df[column].value_counts().items():
        print(f"- {value}: {count}")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.input}. Run training/bootstrap_hf_dataset.py first.")

    df = pd.read_csv(args.input)
    validate_columns(df)

    cleaned_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    for _, row in df.iterrows():
        cleaned, reason = clean_row(row, args)
        if cleaned is None:
            rejected = row.to_dict()
            rejected["rejection_reason"] = reason
            rejected_rows.append(rejected)
            reasons[str(reason)] += 1
            continue
        cleaned_rows.append(cleaned)

    cleaned_df = pd.DataFrame(cleaned_rows, columns=list(df.columns))
    cleaned_df = cleaned_df.drop_duplicates(subset=["article_title"]).reset_index(drop=True)
    if not args.no_shuffle and not cleaned_df.empty:
        cleaned_df = cleaned_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(args.output, index=False)

    if args.rejected_output:
        args.rejected_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rejected_rows).to_csv(args.rejected_output, index=False)

    print(f"Read {len(df)} rows from {args.input}")
    print(f"Saved {len(cleaned_df)} cleaned rows to {args.output}")
    print(f"Rejected {len(rejected_rows)} rows")
    for reason, count in reasons.most_common():
        print(f"- {reason}: {count}")
    print_distribution(cleaned_df, "source_dataset")
    print_distribution(cleaned_df, "topic_category")
    print_distribution(cleaned_df, "industry")


if __name__ == "__main__":
    main()
