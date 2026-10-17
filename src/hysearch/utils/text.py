"""Text processing, cleaning, and snippet extraction utilities for hysearch.

This module provides text normalization and keyword-centered snippet
generation for search results display.
"""

import re
from datetime import datetime


def parse_datetime_str(val: str, param_name: str = "date") -> datetime:
    """Parse a date or datetime string into a datetime object.

    Args:
        val: Input date string.
        param_name: Parameter name for error reporting.

    Returns:
        datetime: Parsed datetime instance.

    Raises:
        ValueError: If the date string cannot be parsed.
    """
    clean_val = val.strip()
    try:
        return datetime.fromisoformat(clean_val)
    except ValueError:
        pass

    for fmt in (
        "%Y/%m/%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m",
        "%Y-%m",
        "%Y",
    ):
        try:
            return datetime.strptime(clean_val, fmt)
        except ValueError:
            pass

    raise ValueError(
        f"Invalid date format for {param_name}: '{val}'. "
        "Use YYYY-MM-DD or ISO 8601 format."
    )


def clean_text(text: str) -> str:
    """Normalize whitespace and remove unprintable control characters.

    Args:
        text: Input string to clean.

    Returns:
        str: Cleaned string with collapsed whitespace.
    """
    if not text:
        return ""
    # 連続する空白・改行を単一スペースに正規化
    normalized = re.sub(r"[\r\n\t]+", " ", text)
    return re.sub(r" {2,}", " ", normalized).strip()


def extract_snippet(
    content: str, query: str = "", max_length: int = 150
) -> str:
    """Extract a concise text snippet centered around matching query terms.

    Args:
        content: Raw document or chunk text.
        query: Search query used to locate relevant excerpt.
        max_length: Maximum target length of the snippet.

    Returns:
        str: Truncated excerpt with ellipsis indicators if trimmed.
    """
    if not content:
        return ""

    cleaned = clean_text(content)
    if len(cleaned) <= max_length:
        return cleaned

    # クエリからキーワード候補（スペース区切りの単語や3文字以上の語）を抽出
    match_index = -1
    if query:
        terms = [t for t in re.split(r"\s+", query.strip()) if t]
        for term in terms:
            idx = cleaned.lower().find(term.lower())
            if idx != -1:
                match_index = idx
                break

    if match_index == -1:
        # キーワードが見つからない場合は先頭から切り出し
        return cleaned[: max_length - 3].rstrip() + "..."

    # マッチ位置を中心にウィンドウを配置
    half_window = (max_length - 6) // 2
    start = max(0, match_index - half_window)
    end = min(len(cleaned), start + max_length - 6)

    # 終端がはみ出さないように start を再調整
    if end == len(cleaned):
        start = max(0, end - (max_length - 6))

    snippet_parts: list[str] = []
    if start > 0:
        snippet_parts.append("...")
    snippet_parts.append(cleaned[start:end])
    if end < len(cleaned):
        snippet_parts.append("...")

    return "".join(snippet_parts)
