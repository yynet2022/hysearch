"""Tokenizer resolution and text segmentation for FTS5 indexing and search.

This module manages tokenizer selection (SudachiPy full/core/small or SQLite
trigram), lazy initialization of Sudachi dictionaries, morphological
segmentation with SplitMode.A and normalization, and FTS5 query formatting.
"""

import importlib.util
import re
from typing import Any

from hysearch.core.config import (
    SUDACHI_DICT_PRIORITY,
    TOKENIZER_SUDACHI_AUTO,
    TOKENIZER_TRIGRAM,
    VALID_TOKENIZERS,
)

# Unicode の文字 (Letter) および数字 (Number) のみを抽出する正規表現
_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

# Sudachi Dictionary インスタンスのキャッシュ (dict_type -> Tokenizer)
_SUDACHI_CACHE: dict[str, Any] = {}


def is_sudachi_dict_available(dict_type: str) -> bool:
    """Check whether sudachipy and the specified dictionary are installed.

    Args:
        dict_type: Sudachi dictionary type ('full', 'core', or 'small').

    Returns:
        bool: True if both sudachipy and sudachidict_<dict_type> exist.
    """
    has_sudachi = importlib.util.find_spec("sudachipy") is not None
    has_dict = importlib.util.find_spec(f"sudachidict_{dict_type}") is not None
    return has_sudachi and has_dict


def detect_default_tokenizer() -> str:
    """Detect the best available tokenizer in priority order.

    Priority: sudachi-full -> sudachi-core -> sudachi-small -> trigram.

    Returns:
        str: Name of the selected tokenizer.
    """
    for dict_type in SUDACHI_DICT_PRIORITY:
        if is_sudachi_dict_available(dict_type):
            return f"sudachi-{dict_type}"
    return TOKENIZER_TRIGRAM


def resolve_tokenizer(requested: str | None = None) -> str:
    """Resolve and validate the tokenizer to use.

    Args:
        requested: Optional user-specified tokenizer name. If None,
            automatically detects the highest-priority available tokenizer.

    Returns:
        str: Concrete tokenizer name ('trigram', 'sudachi-full',
            'sudachi-core', or 'sudachi-small').

    Raises:
        ValueError: If the requested tokenizer name is unknown.
        RuntimeError: If the requested Sudachi tokenizer is not installed.
    """
    if requested is None:
        return detect_default_tokenizer()

    norm = requested.strip().lower()
    if norm not in VALID_TOKENIZERS:
        valid_str = ", ".join(VALID_TOKENIZERS)
        raise ValueError(
            f"Unsupported tokenizer '{requested}'. Choose from: {valid_str}."
        )

    if norm == TOKENIZER_TRIGRAM:
        return TOKENIZER_TRIGRAM

    if norm == TOKENIZER_SUDACHI_AUTO:
        for dict_type in SUDACHI_DICT_PRIORITY:
            if is_sudachi_dict_available(dict_type):
                return f"sudachi-{dict_type}"
        raise RuntimeError(
            "Sudachi tokenizer is not installed. "
            'Install with: pip install "hysearch[sudachi]"'
        )

    dict_type = norm.split("-", 1)[1]
    if not is_sudachi_dict_available(dict_type):
        raise RuntimeError(
            f"Tokenizer '{norm}' is not available. "
            f'Install with: pip install "hysearch[{norm}]"'
        )
    return norm


def _get_sudachi_tokenizer(dict_type: str) -> Any:
    """Lazily initialize and return a Sudachi tokenizer instance.

    Args:
        dict_type: Dictionary type ('full', 'core', or 'small').

    Returns:
        Any: SudachiPy Tokenizer instance.
    """
    if dict_type not in _SUDACHI_CACHE:
        from sudachipy import Dictionary

        _SUDACHI_CACHE[dict_type] = Dictionary(dict=dict_type).create()
    return _SUDACHI_CACHE[dict_type]


def tokenize_with_sudachi(text: str, dict_type: str = "core") -> list[str]:
    """Segment and normalize text using SudachiPy with SplitMode.A.

    Args:
        text: Input text to segment.
        dict_type: Sudachi dictionary type ('full', 'core', or 'small').

    Returns:
        list[str]: List of normalized word tokens excluding punctuation.
    """
    if not text or not text.strip():
        return []

    from sudachipy import SplitMode

    tokenizer_obj = _get_sudachi_tokenizer(dict_type)
    morphemes = tokenizer_obj.tokenize(text, SplitMode.A)

    tokens: list[str] = []
    for m in morphemes:
        norm = m.normalized_form()
        tokens.extend(_WORD_PATTERN.findall(norm))
    return tokens


def format_fts_content(
    text: str, tokenizer_name: str = TOKENIZER_TRIGRAM
) -> str:
    """Format chunk text for insertion into the FTS5 virtual table.

    Args:
        text: Raw chunk text.
        tokenizer_name: Active tokenizer name.

    Returns:
        str: Raw text for trigram, or space-separated tokens for Sudachi.
    """
    if tokenizer_name == TOKENIZER_TRIGRAM:
        return text

    dict_type = tokenizer_name.split("-", 1)[1]
    return " ".join(tokenize_with_sudachi(text, dict_type=dict_type))


def build_fts_query(
    query: str, tokenizer_name: str = TOKENIZER_TRIGRAM
) -> str | None:
    """Build a safe SQLite FTS5 MATCH query string for the given tokenizer.

    Args:
        query: Raw user search query.
        tokenizer_name: Active tokenizer name stored in the database.

    Returns:
        str | None: FTS5 MATCH expression, or None if query cannot be matched
            by FTS5 (e.g., < 3 chars in trigram mode or empty tokens).
    """
    stripped = query.strip()
    if tokenizer_name == TOKENIZER_TRIGRAM:
        compact = "".join(stripped.split())
        if len(compact) < 3:
            return None
        escaped = stripped.replace('"', '""')
        return f'"{escaped}"'

    dict_type = tokenizer_name.split("-", 1)[1]
    tokens = tokenize_with_sudachi(stripped, dict_type=dict_type)
    if not tokens:
        return None
    return " ".join(f'"{t}"' for t in tokens)
