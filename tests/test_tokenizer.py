"""Tests for the tokenizer module."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from hysearch.core import tokenizer
from hysearch.core.config import TOKENIZER_TRIGRAM
from hysearch.core.tokenizer import (
    _get_sudachi_tokenizer,
    build_fts_query,
    detect_default_tokenizer,
    format_fts_content,
    is_sudachi_dict_available,
    resolve_tokenizer,
    tokenize_with_sudachi,
)


def test_is_sudachi_dict_available() -> None:
    """Test checking Sudachi availability via importlib.util.find_spec."""
    with patch("importlib.util.find_spec") as mock_find:
        mock_find.side_effect = lambda name: (
            MagicMock() if name in ("sudachipy", "sudachidict_core") else None
        )
        assert is_sudachi_dict_available("core") is True
        assert is_sudachi_dict_available("full") is False


def test_detect_default_tokenizer_priority() -> None:
    """Test auto-detection priority (full -> core -> small -> trigram)."""
    with patch("hysearch.core.tokenizer.is_sudachi_dict_available") as mock_av:
        mock_av.side_effect = lambda dt: dt in ("core", "small")
        assert detect_default_tokenizer() == "sudachi-core"

        mock_av.side_effect = lambda dt: False
        assert detect_default_tokenizer() == TOKENIZER_TRIGRAM


def test_resolve_tokenizer() -> None:
    """Test resolving explicit and auto tokenizers and handling errors."""
    # None -> detect_default_tokenizer
    with patch(
        "hysearch.core.tokenizer.detect_default_tokenizer",
        return_value="trigram",
    ):
        assert resolve_tokenizer(None) == "trigram"

    # Explicit trigram
    assert resolve_tokenizer("trigram") == "trigram"
    assert resolve_tokenizer("  TRIGRAM ") == "trigram"

    # Invalid tokenizer name
    with pytest.raises(ValueError, match="Unsupported tokenizer"):
        resolve_tokenizer("mecab")

    # 'sudachi' alias when available vs unavailable
    with patch("hysearch.core.tokenizer.is_sudachi_dict_available") as mock_av:
        mock_av.side_effect = lambda dt: dt == "small"
        assert resolve_tokenizer("sudachi") == "sudachi-small"

        mock_av.side_effect = lambda dt: False
        with pytest.raises(
            RuntimeError, match="Sudachi tokenizer is not installed"
        ):
            resolve_tokenizer("sudachi")

    # Explicit 'sudachi-full' when available vs unavailable
    with patch("hysearch.core.tokenizer.is_sudachi_dict_available") as mock_av:
        mock_av.side_effect = lambda dt: dt == "full"
        assert resolve_tokenizer("sudachi-full") == "sudachi-full"

        mock_av.side_effect = lambda dt: False
        with pytest.raises(
            RuntimeError, match="Tokenizer 'sudachi-full' is not available"
        ):
            resolve_tokenizer("sudachi-full")


def test_get_sudachi_tokenizer_and_tokenize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test lazy Sudachi initialization, caching, and SplitMode.A."""
    tokenizer._SUDACHI_CACHE.clear()

    # Create a fake sudachipy module
    fake_sudachipy = types.ModuleType("sudachipy")

    class FakeSplitMode:
        A = "A"

    class FakeMorpheme:
        def __init__(self, norm: str) -> None:
            self._norm = norm

        def normalized_form(self) -> str:
            return self._norm

    mock_tok_instance = MagicMock()
    mock_tok_instance.tokenize.return_value = [
        FakeMorpheme("京都"),
        FakeMorpheme("。"),
        FakeMorpheme("検索"),
        FakeMorpheme("システム"),
    ]

    mock_dict_cls = MagicMock()
    mock_dict_cls.return_value.create.return_value = mock_tok_instance

    fake_sudachipy.Dictionary = mock_dict_cls  # type: ignore[attr-defined]
    fake_sudachipy.SplitMode = FakeSplitMode  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "sudachipy", fake_sudachipy)

    # First call initializes and caches
    tok1 = _get_sudachi_tokenizer("core")
    tok2 = _get_sudachi_tokenizer("core")
    assert tok1 is tok2
    mock_dict_cls.assert_called_once_with(dict="core")

    # Empty or whitespace text returns empty list
    assert tokenize_with_sudachi("", dict_type="core") == []
    assert tokenize_with_sudachi("   ", dict_type="core") == []

    # Tokenize normal text (punctuation '。' should be filtered out)
    tokens = tokenize_with_sudachi("京都。検索システム", dict_type="core")
    assert tokens == ["京都", "検索", "システム"]
    mock_tok_instance.tokenize.assert_called_once_with(
        "京都。検索システム", "A"
    )

    tokenizer._SUDACHI_CACHE.clear()


def test_format_fts_content() -> None:
    """Test formatting chunk text for trigram vs Sudachi."""
    assert format_fts_content("Hello World", "trigram") == "Hello World"

    with patch(
        "hysearch.core.tokenizer.tokenize_with_sudachi",
        return_value=["東京", "タワー"],
    ) as mock_tok:
        res = format_fts_content("東京タワー", "sudachi-core")
        assert res == "東京 タワー"
        mock_tok.assert_called_once_with("東京タワー", dict_type="core")


def test_build_fts_query() -> None:
    """Test building FTS5 MATCH queries for trigram and Sudachi."""
    # Trigram mode
    assert build_fts_query("ab", "trigram") is None
    assert build_fts_query("a b", "trigram") is None
    assert build_fts_query('foo "bar"', "trigram") == '"foo ""bar"""'

    # Sudachi mode
    with patch(
        "hysearch.core.tokenizer.tokenize_with_sudachi",
        return_value=["猫", "走る"],
    ) as mock_tok:
        q = build_fts_query("猫が走る", "sudachi-full")
        assert q == '"猫" "走る"'
        mock_tok.assert_called_once_with("猫が走る", dict_type="full")

    # Sudachi mode when only symbols/empty tokens are given
    with patch(
        "hysearch.core.tokenizer.tokenize_with_sudachi",
        return_value=[],
    ):
        assert build_fts_query("！？", "sudachi-core") is None
