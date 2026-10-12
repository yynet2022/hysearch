"""Document chunking and content hashing for hysearch.

This module provides text segmentation prioritizing Markdown headings
and paragraph boundaries, along with SHA-256 hash calculation for change
detection.
"""

import hashlib
import re
from pathlib import Path

from hysearch.core.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from hysearch.core.models import Chunk

PathLike = str | Path

# Markdown 見出しおよび段落境界の正規表現パターン
HEADING_PATTERN = re.compile(r"\n(?=#{1,6}\s)")
PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")


def compute_file_hash(file_path: PathLike) -> str:
    """Calculate the SHA-256 hexadecimal digest of a file.

    Args:
        file_path: Path to the target file.

    Returns:
        str: SHA-256 hexadecimal digest of the file content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = Path(file_path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_text_hash(text: str) -> str:
    """Calculate the SHA-256 hexadecimal digest of a string.

    Args:
        text: String content to hash.

    Returns:
        str: SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _find_split_point(
    text: str,
    search_start: int,
    search_end: int,
) -> int:
    r"""Find a natural boundary to split text within the given search range.

    Prioritizes:
    1. Markdown heading (newline before '# ')
    2. Paragraph boundary ('\\n\\n')
    3. Single newline ('\\n')
    4. Sentence ending ('。', '.', '!', '?')
    5. Whitespace (' ')

    Args:
        text: Full text being chunked.
        search_start: Starting character index of search window.
        search_end: Ending character index of search window.

    Returns:
        int: Split index within the search window, or search_end if none found.
    """
    sub = text[search_start:search_end]
    if not sub:
        return search_end

    # 1. Markdown 見出しの前
    matches = list(HEADING_PATTERN.finditer(sub))
    if matches:
        return search_start + matches[-1].start() + 1

    # 2. 段落区切り (\n\n)
    matches = list(PARAGRAPH_PATTERN.finditer(sub))
    if matches:
        return search_start + matches[-1].end()

    # 3. 単一の改行
    idx = sub.rfind("\n")
    if idx != -1:
        return search_start + idx + 1

    # 4. 文末記号 (日本語句点・ピリオド等)
    for p in ("。", "．", ".", "!", "?", "！", "？"):
        idx = sub.rfind(p)
        if idx != -1:
            return search_start + idx + len(p)

    # 5. 空白文字
    idx = sub.rfind(" ")
    if idx != -1:
        return search_start + idx + 1

    return search_end


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split text into overlapping chunks respecting document structure.

    Args:
        text: Raw document text to chunk.
        chunk_size: Target character count per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        list[Chunk]: List of extracted Chunk objects.

    Raises:
        ValueError: If chunk_size <= 0 or overlap < 0 or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")

    total_len = len(text)
    if total_len == 0:
        return []

    if total_len <= chunk_size:
        return [
            Chunk(
                chunk_index=0,
                content=text,
                start_char=0,
                end_char=total_len,
            )
        ]

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < total_len:
        ideal_end = min(start + chunk_size, total_len)

        if ideal_end == total_len:
            end = total_len
        else:
            # オーバーラップを考慮した探索開始位置
            search_start = max(start + 1, ideal_end - overlap)
            end = _find_split_point(text, search_start, ideal_end)
            if end <= start:
                end = ideal_end

        content = text[start:end]
        chunks.append(
            Chunk(
                chunk_index=chunk_index,
                content=content,
                start_char=start,
                end_char=end,
            )
        )
        chunk_index += 1

        if end >= total_len:
            break

        # 次の開始位置を算出
        next_start = max(start + 1, end - overlap)
        start = next_start

    return chunks
