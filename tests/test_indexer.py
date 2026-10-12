"""Tests for indexing pipeline and incremental change detection."""

import sqlite3
from pathlib import Path

from hysearch.core.embedder import DummyEmbedder
from hysearch.core.indexer import (
    get_stored_file_hashes,
    index_directory,
    scan_directory,
)


def test_scan_directory_filters_and_recursion(tmp_path: Path) -> None:
    """Verify scan_directory only collects supported file extensions."""
    (tmp_path / "doc1.md").write_text("Markdown 1", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Text note", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()
    (sub_dir / "guide.markdown").write_text("Sub guide", encoding="utf-8")
    (sub_dir / "ignored.py").write_text("print()", encoding="utf-8")

    files = scan_directory(tmp_path)
    names = [f.name for f in files]
    assert "doc1.md" in names
    assert "notes.txt" in names
    assert "guide.markdown" in names
    assert "image.png" not in names
    assert "ignored.py" not in names


def test_index_directory_incremental_and_force(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify incremental indexing skips unchanged files and force works."""
    file1 = tmp_path / "doc1.md"
    file2 = tmp_path / "doc2.txt"
    file1.write_text("First document content.", encoding="utf-8")
    file2.write_text("Second document content.", encoding="utf-8")

    # 初回インデックス
    stats1 = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        chunk_size=100,
        overlap=20,
    )
    assert stats1.files_scanned == 2
    assert stats1.files_indexed == 2
    assert stats1.files_skipped == 0
    assert stats1.chunks_created == 2

    # 変更なしでの2回目インデックス（スキップされること）
    stats2 = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        chunk_size=100,
        overlap=20,
    )
    assert stats2.files_scanned == 2
    assert stats2.files_indexed == 0
    assert stats2.files_skipped == 2
    assert stats2.chunks_created == 0

    # 1ファイルのみ更新
    file1.write_text("Updated first document content.", encoding="utf-8")
    stats3 = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        chunk_size=100,
        overlap=20,
    )
    assert stats3.files_scanned == 2
    assert stats3.files_indexed == 1
    assert stats3.files_skipped == 1

    # force=True で全再構築
    stats4 = index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
        chunk_size=100,
        overlap=20,
        force=True,
    )
    assert stats4.files_scanned == 2
    assert stats4.files_indexed == 2
    assert stats4.files_skipped == 0


def test_get_stored_file_hashes(
    memory_db: sqlite3.Connection,
    tmp_path: Path,
    dummy_embedder: DummyEmbedder,
) -> None:
    """Verify retrieving hash map from database."""
    file1 = tmp_path / "test.md"
    file1.write_text("Sample", encoding="utf-8")

    index_directory(
        conn=memory_db,
        target_dir=tmp_path,
        embedder=dummy_embedder,
    )
    hashes = get_stored_file_hashes(memory_db)
    assert len(hashes) == 1
    assert str(file1.resolve()) in hashes
