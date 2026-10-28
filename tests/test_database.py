"""Tests for database initialization, FTS5 triggers, and sqlite-vec."""

import sqlite3
from pathlib import Path

import pytest
import sqlite_vec

from hysearch.core.config import (
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_MODEL,
    EMBED_MODEL_MPNET,
    TOKENIZER_TRIGRAM,
)
from hysearch.core.database import (
    SCHEMA_VERSION,
    create_connection,
    get_connection,
    get_db_embed_dim,
    get_db_embed_model,
    get_db_status,
    get_db_tokenizer,
    get_schema_version,
    get_stored_embed_dim,
    get_stored_embed_model,
    get_stored_tokenizer,
    init_db,
)


def test_init_db_creates_tables_and_triggers(
    memory_db: sqlite3.Connection,
) -> None:
    """Verify that all required tables and triggers are created."""
    cursor = memory_db.cursor()

    # テーブルの存在確認
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view');"
    )
    names = {row[0] for row in cursor.fetchall()}
    assert "files" in names
    assert "chunks" in names
    assert "fts_chunks" in names
    assert "vec_chunks" in names


def test_fts5_trigram_insert_and_search(
    memory_db: sqlite3.Connection,
) -> None:
    """Verify FTS5 trigram indexing and trigger synchronization."""
    with memory_db:
        memory_db.execute(
            "INSERT INTO files (path, hash, chunk_count) VALUES (?, ?, ?);",
            ("test.md", "hash123", 1),
        )
        file_id = memory_db.execute("SELECT last_insert_rowid();").fetchone()[
            0
        ]
        memory_db.execute(
            """
            INSERT INTO chunks (file_id, chunk_index, content, start_char,
                                end_char)
            VALUES (?, ?, ?, ?, ?);
            """,
            (file_id, 0, "これはハイブリッド検索のテスト文書です。", 0, 20),
        )

    # FTS5 から検索できるか確認 (trigram)
    cursor = memory_db.cursor()
    cursor.execute(
        "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ?;",
        ("ハイブリッド",),
    )
    rows = cursor.fetchall()
    assert len(rows) == 1


def test_sqlite_vec_insert_and_query(
    memory_db: sqlite3.Connection,
) -> None:
    """Verify inserting and querying vectors using sqlite-vec."""
    dummy_vec = [0.1] * DEFAULT_EMBED_DIM
    serialized = sqlite_vec.serialize_float32(dummy_vec)

    with memory_db:
        memory_db.execute(
            "INSERT INTO vec_chunks (id, embedding) VALUES (?, ?);",
            (1, serialized),
        )

    cursor = memory_db.cursor()
    cursor.execute(
        """
        SELECT id, distance
        FROM vec_chunks
        WHERE embedding MATCH ? AND k = 1
        ORDER BY distance;
        """,
        (serialized,),
    )
    rows = cursor.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1] < 1e-5


def test_cascade_delete_and_triggers(
    memory_db: sqlite3.Connection,
) -> None:
    """Verify that deleting a file cascades to chunks, FTS5, and vectors."""
    dummy_vec = [0.2] * DEFAULT_EMBED_DIM
    serialized = sqlite_vec.serialize_float32(dummy_vec)

    with memory_db:
        memory_db.execute(
            "INSERT INTO files (path, hash, chunk_count) VALUES (?, ?, ?);",
            ("doc.md", "hashabc", 1),
        )
        file_id = memory_db.execute("SELECT last_insert_rowid();").fetchone()[
            0
        ]
        memory_db.execute(
            """
            INSERT INTO chunks (file_id, chunk_index, content, start_char,
                                end_char)
            VALUES (?, ?, ?, ?, ?);
            """,
            (file_id, 0, "削除テスト用のチャンクです。", 0, 12),
        )
        chunk_id = memory_db.execute("SELECT last_insert_rowid();").fetchone()[
            0
        ]
        memory_db.execute(
            "INSERT INTO vec_chunks (id, embedding) VALUES (?, ?);",
            (chunk_id, serialized),
        )

    # 削除前の確認
    assert memory_db.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0] == 1
    assert (
        memory_db.execute("SELECT COUNT(*) FROM fts_chunks;").fetchone()[0]
        == 1
    )
    assert (
        memory_db.execute("SELECT COUNT(*) FROM vec_chunks;").fetchone()[0]
        == 1
    )

    # files から削除
    with memory_db:
        memory_db.execute("DELETE FROM files WHERE id = ?;", (file_id,))

    # カスケードおよびトリガーによる削除を確認
    assert memory_db.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0] == 0
    assert (
        memory_db.execute("SELECT COUNT(*) FROM fts_chunks;").fetchone()[0]
        == 0
    )
    assert (
        memory_db.execute("SELECT COUNT(*) FROM vec_chunks;").fetchone()[0]
        == 0
    )


def test_get_connection_context_manager(tmp_path: Path) -> None:
    """Verify get_connection opens, operates, and closes properly."""
    db_file = tmp_path / "test.db"
    with get_connection(db_file) as conn:
        init_db(conn)
        assert conn.execute("SELECT 1;").fetchone()[0] == 1


def test_schema_version_and_migration() -> None:
    """Verify schema version retrieval and migration from v0."""
    conn = create_connection(":memory:")
    # まだバージョン設定されていない初期状態
    assert get_schema_version(conn) == 0

    # v0 相当のテーブルを作成 (mtime なし)
    conn.execute(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            hash TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    # マイグレーション実行
    init_db(conn)

    assert get_schema_version(conn) == SCHEMA_VERSION

    # mtime カラムが存在することを確認
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(files);")
    columns = {row[1] for row in cursor.fetchall()}
    assert "mtime" in columns
    conn.close()


def test_tokenizer_meta_and_reinit() -> None:
    """Verify tokenizer metadata storage, legacy detection, and reinit."""
    conn = create_connection(":memory:")
    # Empty DB has no stored tokenizer; get_db_tokenizer defaults to trigram
    assert get_stored_tokenizer(conn) is None
    assert get_db_tokenizer(conn) == TOKENIZER_TRIGRAM

    # Legacy DB (has fts_chunks but no meta table) -> detected as trigram
    conn.execute("CREATE VIRTUAL TABLE fts_chunks USING fts5(content);")
    assert get_stored_tokenizer(conn) == TOKENIZER_TRIGRAM

    # Initialize with trigram (migrates legacy DB and records meta)
    init_db(conn)
    assert get_stored_tokenizer(conn) == TOKENIZER_TRIGRAM

    # Attempting to change tokenizer without force raises ValueError
    with pytest.raises(ValueError, match="already indexed with tokenizer"):
        init_db(conn, tokenizer="sudachi-core", force_reinit_fts=False)

    # Re-initializing with force_reinit_fts=True switches to unicode61 FTS
    init_db(conn, tokenizer="sudachi-core", force_reinit_fts=True)
    assert get_stored_tokenizer(conn) == "sudachi-core"
    assert get_db_tokenizer(conn) == "sudachi-core"

    # Calling init_db again with None or matching tokenizer preserves it
    init_db(conn, tokenizer=None)
    assert get_stored_tokenizer(conn) == "sudachi-core"
    conn.close()


def test_embed_model_meta_and_reinit() -> None:
    """Verify embedding model metadata storage, legacy fallback, and reinit."""
    conn = create_connection(":memory:")
    # Empty DB has no stored model/dim; getters return defaults
    assert get_stored_embed_model(conn) is None
    assert get_stored_embed_dim(conn) is None
    assert get_db_embed_model(conn) == DEFAULT_EMBED_MODEL
    assert get_db_embed_dim(conn) == DEFAULT_EMBED_DIM

    # Legacy DB (has vec_chunks but no meta table) -> detected as default
    conn.execute(
        "CREATE VIRTUAL TABLE vec_chunks USING vec0("
        "id INTEGER PRIMARY KEY, embedding float[384]);"
    )
    assert get_stored_embed_model(conn) == DEFAULT_EMBED_MODEL
    assert get_stored_embed_dim(conn) == DEFAULT_EMBED_DIM

    # Initialize with default (migrates legacy DB and records meta)
    init_db(conn)
    assert get_stored_embed_model(conn) == DEFAULT_EMBED_MODEL
    assert get_stored_embed_dim(conn) == DEFAULT_EMBED_DIM

    # Attempting to change embedding model without force raises ValueError
    with pytest.raises(
        ValueError, match="already indexed with embedding model"
    ):
        init_db(conn, embed_model=EMBED_MODEL_MPNET, force_reinit_vec=False)

    # Re-initializing with force_reinit_vec=True switches model and dim (768)
    init_db(conn, embed_model=EMBED_MODEL_MPNET, force_reinit_vec=True)
    assert get_stored_embed_model(conn) == EMBED_MODEL_MPNET
    assert get_stored_embed_dim(conn) == 768
    assert get_db_embed_model(conn) == EMBED_MODEL_MPNET
    assert get_db_embed_dim(conn) == 768

    # Calling init_db again with None preserves stored model and dimension
    init_db(conn, embed_model=None)
    assert get_stored_embed_model(conn) == EMBED_MODEL_MPNET
    assert get_stored_embed_dim(conn) == 768
    conn.close()


def test_read_only_connection_and_status(tmp_path: Path) -> None:
    """Verify read_only connection and get_db_status on chmod 444 database."""
    db_file = tmp_path / "ro_test.db"
    with get_connection(db_file) as conn:
        init_db(conn)

    # Set database file to read-only (chmod 444)
    db_file.chmod(0o444)
    try:
        status_data = get_db_status(db_file)
        assert status_data["Files Indexed"] == 0
        assert status_data["Total Chunks"] == 0

        with (
            get_connection(db_file, read_only=True) as ro_conn,
            pytest.raises(sqlite3.OperationalError),
        ):
            ro_conn.execute("INSERT INTO meta (key, value) VALUES ('k', 'v');")

        with (
            get_connection(":memory:", read_only=True) as mem_ro,
            pytest.raises(sqlite3.OperationalError),
        ):
            mem_ro.execute("CREATE TABLE t (id INTEGER);")
    finally:
        db_file.chmod(0o666)
