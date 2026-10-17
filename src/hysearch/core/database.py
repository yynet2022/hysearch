"""SQLite database connection and schema initialization for hysearch.

This module provides functions to create database connections, load the
sqlite-vec extension, initialize schema tables (metadata, chunks, FTS5,
and vector index), and reset the database.
"""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from hysearch.core.config import DEFAULT_EMBED_DIM, TOKENIZER_TRIGRAM

PathLike = str | Path
SCHEMA_VERSION: int = 2

# 基本テーブル初期化用 SQL スクリプト
INIT_BASE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    mtime REAL NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
    UNIQUE(file_id, chunk_index)
);
"""

# Trigram トークナイザー用 FTS5 およびトリガー定義
INIT_TRIGRAM_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, content)
        VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, content)
        VALUES('delete', old.id, old.content);
    INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
END;
"""

# Sudachi (unicode61) トークナイザー用 FTS5 およびトリガー定義
INIT_UNICODE61_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    content,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM fts_chunks WHERE rowid = old.id;
END;
"""

DROP_FTS_SQL = """
DROP TRIGGER IF EXISTS chunks_ai;
DROP TRIGGER IF EXISTS chunks_ad;
DROP TRIGGER IF EXISTS chunks_au;
DROP TABLE IF EXISTS fts_chunks;
"""


def create_connection(db_path: PathLike) -> sqlite3.Connection:
    """Create a new SQLite connection with foreign keys and sqlite-vec loaded.

    Args:
        db_path: Path to the SQLite database file, or ':memory:'.

    Returns:
        sqlite3.Connection: Configured connection with vector extension loaded.

    Raises:
        sqlite3.OperationalError: If extension loading fails or DB is locked.
    """
    path_str = str(db_path)
    conn = sqlite3.connect(path_str)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


@contextmanager
def get_connection(
    db_path: PathLike,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connection to ensure proper closure.

    Args:
        db_path: Path to the SQLite database file or ':memory:'.

    Yields:
        sqlite3.Connection: Open database connection.
    """
    conn = create_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Retrieve the current database schema version.

    Args:
        conn: Open SQLite connection.

    Returns:
        int: Current user_version PRAGMA value.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA user_version;")
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def get_stored_tokenizer(conn: sqlite3.Connection) -> str | None:
    """Retrieve the tokenizer name stored in the database metadata.

    Args:
        conn: Open SQLite connection.

    Returns:
        str | None: Stored tokenizer name, or None if not yet initialized.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='meta';"
    )
    if cursor.fetchone() is not None:
        cursor.execute("SELECT value FROM meta WHERE key = 'tokenizer';")
        row = cursor.fetchone()
        if row is not None:
            return str(row[0])

    # Legacy DB (v0/v1) 判定: fts_chunks が既に存在する場合は trigram とみなす
    cursor.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='fts_chunks';"
    )
    if cursor.fetchone() is not None:
        return TOKENIZER_TRIGRAM

    return None


def get_db_tokenizer(conn: sqlite3.Connection) -> str:
    """Retrieve the active tokenizer for the database (defaults to trigram).

    Args:
        conn: Open SQLite connection.

    Returns:
        str: Active tokenizer name.
    """
    return get_stored_tokenizer(conn) or TOKENIZER_TRIGRAM


def init_db(
    conn: sqlite3.Connection,
    embed_dim: int = DEFAULT_EMBED_DIM,
    tokenizer: str | None = None,
    force_reinit_fts: bool = False,
) -> None:
    """Initialize the required database tables, triggers, and vector tables.

    Args:
        conn: Open SQLite connection.
        embed_dim: Dimension of embeddings for the vector table.
        tokenizer: Optional tokenizer name to configure for FTS5.
        force_reinit_fts: Whether to recreate FTS tables if tokenizer changed.

    Raises:
        ValueError: If tokenizer differs from stored tokenizer without force.
    """
    stored = get_stored_tokenizer(conn)
    if stored is not None:
        if tokenizer is not None and tokenizer != stored:
            if not force_reinit_fts:
                raise ValueError(
                    f"Database is already indexed with tokenizer '{stored}'. "
                    f"Use --force to re-index with '{tokenizer}'."
                )
            active_tokenizer = tokenizer
            recreate_fts = True
        else:
            active_tokenizer = stored
            recreate_fts = False
    else:
        active_tokenizer = (
            tokenizer if tokenizer is not None else TOKENIZER_TRIGRAM
        )
        recreate_fts = False

    with conn:
        conn.executescript(INIT_BASE_TABLES_SQL)

        if recreate_fts:
            conn.executescript(DROP_FTS_SQL)

        if active_tokenizer == TOKENIZER_TRIGRAM:
            conn.executescript(INIT_TRIGRAM_FTS_SQL)
        else:
            conn.executescript(INIT_UNICODE61_FTS_SQL)

        vec_sql = (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(\n"
            f"    id INTEGER PRIMARY KEY,\n"
            f"    embedding float[{embed_dim}]\n"
            f");"
        )
        conn.execute(vec_sql)
        # vec_chunks 同期用 DELETE トリガー
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ad_vec
            AFTER DELETE ON chunks BEGIN
                DELETE FROM vec_chunks WHERE id = old.id;
            END;
            """
        )

        # 既存 DB (v0) からのマイグレーション:
        # mtime カラムが存在しない場合に追加
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(files);")
        columns = {row[1] for row in cursor.fetchall()}
        if "mtime" not in columns:
            conn.execute(
                "ALTER TABLE files ADD COLUMN mtime REAL NOT NULL DEFAULT 0.0;"
            )

        # 使用トークナイザーを meta テーブルに保存
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) "
            "VALUES ('tokenizer', ?);",
            (active_tokenizer,),
        )

        # スキーマバージョンを記録
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
