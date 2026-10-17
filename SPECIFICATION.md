# プロジェクト設計仕様書: hysearch (Local Hybrid Search CLI)

本仕様書は、コーディングエージェント（Claude Code, Cursor, Roo Code, Cline等）が外部対話なしで自律的に実装・テスト・パッケージングを完遂できるように設計された完全な仕様書である。

---

## 1. プロジェクト概要

### 1.1 目的
デーモン常駐やDockerコンテナ、クラウドAPIに一切依存せず、単一のSQLiteデータベースファイルのみで高速なハイブリッド検索（BM25キーワード検索 + ベクトルセマンティック検索）を提供するローカルCLIツール **`hysearch`** を構築する。

### 1.2 コアバリュー
1. **Zero-Daemon / Zero-Container**: SQLite単一ファイル（`.hysearch.db` 等）のみで完結。
2. **Fast Startup / Low Resource**: PyTorch等を使用せず、ONNXベースの `fastembed` を採用しCLI呼び出しオーバーヘッドを最小化。
3. **日本語形態素解析 & Trigram 両対応**: 外部形態素解析エンジンのインストール不要な SQLite FTS5 `trigram` トークナイザーを標準搭載しつつ、オプションで `SudachiPy`（`SplitMode.A` + 正規化）による高精度な日本語分かち書き・表記揺れ吸収検索（1〜2文字単語のBM25一致を含む）に対応。
4. **Unix哲学 & エージェント親和性**: 人間向けのRichテーブル表示と、他CLI/LLMエージェントからパイプ処理しやすい `--json` 出力の両対応。

---

## 2. システムアーキテクチャ

### 2.1 アーキテクチャ概要

```
[Markdown / Text Files]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Chunking & Hash Cache Engine                           │
│  - 見出し認識 / スライディングウィンドウチャンカー     │
│  - SHA-256ハッシュによる差分更新検知                   │
└────────┬───────────────────────────────────────────────┘
         │
         ├─── (Raw / Sudachi Tokenized Text) ──┐
         │                                     ▼
         │                            ┌─────────────────┐
         │                            │  SQLite FTS5    │
         │                            │  (BM25 Score)   │
         │                            └────────┬────────┘
         │                                     │
         ▼                                     │
┌─────────────────┐                            │
│ FastEmbed       │                            │
│ (ONNX Runtime)  │                            │
└────────┬────────┘                            │
         ▼                                     │
┌─────────────────┐                            │
│ sqlite-vec      │                            │
│ (Vector Search) │                            │
└────────┬────────┘                            │
         │                                     │
         ▼                                     ▼
┌────────────────────────────────────────────────────────┐
│ Reciprocal Rank Fusion (RRF) Engine                    │
│  - Rank統合スコアリング                                │
│  - Top-Kソート・スニペット生成                         │
└────────────────────────────────────────────────────────┘
```

### 2.2 RRF (Reciprocal Rank Fusion) アルゴリズム
BM25のスコア（負の値や非正規化スコア）と、ベクトル検索の距離（L2距離 / コサイン類似度）は直接比較できないため、双方の順位（Rank）を用いてスコアを統合する。

ドキュメント $d$ に対する統合スコア:
$$RRF\_Score(d) = \sum_{m \in \{BM25, Vec\}} \frac{w_m}{k + rank_m(d)}$$

* $k$: 平滑化定数（デフォルト: $60$）
* $w_m$: 重み係数（デフォルト: $w_{bm25} = 1.0, w_{vec} = 1.0$）
* $rank_m(d)$: 各検索方式における順位（$1, 2, 3, \dots$）。検索結果に含まれない場合は計算外。

---

## 3. ディレクトリ構成 (`src/` レイアウト)

```
hysearch/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   └── architecture.md
├── src/
│   └── hysearch/
│       ├── __init__.py
│       ├── py.typed
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py            # Typer CLIエントリーポイント
│       │   └── display.py         # Richコンソール・フォーマッタ
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py          # 定数・設定値管理
│       │   ├── database.py        # SQLite接続・sqlite-vec拡張ロード・初期化
│       │   ├── chunker.py         # 文書分割・ハッシュ計算
│       │   ├── embedder.py        # FastEmbedラッパー
│       │   ├── tokenizer.py       # トークナイザー自動検出・Sudachi分かち書き・正規化
│       │   ├── indexer.py         # インデックス登録・更新パイプライン
│       │   ├── searcher.py        # ハイブリッド検索・RRF統合ロジック
│       │   └── models.py          # Pydanticデータモデル / 型定義
│       └── utils/
│           ├── __init__.py
│           └── text.py            # テキストクリーニング・ハイライト処理
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # 共通フィクスチャ (オンメモリDB, ダミーEmbedder)
│   ├── test_cli.py                # Typer Runner によるCLIテスト
│   ├── test_chunker.py            # チャンキング・ハッシュ検知テスト
│   ├── test_database.py           # テーブル作成・FTS5/sqlite-vec動作テスト
│   ├── test_tokenizer.py          # トークナイザー解決・Sudachi分かち書きテスト
│   ├── test_indexer.py            # 差分インデックス処理テスト
│   └── test_searcher.py           # RRF計算・検索ロジックテスト
├── .gitignore
├── pyproject.toml
├── README.md
└── SPECIFICATION.md
```

---

## 4. プロジェクト設定と品質管理 (`pyproject.toml`)

ビルドバックエンドには `hatchling` を採用し、`ruff`, `mypy`, `pytest` を厳格に設定する。

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "hysearch"
version = "0.1.0"
description = "Lightweight, zero-daemon local hybrid search engine powered by SQLite and fastembed"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Author", email = "author@example.com" }]
dependencies = [
    "typer>=0.12.0",
    "rich>=13.7.0",
    "sqlite-vec>=0.1.6",
    "fastembed>=0.4.0",
    "pydantic>=2.7.0",
]

[project.optional-dependencies]
sudachi = [
    "sudachipy>=0.6.8",
    "sudachidict_core>=20240109",
]
sudachi-core = [
    "sudachipy>=0.6.8",
    "sudachidict_core>=20240109",
]
sudachi-small = [
    "sudachipy>=0.6.8",
    "sudachidict_small>=20240109",
]
sudachi-full = [
    "sudachipy>=0.6.8",
    "sudachidict_full>=20240109",
]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]

[project.scripts]
hysearch = "hysearch.cli.main:app"

[tool.ruff]
line-length = 79
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "D",   # pydocstyle
    "B",   # flake8-bugbear
    "UP",  # pyupgrade
    "SIM", # flake8-simplify
    "RUF", # ruff-specific
]
ignore = [
    "D203", # 1 blank line required before class docstring (conflicts with D211)
    "D213", # Multi-line docstring summary should start at the second line (conflicts with D212)
    "D400", # First line should end with a period (incompatible with Japanese)
    "D415", # First line should end with a period (incompatible with Japanese)
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.10"
packages = ["hysearch"]
strict = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = [
    "sqlite_vec.*",
    "fastembed.*",
    "sudachipy.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-ra -q --cov=hysearch --cov-report=term-missing"
```

---

## 5. データベース設計 (SQLite Schema)

### 5.1 スキーマ定義
DBには、DB設定メタデータ（`meta`）、ファイル同期メタデータ（`files`）、元データ実体（`chunks`）、FTS5全文検索テーブル（`fts_chunks`）、`sqlite-vec`仮想テーブル（`vec_chunks`）を構成する。

```sql
-- 0. DBメタデータ管理（使用トークナイザー等の記録）
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- 1. ファイル単位の同期メタデータ（差分検知用）
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    hash TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    mtime REAL NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. チャンク実体
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

-- 3A. 全文検索用 FTS5 仮想テーブル (trigram モード時)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='trigram'
);

-- 4A. FTS5同期用トリガー (trigram モード時)
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO fts_chunks(rowid, content) VALUES (new.id, new.content);
END;

-- 3B. 全文検索用 FTS5 仮想テーブル (sudachi-* モード時: スペース区切りの正規化トークン列を格納)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    content,
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM fts_chunks WHERE rowid = old.id;
END;

-- 5. ベクトル検索テーブル (sqlite-vec)
-- デフォルトモデル: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384次元)
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    id INTEGER PRIMARY KEY,
    embedding float[384]
);
```

### 5.2 スキーマバージョニングと自動マイグレーション
SQLite の `PRAGMA user_version` を用いてスキーマバージョンを管理する（現行バージョン: `2`）。
* `user_version = 0`（初期バージョン）からの移行時、`files` テーブルに `mtime` カラムが存在しない場合は `ALTER TABLE files ADD COLUMN mtime REAL NOT NULL DEFAULT 0.0;` を自動実行してマイグレーションを行う。
* `user_version < 2` からの移行時、`meta` テーブルを作成し、既存の `fts_chunks` テーブルがある場合は `tokenizer = 'trigram'` として記録する。
* `get_schema_version(conn)` により現在のスキーマバージョンを取得可能。

### 5.3 sqlite-vec 接続要件
```python
import sqlite3
import sqlite_vec

def create_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn
```

---

## 6. コアモジュール設計

### 6.1 `src/hysearch/core/models.py`
```python
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    chunk_index: int
    content: str
    start_char: int
    end_char: int

class SearchHit(BaseModel):
    chunk_id: int
    file_path: str
    chunk_index: int
    content: str
    rrf_score: float
    bm25_rank: Optional[int] = None
    vec_rank: Optional[int] = None
    snippet: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    total_hits: int
    hits: list[SearchHit]
    execution_time_ms: float

class IndexingStats(BaseModel):
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    elapsed_seconds: float = 0.0
    tokenizer: str = "trigram"
```

### 6.2 `src/hysearch/core/chunker.py`
* **分割戦略**:
  * ターゲット文字数: デフォルト `500` 文字
  * オーバーラップ: デフォルト `100` 文字
  * 改行（`\n\n`）やMarkdown見出し（`# `）を可能な限り境界として優先。
  * ファイル全体のSHA-256ハッシュを算出し、変更がない場合はスキップする。

### 6.3 `src/hysearch/core/embedder.py`
* `fastembed.TextEmbedding` をラップする。
* デフォルトモデル: `"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"` (次元数: 384)
* テスト実行時は、重いONNXモデルをロードせずランダムまたは固定値ベクトルを返す `DummyEmbedder` に差し替え可能な抽象プロトコル (`EmbedderProtocol`) を定義すること。

### 6.4 `src/hysearch/core/tokenizer.py`
* **トークナイザー解決と自動検出**:
  * 初回インデックス作成時に `--tokenizer` が未指定の場合、`importlib.util.find_spec` を用いて `sudachi-full` → `sudachi-core` → `sudachi-small` → `trigram` の優先順位で利用可能なトークナイザーを自動検出し、`meta` テーブルに記録する。
  * 明示的に指定されたトークナイザーが環境にインストールされていない場合は `RuntimeError` を送出する。
* **Sudachi 分かち書きと正規化 (`tokenize_with_sudachi`)**:
  * `SudachiPy` の `SplitMode.A`（短単位分割）と `morpheme.normalized_form()`（表記正規化）を組み合わせて形態素解析を実行し、記号・句読点を除いた語幹トークン列を抽出する。
  * FTS5（`unicode61`）登録用にスペース区切り文字列へ整形する（`format_fts_content`）とともに、検索クエリも同様に分割・正規化して `MATCH` 式を構築する（`build_fts_query`）。

### 6.5 `src/hysearch/core/indexer.py`
* **インデックスパイプライン**:
  * 指定ディレクトリ配下のファイルを再帰走査（デフォルト: `.md`, `.markdown`, `.txt`, `.py`, `.js`, `.ts`）。
  * SHA-256ハッシュによる差分検知により、未変更ファイルをスキップ。
  * `extensions` 引数により対象拡張子を上書き可能。
  * `chunks` テーブルには常に生テキストを保存し（スニペット表示用）、トークナイザーが `sudachi-*` の場合は `format_fts_content` で分かち書きしたテキストを `fts_chunks` に挿入する。
* **孤立ファイル削除 (`prune_missing_files`)**:
  * DB内の全登録ファイル（`files` テーブル）の実体存在を走査。
  * ディスク上に存在しないファイルを検知し、`dry_run=False` の場合は `DELETE FROM files WHERE id = ?` を実行。外部キー制約（`CASCADE`）およびトリガーにより、チャンク・FTS5・ベクトルインデックスも連動して自動完全消去。
  * 各欠落ファイル検知時にコールバック（`on_missing`）を呼び出し、リアルタイム都度通知を支援。

### 6.6 `src/hysearch/core/searcher.py`
* **メタデータ・属性フィルタリング (`get_filtered_chunk_ids`)**:
  * 検索実行前に、パスプレフィックス（`path_filter`）、ファイルglob（`glob_filter`）、更新日時（`since`, `until`）に基づいて対象ファイルを走査・判定し、合致するチャンクIDリスト（`target_ids`）を抽出する。
  * フィルタ条件に合致するファイル・チャンクが0件の場合は、FTS5およびベクトル検索を実行せず即座に0件の `SearchResponse` を返却する。
* **BM25クエリ**:
  * DBの `meta` テーブルから使用トークナイザーを取得し、`build_fts_query` でクエリを構築する。
  * FTS5 `trigram` の場合、検索文字列が3文字未満のときは BM25 をスキップしてベクトル検索単体に委ねる。一方、`sudachi-*` の場合は1〜2文字の単語（例: 「AI」「設定」「猫」）でも分かち書きトークン単位でBM25検索を実行する。
  * `target_ids` が指定されている場合は `WHERE fts_chunks MATCH ? AND rowid IN (...) ORDER BY rank LIMIT ?` を実行。
* **Vectorクエリ**:
  * `query` を埋め込みベクトルに変換後、`sqlite_vec.serialize_float32(vec)` を使用。
  * `target_ids` が指定されている場合は `WHERE embedding MATCH ? AND k = ? AND id IN (...) ORDER BY distance` を実行。
* **RRF統合ロジック**:
  * BM25上位 $N$ 件（例: 50件）とVector上位 $N$ 件（例: 50件）を取得。
  * 各IDについて $RRF\_Score$ を計算して合算し、降順ソートの上位 $K$ 件を返す。
  * IDから `chunks` と `files` をJOINして本文とファイルパスを取得。

---

## 7. CLI インターフェース設計 (`src/hysearch/cli/main.py`)

Typerを使用して実装。全コマンドで `--db` オプション（デフォルト: `./.hysearch.db`）をサポート。

### 7.1 コマンド一覧

#### 1. `hysearch index [DIR_PATH]`
指定ディレクトリ配下のドキュメントおよびソースコード（デフォルト: `.md`, `.markdown`, `.txt`, `.py`, `.js`, `.ts`）を再帰的に走査してインデックスを作成・更新。
* オプション:
  * `--chunk-size` (int, default: 500)
  * `--overlap` (int, default: 100)
  * `--force` / `-f` (bool, default: False) - ハッシュ差分を無視して全再構築
  * `--ext` / `-e` (list[str] | None, default: None) - インデックス対象の拡張子（指定した場合デフォルトを上書き。既存の他拡張子データはDBに保持）
  * `--tokenizer` / `-t` (str | None, default: None) - FTSトークナイザーの指定（`sudachi-full`, `sudachi-core`, `sudachi-small`, `sudachi`, `trigram`。省略時は初回に自動検出しDBに記録。既存DBと異なるトークナイザーに変更する場合は `--force` が必須）

#### 2. `hysearch query [QUERY]`
自然言語またはキーワードでハイブリッド検索を実行。
* オプション:
  * `--top` / `-k` (int, default: 5): 上位件数
  * `--mode` (choice: `hybrid` | `bm25` | `vector`, default: `hybrid`)
  * `--rrf-k` (int, default: 60): RRF平滑化定数
  * `--path` / `-p` (str | None): ディレクトリパスやファイル名プレフィックスで絞り込み
  * `--glob` / `-g` (str | None): ファイルglobパターン（`*.md`, `src/**/*.py` 等）で絞り込み
  * `--since` (str | None): 指定日時以降に更新されたファイルに限定（ISO 8601, YYYY-MM-DD, YYYY/MM/DD 等）
  * `--after` (str | None): `--since` のエイリアス
  * `--until` (str | None): 指定日時以前に更新されたファイルに限定
  * `--before` (str | None): `--until` のエイリアス
  * `--json` (bool, default: False): 結果をJSONで標準出力（パイプ処理用）

#### 3. `hysearch status`
現在のインデックス状態を表示（使用トークナイザー、ファイル数、チャンク総数、DBファイルサイズ、最終更新時刻）。

#### 4. `hysearch prune`
ディスク上から削除・移動されたファイルを検出し、データベースから安全に除外。
* オプション:
  * `--dry-run` / `-n` (bool, default: False): 実際には削除せず、対象ファイルのみ表示
  * `--json` (bool, default: False): 結果をJSONで標準出力
  * `--db` (Path, default: `./.hysearch.db`): データベースファイルパス

#### 5. `hysearch reset`
DBファイルを削除または全テーブルをクリア（確認プロンプト付き）。

---

## 8. 実装における重要な注意点・落とし穴

1. **`sqlite-vec` のトランザクション**:
   * `vec_chunks` は仮想テーブルであるため、同一トランザクション内で複数件をINSERTする際はコミットのタイミングに注意する。
2. **日本語短文字検索の trigram 挙動と Sudachi 分かち書き**:
   * FTS5 `trigram` トークナイザーは3文字以上のサブストリングでインデックスを作成する。1〜2文字のクエリ（例: 「AI」「設定」）が入力された場合、FTS5の MATCH クエリではヒットしないため、BM25 検索をスキップしてベクトル検索単体でフォールバックする。
   * 一方、`sudachi-*` トークナイザー使用時は `SplitMode.A` と `normalized_form()` により単語単位で `unicode61` FTS5 テーブルへ格納するため、1〜2文字の日本語キーワードでも正確に BM25 検索が可能となる。
3. **リソースの適切なクローズ**:
   * SQLite接続はコンテキストマネージャ（`with` 構文）で管理し、CLI終了時にリークさせない。

---

## 9. テスト計画 (`tests/`)

全テストはローカル環境で高速に実行可能（2秒以内目標）であること。

### 9.1 テストケース一覧
1. **`test_chunker.py`**:
   * 通常テキストが適切なサイズとオーバーラップで分割されること。
   * 同一内容のテキストに対して同一のハッシュ値が返ること。
2. **`test_database.py`**:
   * インメモリDB (`:memory:`) に対するスキーマ初期化が正常に行われること。
   * `sqlite-vec` 拡張が正常にロードされ、ベクトルの保存と近傍探索ができること。
   * `meta` テーブルへのトークナイザー保存と、`force_reinit_fts` によるFTS再構築が動作すること。
3. **`test_tokenizer.py`**:
   * トークナイザーの自動検出優先順位（`full` → `core` → `small` → `trigram`）および明示指定時のバリデーションが正しく動作すること。
   * Sudachi `SplitMode.A` と `normalized_form()` による分かち書き・句読点除外・FTSクエリ生成が正しく動作すること。
4. **`test_indexer.py`**:
   * ファイル更新時に変更のあったファイルのみチャンクが再生成されること（差分更新）。
   * ファイル削除時に `ON DELETE CASCADE` でチャンクとFTS/vecインデックスが削除されること。
5. **`test_searcher.py`**:
   * RRF計算が正しく順位に反比例したスコアを返すこと。
   * 日本語テキストのキーワード検索とベクトル検索がそれぞれ動作すること。
6. **`test_cli.py`**:
   * `typer.testing.CliRunner` を使用し、`index`, `query`, `query --json`, `status`, `prune` コマンドおよび `--tokenizer` オプションが正常に動作すること。

---

## 10. コーディングエージェントへの実装タスク指示

エージェントは以下のステップ順に実装と検証を進めること:

1. **Phase 1: 環境 & パッケージング**
   * `pyproject.toml` を作成し、`src/hysearch` ディレクトリレイアウトを配置。
   * `src/hysearch/py.typed` を作成。
2. **Phase 2: モデル & データベース層**
   * `models.py`, `config.py`, `database.py` を実装。
   * `tests/test_database.py` を作成しパスさせる。
3. **Phase 3: チャンカー & 埋め込み層**
   * `chunker.py`, `embedder.py` を実装。
   * `tests/test_chunker.py` を作成しパスさせる。
4. **Phase 4: インデクサー & サーチャー層**
   * `indexer.py`, `searcher.py`, `utils/text.py` を実装。
   * `tests/test_indexer.py`, `tests/test_searcher.py` を作成しパスさせる。
5. **Phase 5: CLI & UI層**
   * `cli/display.py`, `cli/main.py` を実装。
   * `tests/test_cli.py` を作成しパスさせる。
6. **Phase 6: 静的解析 & CI**
   * `ruff check src tests` の実行と修正。
   * `ruff format --check src tests` の実行と修正。
   * `mypy src` の実行と型エラーの解消。
   * `pytest` ですべてのテストがパスすることを確認。
   * README.md に利用例（CLIコマンド例、アーキテクチャ概要）を記載。
