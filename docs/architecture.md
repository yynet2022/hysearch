# hysearch アーキテクチャ設計書

## 1. 概要
`hysearch` は、デーモン常駐やDockerコンテナ、クラウドAPIに一切依存せず、単一のSQLiteデータベースファイルのみで動作するローカルCLIハイブリッド検索エンジンです。
キーワード検索（BM25）とベクトル類似度検索（セマンティック検索）を融合し、Reciprocal Rank Fusion (RRF) によって高精度な検索結果を提供します。

---

## 2. コアバリュー
1. **Zero-Daemon / Zero-Container**: 外部サービスやコンテナ不要。単一の SQLite ファイル（`.hysearch.db`）で動作。
2. **Fast Startup / Low Resource**: PyTorch等を用いず、ONNX Runtime ベースの `fastembed` を使用し、起動オーバーヘッドを最小化。
3. **日本語形態素解析 & Trigram 両対応**: 追加インストール不要な SQLite FTS5 `trigram` トークナイザーを標準搭載しつつ、`SudachiPy`（`SplitMode.A` + 正規化）を導入すれば高精度な日本語分かち書き・表記揺れ吸収検索（1〜2文字単語のBM25一致を含む）に対応。
4. **Unix哲学 & エージェント親和性**: 人間向けのRichテーブル表示と、LLMエージェントや他CLIから扱いやすい `--json` パイプライン出力に対応。

---

## 3. システムアーキテクチャ図

```
[Documents & Code Files] (.md, .txt, .py, .js, .ts 等)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ Chunking & Hash Cache Engine                           │
│  - 見出し認識 / 段落境界スライディングウィンドウチャンカー     │
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

---

## 4. RRF (Reciprocal Rank Fusion) アルゴリズム
BM25のスコアとベクトル類似度の距離は単位・スケールが異なるため直接比較できません。そのため、双方の順位（Rank）を用いてスコアを統合します。

各ドキュメント（チャンク）$d$ に対する統合スコアの計算式:
$$RRF\_Score(d) = \sum_{m \in \{BM25, Vec\}} \frac{w_m}{k + rank_m(d)}$$

- $k$: 平滑化定数（デフォルト: $60$）
- $w_m$: 重み係数（デフォルト: $w_{bm25} = 1.0, w_{vec} = 1.0$）
- $rank_m(d)$: 各検索方式における順位（$1, 2, 3, \dots$）。検索結果に含まれない場合は計算外。

---

## 5. データベース設計 (SQLite Schema)

SQLite データベースには以下の5要素を構成しています。

1. **`meta` テーブル**: データベース全体の設定メタデータ管理。
   - `key`: 設定キー（`'tokenizer'`, `'embed_model'`, `'embed_dim'`）
   - `value`: 設定値（トークナイザー名、使用中の埋め込みモデル名、ベクトル次元数 等）

2. **`files` テーブル**: ファイル単位の同期メタデータ（差分検知用）。
   - `id`: 主キー
   - `path`: 正規化ファイルパス（UNIQUE）
   - `hash`: SHA-256 16進ダイジェスト
   - `chunk_count`: 生成チャンク数
   - `mtime`: ファイル最終更新日時（POSIX エポック秒、REAL）
   - `updated_at`: 更新日時

3. **`chunks` テーブル**: チャンク実体。
   - `id`: 主キー
   - `file_id`: `files(id)` への外部キー（ON DELETE CASCADE）
   - `chunk_index`: ファイル内連番インデックス
   - `content`: チャンク文字列本文（常に生テキストを保持し、スニペット表示に使用）
   - `start_char`, `end_char`: 元ドキュメント内での文字位置オフセット

4. **`fts_chunks` 仮想テーブル (FTS5)**: 全文検索インデックス。
   - **`trigram` モード時**: `content='chunks'`, `tokenize='trigram'` により外部形態素解析器なしで日本語・英語を高速検索。`chunks_ai`, `chunks_ad`, `chunks_au` トリガーにより `chunks` テーブルの変更と自動同期。
   - **`sudachi-*` モード時**: `tokenize='unicode61'` を使用し、`SudachiPy`（`SplitMode.A` + `normalized_form()`）で分かち書き・正規化したスペース区切りトークン列を格納。`chunks_ad` トリガーにより `chunks` 削除時に自動連動削除。

5. **`vec_chunks` 仮想テーブル (sqlite-vec `vec0`)**: ベクトル類似度インデックス。
   - `id`: 主キー（`chunks(id)` と対応）
   - `embedding`: 選択された埋め込みモデルに応じた次元数（デフォルト `float[384]`、モデルにより `float[768]` または `float[1024]`）の埋め込みベクトル
   - `chunks_ad_vec` トリガーにより `chunks` レコード削除時に自動クリーンアップ。

また、SQLite の `PRAGMA user_version` を用いたスキーマバージョニング（現行: バージョン `2`）および旧スキーマからの自動マイグレーション機構（`mtime` カラム追加・`meta` テーブルへの `tokenizer` / `embed_model` / `embed_dim` 初期化）を備えています。

---

## 6. コンポーネント詳細

### 6.1 Chunker (`hysearch.core.chunker`)
- スライディングウィンドウ方式を採用。
- ターゲットサイズ（デフォルト 500文字）、オーバーラップ（デフォルト 100文字）。
- 単純な文字数分割ではなく、Markdown 見出し（`# `）や段落（`\n\n`）、改行、文末句読点（`。`, `.`, `!` 等）を優先境界として分割。
- ファイル全体の SHA-256 ハッシュを計算し、未更新ファイルはスキップ。

### 6.2 Embedder (`hysearch.core.embedder`)
- `FastEmbedder`: `fastembed.TextEmbedding`（ONNX Runtime）による高速推論。文書チャンクの埋め込み生成には `model.passage_embed()` を、検索クエリの埋め込み生成には `model.query_embed()` を使用。
- 選択可能な埋め込みモデル（`--model` / `-m` オプションで選択可能。初回作成時に `meta` テーブルへモデル名と次元数を記録し、以降の更新・検索で自動参照）:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（384次元 / デフォルト）
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`（768次元）
  - `Qwen/Qwen3-Embedding-0.6B-Q`（1024次元）
  - `google/embeddinggemma-300m`（768次元）
- `DummyEmbedder`: テスト時にミリ秒単位で決定論的ベクトルを生成するモック。

### 6.3 Tokenizer (`hysearch.core.tokenizer`)
- 初回インデックス作成時に `--tokenizer` が省略された場合、`sudachi-full` → `sudachi-core` → `sudachi-small` → `trigram` の優先順でインストール済みトークナイザーを自動検出し、`meta` テーブルに永続化。
- 2回目以降のインデックス更新や `query` / `status` 実行時は、`meta` テーブルに保存されたトークナイザー設定を自動的に参照。
- `SudachiPy` 利用時は `SplitMode.A`（短単位分割）と `morpheme.normalized_form()`（表記揺れ正規化）を実行し、句読点・記号を除外した語幹トークン列を生成。

### 6.4 Indexer (`hysearch.core.indexer`)
- 対象ディレクトリ配下のドキュメントおよびソースコード（デフォルト: `.md`, `.markdown`, `.txt`, `.py`, `.js`, `.ts`）を再帰走査。
- `--ext` オプション指定時は指定された拡張子のみを対象に走査（既存の他拡張子レコードはDB内に保持）。
- `--exclude-dir`（`-x`）オプション指定時は、指定されたディレクトリ名（例: `.venv`, `node_modules`）や相対パスを走査時に枝刈り除外。
- `--dry-run`（`-n`）オプション指定時は、DBへの書き込みや埋め込みモデルの推論を行わず、スキャン対象ファイル一覧の都度出力と統計集計のみを実行。
- 差分検知により変更があったファイルのみ再インデックス。
- `--force` オプションで全ドキュメントの再構築（および `--tokenizer` 変更時の FTS テーブル再初期化、`--model` 変更時のベクトルテーブル再初期化）をサポート。
- 存在しなくなったファイルを検知し、外部キー制約のカスケードとトリガーにより関連全データを安全に連動削除する `prune_missing_files` を提供。

### 6.5 Searcher (`hysearch.core.searcher`)
- **メタデータ・属性フィルタリング**: パスプレフィックス（`--path`）、globパターン（`--glob`）、更新日時範囲（`--since` / `--after`, `--until` / `--before`）に基づいて対象ファイルを検索前に判定し、合致するチャンクID群を抽出。
- 事前抽出された対象チャンクIDに基づき、BM25（`rowid IN (...)`）およびベクトル検索（`id IN (...)`）で探索空間を絞り込み。合致ファイルが0件の場合は即時早期リターン。
- BM25 検索とベクトル検索を並行実行し、RRF により統合順位を算出。
- `trigram` モード時は3文字未満のクエリに対して安全にベクトル検索へフォールバックし、`sudachi-*` モード時は1〜2文字の単語クエリでも形態素トークン一致による高精度な BM25 検索を実行。
- `hybrid`, `bm25`, `vector` のモード切り替えに対応。

### 6.6 CLI & UI (`hysearch.cli`)
- Typer による直感的なサブコマンド（`index`, `query`, `status`, `prune`, `reset`, `server`, `mcp`）。
- `index` コマンドにおける特定拡張子の指定（`--ext`）、除外ディレクトリの指定（`--exclude-dir` / `-x`）、ドライラン確認（`--dry-run` / `-n`）、トークナイザー指定（`--tokenizer` / `-t`）、および埋め込みモデル指定（`--model` / `-m`）。
- `query` コマンドにおける多彩な検索フィルタ（`--path`, `--glob`, `--since`, `--until` 等）と柔軟な日時パース（ISO 8601, YYYY-MM-DD 等）。
- `prune` コマンドにおける削除ファイルのリアルタイム都度通知とドライラン（`--dry-run`）。
- Rich による洗練されたテーブル・パネル表示（ライト/ダーク両背景で見やすい配色）と、パイプ処理用の `--json` 出力。

### 6.7 Web Server & Browser UI (`hysearch.web`)
- FastAPI と Uvicorn による軽量ローカル Web サーバー（`hysearch server` コマンドで起動）。
- 初回検索時に DB の `meta` テーブルに記録された埋め込みモデル・次元数を読み取って `FastEmbedder` インスタンスをメモリ内にキャッシュし、2回目以降の検索レスポンスを高速化。
- デフォルトポート（`8888`）が使用中の場合は空きポートを自動探索してバインドし、`--port` 明示指定時はポート競合を即座に検知してエラー終了する安全なポート解決機構（`resolve_server_port`）。
- サーバー起動と同時に標準ブラウザを自動で開く機能（`--open` / `--no-open`）。
- シングルページ HTML UI（`/`）と REST エンドポイント（`/api/status`, `/api/query`）により、ブラウザ上でインデックス状態の確認および各種パラメータ・フィルタを用いた対話的な連続検索を提供。

### 6.8 MCP Server (`hysearch.mcp`)
- 公式 `mcp` パッケージ（`mcp.server.mcpserver.MCPServer`）による Model Context Protocol (MCP) サーバー（`hysearch mcp --db <path>` コマンドで `stdio` トランスポートにて起動）。
- 読み取り専用の2つのツール（`status` および `query`）のみを公開し、双方に `READ_ONLY_ANNOTATIONS`（`readOnlyHint=True`, `destructiveHint=False`, `idempotentHint=True`, `openWorldHint=False`）を付与。
- `status` ツールは DB の統計情報（パス、サイズ、トークナイザー、埋め込みモデル、次元数、ファイル数、チャンク数、最終更新日時）を JSON 文字列で返却。
- `query` ツールは CLI / Web と同一の検索パラメータ・フィルタ（`query`, `top`, `mode`, `rrf_k`, `path`, `glob`, `since`, `after`, `until`, `before`）を受け付け、`SearchResponse` を JSON 文字列で返却。
- ベクトル検索を含むモードでは、DB に記録された埋め込みモデル名・次元数に対応する `FastEmbedder` インスタンスを MCP サーバープロセス内でキャッシュし、連続する検索ツール呼び出しを高速化。

