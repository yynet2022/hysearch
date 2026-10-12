# hysearch アーキテクチャ設計書

## 1. 概要
`hysearch` は、デーモン常駐やDockerコンテナ、クラウドAPIに一切依存せず、単一のSQLiteデータベースファイルのみで動作するローカルCLIハイブリッド検索エンジンです。
キーワード検索（BM25）とベクトル類似度検索（セマンティック検索）を融合し、Reciprocal Rank Fusion (RRF) によって高精度な検索結果を提供します。

---

## 2. コアバリュー
1. **Zero-Daemon / Zero-Container**: 外部サービスやコンテナ不要。単一の SQLite ファイル（`.hysearch.db`）で動作。
2. **Fast Startup / Low Resource**: PyTorch等を用いず、ONNX Runtime ベースの `fastembed` を使用し、起動オーバーヘッドを最小化。
3. **日本語対応 (Portable)**: 外部形態素解析エンジンのインストール不要な SQLite FTS5 `trigram` トークナイザーを採用。
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
         ├─── (Raw Text) ───────┐
         │                      ▼
         │             ┌─────────────────┐
         │             │  SQLite FTS5    │
         │             │  (BM25 Score)   │
         │             └────────┬────────┘
         │                      │
         ▼                      │
┌─────────────────┐             │
│ FastEmbed       │             │
│ (ONNX Runtime)  │             │
└────────┬────────┘             │
         ▼                      │
┌─────────────────┐             │
│ sqlite-vec      │             │
│ (Vector Search) │             │
└────────┬────────┘             │
         │                      │
         ▼                      ▼
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

SQLite データベースには以下の4要素を構成しています。

1. **`files` テーブル**: ファイル単位の同期メタデータ（差分検知用）。
   - `id`: 主キー
   - `path`: 正規化ファイルパス（UNIQUE）
   - `hash`: SHA-256 16進ダイジェスト
   - `chunk_count`: 生成チャンク数
   - `mtime`: ファイル最終更新日時（POSIX エポック秒、REAL）
   - `updated_at`: 更新日時

2. **`chunks` テーブル**: チャンク実体。
   - `id`: 主キー
   - `file_id`: `files(id)` への外部キー（ON DELETE CASCADE）
   - `chunk_index`: ファイル内連番インデックス
   - `content`: チャンク文字列本文
   - `start_char`, `end_char`: 元ドキュメント内での文字位置オフセット

3. **`fts_chunks` 仮想テーブル (FTS5 trigram)**: 全文検索インデックス。
   - `content='chunks'`, `tokenize='trigram'` により外部形態素解析器なしで日本語・英語を高速検索。
   - `chunks_ai`, `chunks_ad`, `chunks_au` トリガーにより `chunks` テーブルの変更と自動同期。

4. **`vec_chunks` 仮想テーブル (sqlite-vec `vec0`)**: ベクトル類似度インデックス。
   - `id`: 主キー（`chunks(id)` と対応）
   - `embedding`: float[384] 次元埋め込みベクトル
   - `chunks_ad_vec` トリガーにより `chunks` レコード削除時に自動クリーンアップ。

また、SQLite の `PRAGMA user_version` を用いたスキーマバージョニング（現行: バージョン `1`）および初期スキーマからの自動マイグレーション機構（`mtime` カラム追加）を備えています。

---

## 6. コンポーネント詳細

### 6.1 Chunker (`hysearch.core.chunker`)
- スライディングウィンドウ方式を採用。
- ターゲットサイズ（デフォルト 500文字）、オーバーラップ（デフォルト 100文字）。
- 単純な文字数分割ではなく、Markdown 見出し（`# `）や段落（`\n\n`）、改行、文末句読点（`。`, `.`, `!` 等）を優先境界として分割。
- ファイル全体の SHA-256 ハッシュを計算し、未更新ファイルはスキップ。

### 6.2 Embedder (`hysearch.core.embedder`)
- `FastEmbedder`: `fastembed.TextEmbedding`（ONNX Runtime）による高速推論。デフォルトモデルは多言語対応の `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（384次元）。
- `DummyEmbedder`: テスト時にミリ秒単位で決定論的ベクトルを生成するモック。

### 6.3 Indexer (`hysearch.core.indexer`)
- 対象ディレクトリ配下のドキュメントおよびソースコード（デフォルト: `.md`, `.markdown`, `.txt`, `.py`, `.js`, `.ts`）を再帰走査。
- `--ext` オプション指定時は指定された拡張子のみを対象に走査（既存の他拡張子レコードはDB内に保持）。
- 差分検知により変更があったファイルのみ再インデックス。
- `--force` オプションで全ドキュメントの再構築をサポート。

### 6.4 Searcher (`hysearch.core.searcher`)
- **メタデータ・属性フィルタリング**: パスプレフィックス（`--path`）、globパターン（`--glob`）、更新日時範囲（`--since` / `--after`, `--until` / `--before`）に基づいて対象ファイルを検索前に判定し、合致するチャンクID群を抽出。
- 事前抽出された対象チャンクIDに基づき、BM25（`rowid IN (...)`）およびベクトル検索（`id IN (...)`）で探索空間を絞り込み。合致ファイルが0件の場合は即時早期リターン。
- BM25 検索とベクトル検索を並行実行し、RRF により統合順位を算出。
- FTS5 trigram が要求する3文字未満のクエリ（「AI」等）に対しては、安全にベクトル検索へフォールバックする防御設計。
- `hybrid`, `bm25`, `vector` のモード切り替えに対応。

### 6.5 CLI & UI (`hysearch.cli`)
- Typer による直感的なサブコマンド（`index`, `query`, `status`, `reset`）。
- `index` コマンドにおける特定拡張子の指定（`--ext`）。
- `query` コマンドにおける多彩な検索フィルタ（`--path`, `--glob`, `--since`, `--until` 等）と柔軟な日時パース（ISO 8601, YYYY-MM-DD 等）。
- Rich による洗練されたテーブル・パネル表示と、パイプ処理用の `--json` 出力。
