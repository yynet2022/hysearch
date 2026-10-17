# hysearch

Lightweight, zero-daemon local hybrid search engine powered by SQLite and fastembed.

`hysearch` は、デーモン常駐やDockerコンテナ、クラウドAPIに一切依存せず、単一のSQLiteファイル（`.hysearch.db`）のみで高速なハイブリッド検索（BM25キーワード検索 + ベクトルセマンティック検索）を提供するローカルCLIツールです。

---

## 主な特徴

- **Zero-Daemon / Zero-Container**: 外部サービスやDockerコンテナ不要。単一の SQLite データベースファイルのみで完結。
- **Fast Startup / Low Resource**: PyTorch等を使わず、ONNX Runtime ベースの `fastembed` を採用し、高速なCLI起動を実現。
- **日本語形態素解析 & Trigram 両対応**: 追加インストール不要な SQLite FTS5 `trigram` トークナイザーを標準搭載しつつ、`SudachiPy`（`SplitMode.A` + 正規化）を導入すれば高精度な日本語分かち書き・表記揺れ吸収検索（1〜2文字単語のBM25一致を含む）に対応。
- **Unix哲学 & エージェント親和性**: 人間向けのRichテーブル表示と、他CLIやLLMエージェントからパイプ処理しやすい `--json` 出力の両対応。
- **RRF (Reciprocal Rank Fusion)**: キーワード検索の順位とベクトル類似度の順位を平滑化統合し、高精度な検索結果を提示。

---

## 必要要件

- Python >= 3.10
- Windows, macOS, Linux

---

## インストール方法

```bash
# リポジトリのクローン
git clone https://github.com/yynet2022/hysearch.git
cd hysearch

# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # Windows の場合: .venv\Scripts\Activate.ps1

# 標準インストール (SQLite FTS5 trigram を使用)
pip install -e .

# 日本語の高精度分かち書き (SudachiPy) を使う場合 (辞書サイズ別に選択可能)
pip install -e ".[sudachi]"        # 標準の core 辞書 (sudachidict_core)
pip install -e ".[sudachi-small]"  # 軽量な small 辞書 (sudachidict_small)
pip install -e ".[sudachi-full]"   # 固有名詞も豊富な full 辞書 (sudachidict_full)

# 開発用ツール（pytest, ruff, mypy 等）も含める場合
pip install -e ".[dev]"
```

---

## 使い方 (CLI コマンド)

### 1. ドキュメントのインデックス作成・更新 (`index`)
指定ディレクトリ配下のドキュメントおよびソースコード（デフォルト: `.md`, `.markdown`, `.txt`, `.py`, `.js`, `.ts`）を再帰的に走査し、インデックスを作成します。SHA-256 ハッシュによる差分検知を行い、変更のあったファイルのみ更新します。

初回インデックス作成時、`--tokenizer` (`-t`) を省略すると、環境にインストールされているトークナイザーを **`sudachi-full` → `sudachi-core` → `sudachi-small` → `trigram`** の優先順で自動検出し、データベース内のメタデータに記録します。2回目以降の更新や検索 (`query`) では、記録されたトークナイザーが自動的に使用されます。

```bash
# カレントディレクトリ配下をインデックス化 (トークナイザーは初回に自動検出)
hysearch index .

# トークナイザーを明示的に指定してインデックス化 (-t / --tokenizer)
# 指定可能な値: sudachi-full, sudachi-core, sudachi-small, sudachi, trigram
hysearch index ./docs --tokenizer sudachi-core
hysearch index ./docs -t trigram

# 既存DBのトークナイザーを別のものに変更する場合は --force (-f) を併用
hysearch index ./docs -t sudachi-full --force

# チャンクサイズとオーバーラップを指定してインデックス化
hysearch index ./docs --chunk-size 400 --overlap 80

# 特定の拡張子だけを対象にして追加・更新 (--ext / -e, 既存の別拡張子はDB内にそのまま保持)
hysearch index ./src --ext .py
hysearch index ./src -e .py -e .ts
hysearch index ./src --ext "py,js,ts"

# 差分キャッシュを無視して強制的に全再構築
hysearch index ./docs --force

# データベースファイルの保存先を指定
hysearch index ./docs --db ./my_search.db
```

### 2. ハイブリッド検索の実行 (`query`)
自然言語やキーワードで検索を実行します。

```bash
# ハイブリッド検索 (デフォルト: BM25 + Vector の RRF 統合)
hysearch query "データベース接続の管理"

# 上位件数を指定 (例: 上位 10 件)
hysearch query "セマンティック検索" -k 10

# 検索モードの切り替え (hybrid, bm25, vector)
hysearch query "FastEmbed" --mode bm25
hysearch query "自然言語処理" --mode vector

# ディレクトリやパスプレフィックスで絞り込み (--path / -p)
hysearch query "設定" --path ./docs
hysearch query "アーキテクチャ" -p src/hysearch/

# ファイルの glob パターンで絞り込み (--glob / -g)
hysearch query "仕様" --glob "*.md"
hysearch query "アルゴリズム" -g "src/**/*.py"

# 更新日時で絞り込み (--since / --after: 指定日時以降、--until / --before: 指定日時以前)
# 日時は ISO 8601 (YYYY-MM-DD 等) や YYYY/MM/DD、年月指定 (YYYY-MM 等) に対応
hysearch query "リリースノート" --since 2026-10-01
hysearch query "変更履歴" --after 2026/10
hysearch query "旧仕様" --until 2026-09-30
hysearch query "設計方針" --before 2026/08

# フィルタの組み合わせ（例: 2026年10月以降に更新された docs 配下の Markdown）
hysearch query "ハイブリッド検索" -p ./docs -g "*.md" --since 2026-10-01

# JSON 出力 (jq や他ツール、LLM エージェントへのパイプライン処理用)
hysearch query "アーキテクチャ" --json | jq .
```

### 3. インデックス状態の確認 (`status`)
現在のインデックス状態（使用中のトークナイザー、登録ファイル数、チャンク総数、DBファイルサイズ、最終更新時刻）を表示します。

```bash
hysearch status

# JSON 形式で出力
hysearch status --json
```

### 4. 削除されたファイルの整理・除外 (`prune`)
ディスク上から削除・移動されたファイルを検出し、データベースから安全に除外します。外部キー制約とトリガーにより、関連するチャンク・FTS5 全文インデックス・sqlite-vec ベクトルインデックスもすべて連動して自動削除されます。

```bash
# 存在しなくなったファイルを DB から削除（検出のたびに都度表示）
hysearch prune

# 実際には削除せず、対象ファイルのみ確認 (--dry-run / -n)
hysearch prune --dry-run

# JSON 形式で結果を出力
hysearch prune --json
```

### 5. インデックスのリセット (`reset`)
データベースファイルを削除し、インデックスを初期化します。

```bash
# 確認プロンプト付きでリセット
hysearch reset

# 確認プロンプトをスキップして強制削除
hysearch reset -y
```

### 6. ブラウザ Web UI & サーバー起動 (`server`)
FastAPI / Uvicorn によるローカル Web サーバーを起動し、ブラウザ上でインデックス状態（`status`）の確認や、パラメータ・フィルタ（`mode`, `top_k`, `rrf_k`, `path`, `glob`, `since`, `until`）を指定した対話的な繰り返し検索（`query`）を行えます。
`--port` 省略時はデフォルトの `8888` から順に空きポートを自動探索し、起動と同時にブラウザを自動で開きます（`Ctrl+C` で停止）。`--port` を明示指定した場合、そのポートが使用中であればエラー終了します。

```bash
# デフォルト設定 (127.0.0.1:8888 から空きポートを自動探索してブラウザを起動)
hysearch server

# ホスト名やポート番号を明示指定 (--host / -H, --port / -p)
hysearch server --host 0.0.0.0 --port 9000

# ブラウザの自動起動を抑止 (--no-open)
hysearch server --no-open
```

---

## 検索アルゴリズム: RRF (Reciprocal Rank Fusion)

BM25（FTS5）とベクトル類似度（sqlite-vec）の検索スコアを統合するため、Reciprocal Rank Fusion を採用しています。

$$RRF\_Score(d) = \sum_{m \in \{BM25, Vec\}} \frac{w_m}{k + rank_m(d)}$$

- $k$: 平滑化定数（デフォルト: 60）
- $w_m$: 重み係数（デフォルト: $w_{bm25} = 1.0, w_{vec} = 1.0$）
- $rank_m(d)$: 各検索方式における順位（1-based）

---

## 開発とテスト

```bash
# Linter & Formatter (Ruff)
ruff check src tests
ruff format --check src tests

# 型チェック (Mypy: strict モード)
mypy src tests

# 単体テスト & カバレッジ測定 (Pytest)
pytest
```

---

## ライセンス

MIT License
