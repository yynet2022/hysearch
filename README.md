# hysearch

Lightweight, zero-daemon local hybrid search engine powered by SQLite and fastembed.

`hysearch` は、デーモン常駐やDockerコンテナ、クラウドAPIに一切依存せず、単一のSQLiteファイル（`.hysearch.db`）のみで高速なハイブリッド検索（BM25キーワード検索 + ベクトルセマンティック検索）を提供するローカルCLIツールです。

---

## 主な特徴

- **Zero-Daemon / Zero-Container**: 外部サービスやDockerコンテナ不要。単一の SQLite データベースファイルのみで完結。
- **Fast Startup / Low Resource**: PyTorch等を使わず、ONNX Runtime ベースの `fastembed` を採用し、高速なCLI起動を実現。
- **日本語完全対応**: 外部形態素解析エンジンのインストールが不要な SQLite FTS5 `trigram` トークナイザーを採用。
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

# パッケージおよび依存モジュールのインストール
pip install -e .

# 開発用ツール（pytest, ruff, mypy 等）も含める場合
pip install -e ".[dev]"
```

---

## 使い方 (CLI コマンド)

### 1. ドキュメントのインデックス作成・更新 (`index`)
指定ディレクトリ配下のドキュメントおよびソースコード（デフォルト: `.md`, `.markdown`, `.txt`, `.py`, `.js`, `.ts`）を再帰的に走査し、インデックスを作成します。SHA-256 ハッシュによる差分検知を行い、変更のあったファイルのみ更新します。

```bash
# カレントディレクトリ配下をインデックス化 (デフォルト拡張子すべて)
hysearch index .

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
現在のインデックス状態（登録ファイル数、チャンク総数、DBファイルサイズ、最終更新時刻）を表示します。

```bash
hysearch status

# JSON 形式で出力
hysearch status --json
```

### 4. インデックスのリセット (`reset`)
データベースファイルを削除し、インデックスを初期化します。

```bash
# 確認プロンプト付きでリセット
hysearch reset

# 確認プロンプトをスキップして強制削除
hysearch reset -y
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
