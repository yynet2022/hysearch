"""Configuration and constant values for hysearch.

This module centralizes default configurations, embedding settings,
and indexing parameters.
"""

from pathlib import Path

# デフォルトのデータベースファイルパス
DEFAULT_DB_PATH: Path = Path(".hysearch.db")

# デフォルトのテキスト埋め込みモデル (FastEmbed対応)
DEFAULT_EMBED_MODEL: str = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 埋め込みベクトルの次元数 (デフォルト)
DEFAULT_EMBED_DIM: int = 384

# 選択可能なテキスト埋め込みモデル定数
EMBED_MODEL_MINILM: str = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBED_MODEL_MPNET: str = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
EMBED_MODEL_QWEN3: str = "Qwen/Qwen3-Embedding-0.6B-Q"
EMBED_MODEL_GEMMA: str = "google/embeddinggemma-300m"

# サポートする埋め込みモデル名と次元数のマッピング
SUPPORTED_EMBED_MODELS: dict[str, int] = {
    EMBED_MODEL_MINILM: 384,
    EMBED_MODEL_MPNET: 768,
    EMBED_MODEL_QWEN3: 1024,
    EMBED_MODEL_GEMMA: 768,
}

VALID_EMBED_MODELS: tuple[str, ...] = tuple(SUPPORTED_EMBED_MODELS.keys())

# チャンキング設定
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 100

# RRF (Reciprocal Rank Fusion) スコア計算のデフォルト設定
DEFAULT_RRF_K: int = 60
DEFAULT_WEIGHT_BM25: float = 1.0
DEFAULT_WEIGHT_VEC: float = 1.0

# 対象とするファイル拡張子 (デフォルト)
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".md",
    ".markdown",
    ".txt",
    ".py",
    ".js",
    ".ts",
)

# FTS トークナイザー設定
TOKENIZER_TRIGRAM: str = "trigram"
TOKENIZER_SUDACHI_SMALL: str = "sudachi-small"
TOKENIZER_SUDACHI_CORE: str = "sudachi-core"
TOKENIZER_SUDACHI_FULL: str = "sudachi-full"
TOKENIZER_SUDACHI_AUTO: str = "sudachi"

SUDACHI_DICT_PRIORITY: tuple[str, ...] = ("full", "core", "small")

VALID_TOKENIZERS: tuple[str, ...] = (
    TOKENIZER_SUDACHI_FULL,
    TOKENIZER_SUDACHI_CORE,
    TOKENIZER_SUDACHI_SMALL,
    TOKENIZER_SUDACHI_AUTO,
    TOKENIZER_TRIGRAM,
)

# Web サーバー設定
DEFAULT_SERVER_HOST: str = "127.0.0.1"
DEFAULT_SERVER_PORT: int = 8888
MAX_PORT_SEARCH_TRIES: int = 100
