"""Text embedding module using FastEmbed and testing mocks for hysearch.

This module provides the FastEmbed wrapper, an abstract EmbedderProtocol,
and a deterministic DummyEmbedder for fast, reproducible testing.
"""

import hashlib
import math
from typing import Protocol

from hysearch.core.config import (
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_MODEL,
    SUPPORTED_EMBED_MODELS,
    VALID_EMBED_MODELS,
)


def resolve_embed_model(requested: str | None = None) -> tuple[str, int]:
    """Resolve and validate the embedding model name and its vector dimension.

    Args:
        requested: Optional user-specified embedding model name. If None,
            defaults to DEFAULT_EMBED_MODEL.

    Returns:
        tuple[str, int]: Canonical model name and vector dimension.

    Raises:
        ValueError: If the requested model name is not supported.
    """
    if requested is None:
        return DEFAULT_EMBED_MODEL, DEFAULT_EMBED_DIM

    stripped = requested.strip()
    if stripped in SUPPORTED_EMBED_MODELS:
        return stripped, SUPPORTED_EMBED_MODELS[stripped]

    lower_map = {k.lower(): (k, v) for k, v in SUPPORTED_EMBED_MODELS.items()}
    if stripped.lower() in lower_map:
        return lower_map[stripped.lower()]

    valid_str = ", ".join(VALID_EMBED_MODELS)
    raise ValueError(
        f"Unsupported embedding model '{requested}'. Choose from: {valid_str}."
    )


class EmbedderProtocol(Protocol):
    """Protocol defining the interface for text embedding providers."""

    @property
    def dimension(self) -> int:
        """Return the dimension of generated embedding vectors."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a batch of text documents.

        Args:
            texts: List of input string documents.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """Compute an embedding vector for a single search query.

        Args:
            query: Query string.

        Returns:
            list[float]: Embedding vector.
        """
        ...


class FastEmbedder:
    """Wrapper around fastembed.TextEmbedding with lazy initialization."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        dimension: int | None = None,
    ) -> None:
        """Initialize FastEmbedder with model configuration.

        Args:
            model_name: Name of the FastEmbed model.
            dimension: Expected vector dimension (auto-resolved if None).
        """
        self.model_name = model_name
        self._dimension = (
            dimension
            if dimension is not None
            else SUPPORTED_EMBED_MODELS.get(model_name, DEFAULT_EMBED_DIM)
        )
        self._model_instance: object | None = None

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def _get_model(self) -> object:
        """Lazily initialize and return FastEmbed TextEmbedding instance."""
        if self._model_instance is None:
            from fastembed import TextEmbedding

            self._model_instance = TextEmbedding(model_name=self.model_name)
        return self._model_instance

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate passage embeddings for a list of texts using FastEmbed.

        Args:
            texts: List of strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        if not texts:
            return []
        model = self._get_model()
        # FastEmbed の passage_embed から numpy 配列を取得し float リストに変換
        raw_embeddings = model.passage_embed(texts)  # type: ignore[attr-defined]
        return [vec.tolist() for vec in raw_embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Generate a query embedding vector for a search query string.

        Args:
            query: Search query text.

        Returns:
            list[float]: Query embedding vector.
        """
        model = self._get_model()
        # FastEmbed の query_embed から単一クエリのベクトルを取得
        raw_embeddings = model.query_embed(query)  # type: ignore[attr-defined]
        return next(vec.tolist() for vec in raw_embeddings)


class DummyEmbedder:
    """Deterministic mock embedder for lightning-fast testing without ONNX."""

    def __init__(
        self,
        dimension: int | None = None,
        fixed_value: float | None = None,
        model_name: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        """Initialize DummyEmbedder.

        Args:
            dimension: Dimensionality of generated vectors.
            fixed_value: Optional constant float value for every dimension.
            model_name: Model name associated with this embedder instance.
        """
        self.model_name = model_name
        self._dimension = (
            dimension
            if dimension is not None
            else SUPPORTED_EMBED_MODELS.get(model_name, DEFAULT_EMBED_DIM)
        )
        self._fixed_value = fixed_value

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._dimension

    def _generate_vector(self, text: str) -> list[float]:
        """Generate a deterministic normalized vector from string content."""
        if self._fixed_value is not None:
            return [self._fixed_value] * self._dimension

        # テキストの MD5 ハッシュ値をシードにして決定論的ベクトルを作成
        h = hashlib.md5(text.encode("utf-8")).digest()
        raw_vals: list[float] = []
        for i in range(self._dimension):
            byte_val = h[i % len(h)]
            # -1.0 から 1.0 の範囲に正規化
            raw_vals.append((byte_val / 127.5) - 1.0 + (i * 0.001))

        # L2 ノルムで正規化
        norm = math.sqrt(sum(v * v for v in raw_vals)) or 1.0
        return [v / norm for v in raw_vals]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic dummy embeddings for testing.

        Args:
            texts: List of strings to embed.

        Returns:
            list[list[float]]: Generated dummy vectors.
        """
        return [self._generate_vector(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """Return a deterministic dummy query vector for testing.

        Args:
            query: Search query string.

        Returns:
            list[float]: Generated dummy query vector.
        """
        return self._generate_vector(query)
