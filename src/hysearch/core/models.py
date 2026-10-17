"""Data models and type definitions for hysearch.

This module defines Pydantic data structures used across the indexing
and search pipelines.
"""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a chunk of text extracted from a document."""

    chunk_index: int = Field(
        ..., description="0-based sequential index of the chunk in the file"
    )
    content: str = Field(..., description="Raw text content of the chunk")
    start_char: int = Field(
        ..., description="Start character offset in the source document"
    )
    end_char: int = Field(
        ..., description="End character offset in the source document"
    )


class SearchHit(BaseModel):
    """Represents a single search hit returned by hybrid search."""

    chunk_id: int = Field(..., description="Database ID of the chunk")
    file_path: str = Field(
        ..., description="Path to the source file of the chunk"
    )
    chunk_index: int = Field(
        ..., description="0-based chunk index in the source file"
    )
    content: str = Field(..., description="Raw text content of the chunk")
    rrf_score: float = Field(
        ..., description="Combined Reciprocal Rank Fusion score"
    )
    bm25_rank: int | None = Field(
        default=None, description="Rank from BM25 full-text search (1-based)"
    )
    vec_rank: int | None = Field(
        default=None, description="Rank from vector search (1-based)"
    )
    snippet: str | None = Field(
        default=None, description="Highlighted or trimmed text snippet"
    )


class SearchResponse(BaseModel):
    """Encapsulates the full response to a search query."""

    query: str = Field(..., description="Query string used for search")
    total_hits: int = Field(
        ..., description="Total number of hits returned in this response"
    )
    hits: list[SearchHit] = Field(
        default_factory=list, description="List of ranked search hits"
    )
    execution_time_ms: float = Field(
        ..., description="Search execution duration in milliseconds"
    )


class IndexingStats(BaseModel):
    """Statistics gathered during an indexing run."""

    files_scanned: int = Field(
        default=0, description="Total files discovered during scanning"
    )
    files_indexed: int = Field(
        default=0, description="Number of files newly or updated and indexed"
    )
    files_skipped: int = Field(
        default=0, description="Number of files skipped because unchanged"
    )
    chunks_created: int = Field(
        default=0, description="Total chunks extracted and stored"
    )
    elapsed_seconds: float = Field(
        default=0.0, description="Total time taken for indexing in seconds"
    )
    tokenizer: str = Field(
        default="trigram",
        description="Tokenizer name used for FTS indexing",
    )
