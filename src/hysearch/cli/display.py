"""Rich console formatting and terminal displays for hysearch CLI.

This module formats indexing progress, search result tables, and
index status summaries for human-friendly terminal viewing.
"""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hysearch.core.models import IndexingStats, SearchResponse

console = Console()


def print_indexing_stats(stats: IndexingStats) -> None:
    """Print indexing statistics summary using Rich panel.

    Args:
        stats: Completed indexing run statistics.
    """
    text = (
        f"[bold green]Files Scanned:[/] {stats.files_scanned}\n"
        f"[bold cyan]Files Indexed:[/] {stats.files_indexed}\n"
        f"[bold yellow]Files Skipped:[/] {stats.files_skipped}\n"
        f"[bold magenta]Chunks Created:[/] {stats.chunks_created}\n"
        f"[bold white]Elapsed Time:[/] {stats.elapsed_seconds:.3f} s"
    )
    panel = Panel(text, title="Indexing Complete", expand=False)
    console.print(panel)


def print_search_results(response: SearchResponse) -> None:
    """Print ranked search hits in a formatted Rich table.

    Args:
        response: Search response containing query hits and timings.
    """
    if response.total_hits == 0:
        console.print(
            "[yellow]No results found for query:[/] "
            f"[bold]'{response.query}'[/]"
        )
        return

    duration = f"{response.execution_time_ms:.1f} ms"
    summary = f"({response.total_hits} hits, {duration})"
    table = Table(
        title=f"Query: '{response.query}' {summary}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", justify="right", style="green", width=10)
    table.add_column("File", style="blue")
    table.add_column("Ranks (BM25 / Vec)", justify="center", width=18)
    table.add_column("Snippet", style="white")

    for i, hit in enumerate(response.hits, start=1):
        b_rank = str(hit.bm25_rank) if hit.bm25_rank is not None else "-"
        v_rank = str(hit.vec_rank) if hit.vec_rank is not None else "-"
        ranks_str = f"{b_rank} / {v_rank}"
        snippet_text = hit.snippet or hit.content[:100]

        table.add_row(
            str(i),
            f"{hit.rrf_score:.5f}",
            hit.file_path,
            ranks_str,
            snippet_text,
        )

    console.print(table)


def print_status(status_info: dict[str, Any]) -> None:
    """Print index database status summary.

    Args:
        status_info: Dictionary containing status fields.
    """
    table = Table(
        title="Index Status", show_header=True, header_style="bold cyan"
    )
    table.add_column("Property", style="bold")
    table.add_column("Value", style="green")

    for key, value in status_info.items():
        table.add_row(str(key), str(value))

    console.print(table)
