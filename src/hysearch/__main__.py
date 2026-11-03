"""Entry point for running hysearch as a module.

This module enables running the CLI via `python -m hysearch`.
"""

from hysearch.cli.main import app

if __name__ == "__main__":
    app(prog_name="hysearch")
