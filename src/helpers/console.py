"""
src/helpers/console.py
─────────────────────────────────────────────────────────────────────────────
Module-level Rich console and logging integration.

Usage anywhere in the project:

    from src.helpers.console import console, get_logger

    log = get_logger(__name__)
    log.info("Starting inference …")
    console.rule("[bold cyan]LGX[/]")
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install as install_rich_traceback

from src.config import LOG_LEVEL, LOG_RICH_TRACEBACK, CONSOLE_THEME

# ── Theme ─────────────────────────────────────────────────────────────────────
_DARK_THEME = Theme(
    {
        "info":       "bold cyan",
        "warning":    "bold yellow",
        "error":      "bold red",
        "success":    "bold green",
        "predicate":  "bold magenta",
        "atom":       "dim white",
        "llm":        "bold blue",
        "cache.hit":  "green",
        "cache.miss": "yellow",
    }
)

_LIGHT_THEME = Theme(
    {
        "info":       "blue",
        "warning":    "dark_orange",
        "error":      "red",
        "success":    "dark_green",
        "predicate":  "purple",
        "atom":       "grey50",
        "llm":        "navy_blue",
        "cache.hit":  "dark_green",
        "cache.miss": "dark_orange",
    }
)

console: Console = Console(
    theme=_DARK_THEME if CONSOLE_THEME != "light" else _LIGHT_THEME,
    highlight=True,
)

# ── Rich tracebacks ───────────────────────────────────────────────────────────
if LOG_RICH_TRACEBACK:
    install_rich_traceback(show_locals=False, console=console)


# ── Logging integration ───────────────────────────────────────────────────────
def _configure_root_logger() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=LOG_RICH_TRACEBACK,
                show_path=False,
                markup=True,
            )
        ],
        force=True,   # override any prior basicConfig call
    )


_configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Return a standard Logger whose output is routed through the Rich handler."""
    return logging.getLogger(name)