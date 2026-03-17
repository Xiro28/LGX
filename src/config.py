"""
src/config.py
─────────────────────────────────────────────────────────────────────────────
Centralised, module-level configuration loaded from environment variables
(and optionally from a .env file via python-dotenv).

Import individual constants rather than the module to make usage explicit:

    from src.config import OLLAMA_URL, ENABLE_PROMPT_CACHE
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # .env values only fill *missing* vars
except ImportError:
    pass  # python-dotenv is optional; plain os.getenv fallbacks apply below

# ── Ollama / LLM ─────────────────────────────────────────────────────────────
OLLAMA_URL: str        = os.getenv("LGX_OLLAMA_URL",       "http://localhost:11434")
OLLAMA_API_KEY: str    = os.getenv("LGX_OLLAMA_API_KEY",   "")
LLM_MODEL: str         = os.getenv("LGX_LLM_MODEL",        "llama3.1:8b")
LLM_TEMPERATURE: int   = int(os.getenv("LGX_LLM_TEMPERATURE", "0"))
LLM_MAX_RETRIES: int   = int(os.getenv("LGX_LLM_MAX_RETRIES", "3"))

# ── Prompt / response cache ───────────────────────────────────────────────────
ENABLE_PROMPT_CACHE: bool = os.getenv("LGX_ENABLE_PROMPT_CACHE", "true").lower() == "true"
DB_FILENAME: str           = os.getenv("LGX_DB_FILENAME",         "cached_prompts.db")

# ── Condition cache ───────────────────────────────────────────────────────────
ENABLE_CONDITION_CACHE: bool = os.getenv("LGX_ENABLE_CONDITION_CACHE", "true").lower() == "true"
CONDITION_CACHE_MODE: str    = os.getenv("LGX_CONDITION_CACHE_MODE",   "all").lower()

# ── File paths ────────────────────────────────────────────────────────────────
APPLICATION_YAML: str = os.getenv("LGX_APPLICATION_YAML", "")
BEHAVIOUR_YAML: str   = os.getenv("LGX_BEHAVIOUR_YAML",   "")

# ── Logging / console ─────────────────────────────────────────────────────────
LOG_LEVEL: str           = os.getenv("LGX_LOG_LEVEL",           "INFO").upper()
LOG_RICH_TRACEBACK: bool = os.getenv("LGX_LOG_RICH_TRACEBACK",  "true").lower() == "true"
CONSOLE_THEME: str       = os.getenv("LGX_CONSOLE_THEME",       "dark")   # "dark" | "light"

# ── Progress bar ──────────────────────────────────────────────────────────────
TQDM_DISABLE: bool  = os.getenv("LGX_TQDM_DISABLE",  "false").lower() == "true"
TQDM_NCOLS: int     = int(os.getenv("LGX_TQDM_NCOLS",  "100"))
TQDM_COLOUR: str    = os.getenv("LGX_TQDM_COLOUR",   "cyan")