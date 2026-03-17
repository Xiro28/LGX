from __future__ import annotations

import importlib

import pytest


def _reload_config(monkeypatch, env: dict[str, str]):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import src.config as cfg
    importlib.reload(cfg)
    return cfg


class TestOllamaConfig:

    def test_default_ollama_url(self, monkeypatch):
        monkeypatch.delenv("LGX_OLLAMA_URL", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.OLLAMA_URL == "http://localhost:11434"

    def test_custom_ollama_url(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"LGX_OLLAMA_URL": "http://myhost:9999"})
        assert cfg.OLLAMA_URL == "http://myhost:9999"

    def test_empty_api_key_by_default(self, monkeypatch):
        monkeypatch.delenv("LGX_OLLAMA_API_KEY", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.OLLAMA_API_KEY == ""

    def test_custom_api_key(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"LGX_OLLAMA_API_KEY": "tok-abc"})
        assert cfg.OLLAMA_API_KEY == "tok-abc"

    def test_temperature_default(self, monkeypatch):
        monkeypatch.delenv("LGX_LLM_TEMPERATURE", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.LLM_TEMPERATURE == 0
        assert isinstance(cfg.LLM_TEMPERATURE, int)

    def test_temperature_custom(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"LGX_LLM_TEMPERATURE": "1"})
        assert cfg.LLM_TEMPERATURE == 1


class TestPromptCacheConfig:

    def test_cache_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("LGX_ENABLE_PROMPT_CACHE", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.ENABLE_PROMPT_CACHE is True

    @pytest.mark.parametrize("raw, expected", [
        ("true",  True),
        ("True",  True),
        ("TRUE",  True),
        ("false", False),
        ("False", False),
        ("0",     False),   # any non-"true" string → False
    ])
    def test_cache_flag_parsing(self, monkeypatch, raw, expected):
        cfg = _reload_config(monkeypatch, {"LGX_ENABLE_PROMPT_CACHE": raw})
        assert cfg.ENABLE_PROMPT_CACHE is expected

    def test_db_filename_default(self, monkeypatch):
        monkeypatch.delenv("LGX_DB_FILENAME", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.DB_FILENAME == "cached_prompts.db"

    def test_db_filename_custom(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"LGX_DB_FILENAME": "/tmp/my.db"})
        assert cfg.DB_FILENAME == "/tmp/my.db"


class TestConditionCacheConfig:

    def test_condition_cache_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("LGX_ENABLE_CONDITION_CACHE", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.ENABLE_CONDITION_CACHE is True

    @pytest.mark.parametrize("mode", ["all", "monotone", "non_monotone"])
    def test_valid_cache_modes(self, monkeypatch, mode):
        cfg = _reload_config(monkeypatch, {"LGX_CONDITION_CACHE_MODE": mode})
        assert cfg.CONDITION_CACHE_MODE == mode

    def test_cache_mode_lowercased(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"LGX_CONDITION_CACHE_MODE": "ALL"})
        assert cfg.CONDITION_CACHE_MODE == "all"


class TestLoggingConfig:

    def test_log_level_default(self, monkeypatch):
        monkeypatch.delenv("LGX_LOG_LEVEL", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.LOG_LEVEL == "INFO"

    def test_log_level_uppercased(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"LGX_LOG_LEVEL": "debug"})
        assert cfg.LOG_LEVEL == "DEBUG"

    def test_rich_traceback_default(self, monkeypatch):
        monkeypatch.delenv("LGX_LOG_RICH_TRACEBACK", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.LOG_RICH_TRACEBACK is True


class TestTqdmConfig:

    def test_tqdm_disabled_false_by_default(self, monkeypatch):
        monkeypatch.delenv("LGX_TQDM_DISABLE", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.TQDM_DISABLE is False

    def test_tqdm_disable_true(self, monkeypatch):
        cfg = _reload_config(monkeypatch, {"LGX_TQDM_DISABLE": "true"})
        assert cfg.TQDM_DISABLE is True

    def test_tqdm_ncols_default(self, monkeypatch):
        monkeypatch.delenv("LGX_TQDM_NCOLS", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.TQDM_NCOLS == 100
        assert isinstance(cfg.TQDM_NCOLS, int)

    def test_tqdm_colour_default(self, monkeypatch):
        monkeypatch.delenv("LGX_TQDM_COLOUR", raising=False)
        cfg = _reload_config(monkeypatch, {})
        assert cfg.TQDM_COLOUR == "cyan"