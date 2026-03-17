"""
tests/test_handler.py
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.atom import atom
from src.core.atom_list import atomList
from src.core.cache import ConditionCache
from src.core.llm_handler import llmHandler
from src.core.predicate import predicate
from src.core.knowledge_base import knowledgeBase, conditionProgram


class TestLlmHandlerFactory:

    @patch("ollama.Client")
    @patch("src.core.prompt_database.promptDatabase.initialize")
    def test_creates_handler_with_prompt_cache(
        self, mock_db_init, mock_ollama, mock_app_cfg, mock_beh_cfg, monkeypatch
    ):
        monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE", "true")
        mock_db_init.return_value = MagicMock()

        handler = llmHandler.create("test-model", mock_app_cfg, mock_beh_cfg)

        assert handler.llm_model == "test-model"
        assert handler.prompt_database is not None
        mock_db_init.assert_called_once()

    @patch("ollama.Client")
    @patch("src.core.prompt_database.promptDatabase.initialize")
    def test_creates_handler_without_prompt_cache(
        self, mock_db_init, mock_ollama, mock_app_cfg, mock_beh_cfg, monkeypatch
    ):
        monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE", "false")

        handler = llmHandler.create("test-model", mock_app_cfg, mock_beh_cfg)

        assert handler.prompt_database is None
        mock_db_init.assert_not_called()

    @patch("ollama.Client")
    def test_ollama_uses_env_url(
        self, mock_ollama_cls, mock_app_cfg, mock_beh_cfg, monkeypatch
    ):
        monkeypatch.setenv("LGX_OLLAMA_URL",          "http://custom:9999")
        monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE", "false")

        llmHandler.create("m", mock_app_cfg, mock_beh_cfg)

        call_kwargs = mock_ollama_cls.call_args.kwargs
        assert call_kwargs["host"] == "http://custom:9999"

    @patch("ollama.Client")
    def test_ollama_auth_header_set_when_api_key_present(
        self, mock_ollama_cls, mock_app_cfg, mock_beh_cfg, monkeypatch
    ):
        monkeypatch.setenv("LGX_OLLAMA_API_KEY",      "secret-key")
        monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE", "false")

        llmHandler.create("m", mock_app_cfg, mock_beh_cfg)

        headers = mock_ollama_cls.call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer secret-key"

    @patch("ollama.Client")
    def test_condition_cache_mode_from_env(
        self, mock_ollama_cls, mock_app_cfg, mock_beh_cfg, monkeypatch
    ):
        monkeypatch.setenv("LGX_CONDITION_CACHE_MODE", "monotone")
        monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE",  "false")

        handler = llmHandler.create("m", mock_app_cfg, mock_beh_cfg)

        assert handler.predicate_condition_cache.cache_mode == "monotone"


class TestCraftMessageHistory:

    @patch("ollama.Client")
    def test_builds_correct_history(self, _, mock_app_cfg, mock_beh_cfg, monkeypatch):
        monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE", "false")
        handler = llmHandler.create("m", mock_app_cfg, mock_beh_cfg)

        history = handler.craft_message_history("hello", " extra")

        assert history[0]["role"] == "system"
        assert "extra" in history[0]["content"]
        assert history[1]["role"] == "user"
        assert history[1]["content"] == "hello"


class TestInvokeLlmConstrained:

    def _make_handler(self, mock_app_cfg, mock_beh_cfg, mock_ollama_client, monkeypatch):
        monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE", "false")
        with patch("ollama.Client") as mock_cls:
            mock_cls.return_value = mock_ollama_client
            return llmHandler.create("m", mock_app_cfg, mock_beh_cfg)

    def test_returns_none_on_llm_error(
        self, mock_app_cfg, mock_beh_cfg, mock_ollama_client, monkeypatch
    ):
        mock_ollama_client.chat.side_effect = RuntimeError("connection refused")
        handler = self._make_handler(mock_app_cfg, mock_beh_cfg, mock_ollama_client, monkeypatch)

        class_response = MagicMock()
        class_response.model_json_schema.return_value = {}
        result = handler.invoke_llm_constrained("prompt", class_response, "")

        assert result is None

    def test_uses_cache_on_hit(
        self, mock_app_cfg, mock_beh_cfg, mock_ollama_client, monkeypatch
    ):
        monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE", "true")
        mock_db = MagicMock()
        mock_db.get_cached_response.return_value = ('{"x": 1}', 5, 5, 100)

        class_response = MagicMock()
        class_response.model_json_schema.return_value = {}
        class_response.model_validate_json.return_value = object()

        with patch("ollama.Client"):
            with patch(
                "src.core.prompt_database.promptDatabase.initialize",
                return_value=mock_db,
            ):
                handler = llmHandler.create("m", mock_app_cfg, mock_beh_cfg)

        handler.invoke_llm_constrained("p", class_response, "")
        mock_ollama_client.chat.assert_not_called()

class TestRunPipeline:

    @patch("ollama.Client")
    def test_run_accumulates_atoms(
        self, mock_ollama_cls, mock_beh_cfg, monkeypatch
    ):
        monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE", "false")

        # Build a mock predicate that yields one atom
        mock_pred = MagicMock(spec=predicate)
        mock_pred.has_condition.return_value = False
        mock_pred.prompt_description = "Extract something."
        mock_pred.predicate_formatted = "{}"
        mock_pred.parse_response.return_value = ["node(a)."]
        mock_pred.execute_knowledge.return_value = atomList(atoms=[atom("node(a).")])

        from src.core.yaml_parser import application_configuration
        app_cfg = application_configuration(
            context=None,
            strings=[],
            predicates=[mock_pred],
            kb=None,
        )

        with patch("src.helpers.filter_atoms.filter_asp_atoms") as mock_filter:
            mock_filter.return_value = atomList(atoms=[atom("node(a).")])
            handler = llmHandler.create("m", app_cfg, mock_beh_cfg)

            # Patch invoke to return a fake response
            with patch.object(handler, "invoke_llm_constrained", return_value=MagicMock()):
                result = handler.run("test prompt")

        assert result.not_empty()