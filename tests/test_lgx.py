"""
tests/test_lgx.py
─────────────────────────────────────────────────────────────────────────────
Tests for the lgx public façade.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.atom import atom
from src.core.atom_list import atomList
from src.core.knowledge_base import knowledgeBase
from src.lgx import lgx


# ── Helpers ───────────────────────────────────────────────────────────────────

def _patched_lgx(mock_app_cfg, mock_beh_cfg, monkeypatch) -> lgx:
    """Build an lgx instance with all external I/O mocked out."""
    monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE", "false")
    with (
        patch("src.lgx.applicationParser.from_yaml", return_value=mock_app_cfg),
        patch("src.lgx.behaviourParser.from_yaml",   return_value=mock_beh_cfg),
        patch("ollama.Client"),
    ):
        return lgx.create(
            llm_model="test-model",
            behaviour_filename="beh.yml",
            application_filename="app.yml",
        )


# ── Factory ───────────────────────────────────────────────────────────────────

class TestLgxFactory:

    def test_create_returns_lgx_instance(self, mock_app_cfg, mock_beh_cfg, monkeypatch):
        instance = _patched_lgx(mock_app_cfg, mock_beh_cfg, monkeypatch)
        assert isinstance(instance, lgx)

    def test_create_raises_on_missing_model(self, mock_app_cfg, mock_beh_cfg, monkeypatch):
        monkeypatch.delenv("LGX_LLM_MODEL", raising=False)
        with pytest.raises(ValueError, match="llm_model"):
            with (
                patch("src.lgx.applicationParser.from_yaml", return_value=mock_app_cfg),
                patch("src.lgx.behaviourParser.from_yaml",   return_value=mock_beh_cfg),
            ):
                lgx.create(
                    llm_model="",
                    behaviour_filename="beh.yml",
                    application_filename="app.yml",
                )

    def test_create_raises_on_missing_behaviour_filename(
        self, mock_app_cfg, mock_beh_cfg, monkeypatch
    ):
        monkeypatch.delenv("LGX_BEHAVIOUR_YAML", raising=False)
        with pytest.raises(ValueError, match="behaviour_filename"):
            lgx.create(llm_model="m", behaviour_filename="", application_filename="a.yml")

    def test_create_raises_on_missing_application_filename(
        self, mock_app_cfg, mock_beh_cfg, monkeypatch
    ):
        monkeypatch.delenv("LGX_APPLICATION_YAML", raising=False)
        with pytest.raises(ValueError, match="application_filename"):
            lgx.create(llm_model="m", behaviour_filename="b.yml", application_filename="")

    def test_create_reads_model_from_env(self, mock_app_cfg, mock_beh_cfg, monkeypatch):
        monkeypatch.setenv("LGX_LLM_MODEL",        "env-model")
        monkeypatch.setenv("LGX_APPLICATION_YAML", "app.yml")
        monkeypatch.setenv("LGX_BEHAVIOUR_YAML",   "beh.yml")
        monkeypatch.setenv("LGX_ENABLE_PROMPT_CACHE", "false")
        with (
            patch("src.lgx.applicationParser.from_yaml", return_value=mock_app_cfg),
            patch("src.lgx.behaviourParser.from_yaml",   return_value=mock_beh_cfg),
            patch("ollama.Client"),
        ):
            instance = lgx.create()
        assert instance.llm_instance.llm_model == "env-model"


# ── infer ─────────────────────────────────────────────────────────────────────

class TestLgxInfer:

    def test_infer_returns_self_for_chaining(self, mock_app_cfg, mock_beh_cfg, monkeypatch):
        instance = _patched_lgx(mock_app_cfg, mock_beh_cfg, monkeypatch)
        with patch.object(instance.llm_instance, "run", return_value=atomList(atoms=[])):
            result = instance.infer("test prompt")
        assert result is instance

    def test_infer_delegates_to_llm_run(self, mock_app_cfg, mock_beh_cfg, monkeypatch):
        instance = _patched_lgx(mock_app_cfg, mock_beh_cfg, monkeypatch)
        with patch.object(instance.llm_instance, "run") as mock_run:
            instance.infer("hello world")
        mock_run.assert_called_once_with("hello world")


# ── get_extracted_atoms ───────────────────────────────────────────────────────

class TestLgxGetExtractedAtoms:

    def test_returns_atom_list(self, mock_app_cfg, mock_beh_cfg, monkeypatch):
        instance = _patched_lgx(mock_app_cfg, mock_beh_cfg, monkeypatch)
        result   = instance.get_extracted_atoms()
        assert isinstance(result, atomList)

    def test_returns_atoms_after_infer(self, mock_app_cfg, mock_beh_cfg, monkeypatch):
        instance   = _patched_lgx(mock_app_cfg, mock_beh_cfg, monkeypatch)
        expected   = atomList(atoms=[atom("node(x).")])
        with patch.object(instance.llm_instance, "run", return_value=expected):
            with patch.object(instance.llm_instance, "get_extracted_atoms", return_value=expected):
                instance.infer("p")
                result = instance.get_extracted_atoms()
        assert result is expected


# ── execute_knowledge_base ────────────────────────────────────────────────────

class TestLgxExecuteKnowledgeBase:

    def test_delegates_to_kb_execute(self, mock_app_cfg, mock_beh_cfg, monkeypatch):
        instance    = _patched_lgx(mock_app_cfg, mock_beh_cfg, monkeypatch)
        mock_kb     = MagicMock(spec=knowledgeBase)
        mock_kb.execute.return_value = atomList(atoms=[atom("out(1).")])
        result = instance.execute_knowledge_base(mock_kb)
        mock_kb.execute.assert_called_once()
        assert result.atoms[0].atom_str == "out(1)."


# ── cleanup ───────────────────────────────────────────────────────────────────

def test_cleanup_delegates(mock_app_cfg, mock_beh_cfg, monkeypatch):
    instance = _patched_lgx(mock_app_cfg, mock_beh_cfg, monkeypatch)
    with patch.object(instance.llm_instance, "cleanup") as mock_cleanup:
        instance.cleanup()
    mock_cleanup.assert_called_once()


# ── explain (not implemented) ─────────────────────────────────────────────────

def test_explain_raises_not_implemented(mock_app_cfg, mock_beh_cfg, monkeypatch):
    instance = _patched_lgx(mock_app_cfg, mock_beh_cfg, monkeypatch)
    with pytest.raises(NotImplementedError):
        instance.explain()