"""
tests/test_yaml_parser.py
"""
from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from src.core.yaml_parser import (
    applicationParser,
    behaviourParser,
    yamlParser,
)


# ── yamlParser ────────────────────────────────────────────────────────────────

class TestYamlParser:

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            yamlParser.parse("does_not_exist.yml")

    @patch("builtins.open", new_callable=mock_open, read_data="key: value\nnested:\n  a: 1")
    def test_parses_valid_yaml(self, _):
        data = yamlParser.parse("any.yml")
        assert data == {"key": "value", "nested": {"a": 1}}

    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_empty_file_returns_empty_dict(self, _):
        assert yamlParser.parse("empty.yml") == {}

    @patch("builtins.open", new_callable=mock_open, read_data=": : invalid: yaml: [")
    def test_invalid_yaml_raises_value_error(self, _):
        with pytest.raises(ValueError, match="YAML parse error"):
            yamlParser.parse("bad.yml")


# ── applicationParser ─────────────────────────────────────────────────────────

_VALID_APP_YAML = """
context: "test context"
strings:
  - text: "hello"
extract:
  - person: "Extract person names."
knowledge_base: null
"""

_APP_MISSING_EXTRACT = """
context: null
strings: []
knowledge_base: null
"""


class TestApplicationParser:

    @patch("builtins.open", new_callable=mock_open, read_data=_VALID_APP_YAML)
    @patch("src.core.predicate.JSONSchemaBuilder")
    def test_valid_yaml_returns_config(self, mock_jsb, _):
        grammar_mock = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        grammar_mock.__info__ = {"list_person": [{"name": "any"}]}
        mock_jsb.return_value.generate.return_value = grammar_mock

        cfg = applicationParser.from_yaml("app.yml")

        assert cfg.context == "test context"
        assert len(cfg.predicates) == 1

    @patch("builtins.open", new_callable=mock_open, read_data=_APP_MISSING_EXTRACT)
    def test_missing_extract_raises_key_error(self, _):
        with pytest.raises(KeyError, match="extract"):
            applicationParser.from_yaml("bad_app.yml")

    def test_file_not_found_propagates(self):
        with pytest.raises(FileNotFoundError):
            applicationParser.from_yaml("no_such_file.yml")


# ── behaviourParser ───────────────────────────────────────────────────────────

_VALID_BEH_YAML = """
preprocessing:
  init: "You are an assistant."
  context: "Context: {context}"
  mapping: "{input} {instructions} {atom}"
"""

_BEH_MISSING_MAPPING = """
preprocessing:
  init: "You are an assistant."
  context: "Context: {context}"
"""


class TestBehaviourParser:

    @patch("builtins.open", new_callable=mock_open, read_data=_VALID_BEH_YAML)
    def test_valid_yaml_returns_config(self, _):
        cfg = behaviourParser.from_yaml("beh.yml")
        assert cfg.init    == "You are an assistant."
        assert "{context}" in cfg.context
        assert "{input}"   in cfg.mapping

    @patch("builtins.open", new_callable=mock_open, read_data=_BEH_MISSING_MAPPING)
    def test_missing_mapping_raises_key_error(self, _):
        with pytest.raises(KeyError, match="mapping"):
            behaviourParser.from_yaml("bad_beh.yml")

    @patch("builtins.open", new_callable=mock_open, read_data="- not a dict")
    def test_non_dict_yaml_raises_type_error(self, _):
        with pytest.raises(TypeError):
            behaviourParser.from_yaml("list.yml")

    def test_file_not_found_propagates(self):
        with pytest.raises(FileNotFoundError):
            behaviourParser.from_yaml("no_such_file.yml")