import pytest
from src.core.cache import ConditionCache
from src.core.prompt_database import promptDatabase
from unittest.mock import MagicMock, patch
from src.core.llm_handler import llmHandler
from src.core.yaml_parser import applicationParser, behaviourParser
from src.core.atom_list import atomList

@pytest.fixture
def mock_configs():
    app = MagicMock(spec=applicationParser)
    app.context = "app_context"
    app.predicates = []
    
    beh = MagicMock(spec=behaviourParser)
    beh.init = "system_prompt"
    beh.mapping = "mapping {input} {instructions} {atom}"
    beh.context = "context {context}"
    return app, beh

@patch("ollama.Client")
@patch("src.core.prompt_database.promptDatabase.initialize")
def test_llm_handler_factory(mock_db_init, mock_ollama, mock_configs):
    app_cfg, beh_cfg = mock_configs
    
    # Test Factory Method
    handler = llmHandler.create(
        llm_model="test-model",
        system_prompt="sys",
        application_cfg=app_cfg,
        behaviour_cfg=beh_cfg
    )
    
    assert handler.llm_model == "test-model"
    assert mock_ollama.called
    assert mock_db_init.called