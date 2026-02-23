import pytest
from unittest.mock import patch, MagicMock
from src.lgx import lgx

@patch("src.core.yaml_parser.applicationParser.from_yaml")
@patch("src.core.yaml_parser.behaviourParser.from_yaml")
@patch("src.core.llm_handler.llmHandler.run")
def test_lgx_full_flow(mock_run, mock_beh, mock_app):
    # Setup mock configs
    mock_app.return_value = MagicMock()
    mock_beh.return_value = MagicMock()
    mock_run.return_value = MagicMock()
    
    # Create instance
    system = lgx.create(
        llm_model="llama3",
        behaviour_filename="beh.yml",
        application_filename="app.yml"
    )
    
    # Run inference
    system.infer("What is the capital of Italy?")
    
    assert mock_run.called
    assert isinstance(system, lgx)