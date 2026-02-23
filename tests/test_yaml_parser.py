import pytest
from unittest.mock import patch, mock_open
from src.core.yaml_parser import yamlParser

def test_yaml_parser_not_found():
    with pytest.raises(FileNotFoundError):
        yamlParser.parse("non_existent.yml")

@patch("builtins.open", new_callable=mock_open, read_data="key: value")
def test_yaml_parser_success(mock_file):
    data = yamlParser.parse("fake.yml")
    assert data == {"key": "value"}