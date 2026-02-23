import yaml
from typing import Optional

from dataclasses import dataclass, field
from src.core.predicate import predicate
from typeguard   import typechecked

class yamlParser:

    @staticmethod
    def parse(file_path: str) -> dict:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
            return data
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File not found: {file_path}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}") from exc



@typechecked
@dataclass(frozen=True)
class application_configuration:
    context: Optional[str] = field(default=None)
    strings: Optional[str] = field(default=None)

    predicates: list[predicate] = field(default=None) 
    kb: str = field(default=None)

@typechecked
@dataclass(frozen=True)
class behaviour_configuration:
    init:    str 
    context: str
    mapping: str

class applicationParser(yamlParser):
    @classmethod
    def from_yaml(cls, file_path: str) -> "application_configuration":
        data = cls.parse(file_path)

        assert "context" in data, "Missing 'context' section in YAML file."
        assert "strings" in data, "Missing 'strings' section in YAML file."
        assert "extract" in data, "Missing 'extract' section in YAML file."
        assert "knowledge_base" in data, "Missing 'knowledge_base' section in YAML file."

        # Create predicate instances from the 'extract' section
        predicates = [predicate.create(predicate_definition=pred_def, strings=data.get("strings", [])) for pred_def in data["extract"]]

        return application_configuration(
            context=data.get("context"),
            strings=data.get("strings"),
            predicates=predicates,
            kb=data.get("kb"),
        )
            
class behaviourParser(yamlParser):
    @classmethod
    def from_yaml(cls, file_path: str) -> "behaviour_configuration":
        data = cls.parse(file_path)

        assert isinstance(data, dict), "YAML file must contain a dictionary at the top level."

        assert "preprocessing" in data, "Missing 'preprocessing' section in YAML file."

        assert "init"    in data["preprocessing"], "Missing 'init' section in YAML file."
        assert "context" in data["preprocessing"], "Missing 'context' section in YAML file."
        assert "mapping" in data["preprocessing"], "Missing 'mapping' section in YAML file."

        return behaviour_configuration(
            init=data["preprocessing"].get("init"),
            context=data["preprocessing"].get("context"),
            mapping=data["preprocessing"].get("mapping"),
        )