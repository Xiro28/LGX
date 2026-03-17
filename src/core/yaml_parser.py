from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import yaml
from typeguard import typechecked

from src.core.predicate import predicate
from src.helpers.console import get_logger

log = get_logger(__name__)

_APP_REQUIRED  = ("context", "strings", "extract", "knowledge_base")
_BEH_REQUIRED  = ("init", "context", "mapping")


class yamlParser:
    @staticmethod
    def parse(file_path: str) -> dict:
        log.debug(f"[info]Parsing YAML[/] [bold]{file_path}[/]")
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            return data
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"YAML file not found: {file_path}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"YAML parse error in {file_path}: {exc}") from exc



@typechecked
@dataclass(frozen=True)
class application_configuration:
    context:    Optional[str]           = field(default=None)
    strings:    Optional[list]          = field(default=None)
    predicates: Optional[List[predicate]] = field(default=None)
    kb:         Optional[str]           = field(default=None)


@typechecked
@dataclass(frozen=True)
class behaviour_configuration:
    init:    str
    context: str
    mapping: str



class applicationParser(yamlParser):
    @classmethod
    def from_yaml(cls, file_path: str) -> application_configuration:
        data = cls.parse(file_path)

        missing = [k for k in _APP_REQUIRED if k not in data]
        if missing:
            raise KeyError(f"Missing keys in application YAML {file_path}: {missing}")

        strings    = data.get("strings") or []
        predicates = [
            predicate.create(predicate_definition=pd, strings=strings)
            for pd in data["extract"]
        ]

        log.info(
            f"[info]Application config loaded[/] predicates=[bold]{len(predicates)}[/]"
            f" from [bold]{file_path}[/]"
        )
        return application_configuration(
            context=data.get("context"),
            strings=strings,
            predicates=predicates,
            kb=data.get("knowledge_base"),
        )


class behaviourParser(yamlParser):
    @classmethod
    def from_yaml(cls, file_path: str) -> behaviour_configuration:
        data = cls.parse(file_path)

        if not isinstance(data, dict):
            raise TypeError(f"Behaviour YAML must be a dict, got {type(data).__name__}")

        prep = data.get("preprocessing", {})
        missing = [k for k in _BEH_REQUIRED if k not in prep]
        if missing:
            raise KeyError(
                f"Missing keys in behaviour YAML {file_path} → preprocessing: {missing}"
            )

        log.info(f"[info]Behaviour config loaded[/] from [bold]{file_path}[/]")
        return behaviour_configuration(
            init=prep["init"],
            context=prep["context"],
            mapping=prep["mapping"],
        )