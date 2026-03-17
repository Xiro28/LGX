"""
src/lgx.py
─────────────────────────────────────────────────────────────────────────────
Public façade for Logic-Guided eXtraction (LGX).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from typeguard import typechecked

from src.config import APPLICATION_YAML, BEHAVIOUR_YAML, LLM_MODEL
from src.core.atom_list import atomList
from core.knowledge_base import knowledgeBase
from src.core.llm_handler import llmHandler
from src.core.yaml_parser import applicationParser, behaviourParser
from src.helpers.console import console, get_logger

log = get_logger(__name__)


@typechecked
@dataclass(frozen=True)
class lgx:
    llm_instance: llmHandler

    @classmethod
    def create(
        cls,
        llm_model: str = LLM_MODEL,
        behaviour_filename: str = BEHAVIOUR_YAML,
        application_filename: str = APPLICATION_YAML,
    ) -> "lgx":
        """
        Build an ``lgx`` instance from YAML configuration files.

        All three parameters fall back to the corresponding environment
        variables (``LGX_LLM_MODEL``, ``LGX_BEHAVIOUR_YAML``,
        ``LGX_APPLICATION_YAML``) when not supplied explicitly.
        """
        if not behaviour_filename:
            raise ValueError(
                "behaviour_filename is required (or set LGX_BEHAVIOUR_YAML)"
            )
        if not application_filename:
            raise ValueError(
                "application_filename is required (or set LGX_APPLICATION_YAML)"
            )
        if not llm_model:
            raise ValueError("llm_model is required (or set LGX_LLM_MODEL)")

        log.info(
            f"[info]Loading configs[/] app=[bold]{application_filename}[/]"
            f"  beh=[bold]{behaviour_filename}[/]"
        )
        app_cfg = applicationParser.from_yaml(application_filename)
        beh_cfg = behaviourParser.from_yaml(behaviour_filename)

        llm_inst = llmHandler.create(llm_model, app_cfg, beh_cfg)
        return cls(llm_instance=llm_inst)

    # ── Public API ────────────────────────────────────────────────────────────
    def infer(self, prompt: str) -> "lgx":
        """Run the full extraction pipeline and return *self* for chaining."""
        self.llm_instance.run(prompt)
        return self

    def execute_knowledge_base(self, kb: knowledgeBase) -> atomList:
        """Execute an external KB against the currently extracted atoms."""
        return kb.execute(self.llm_instance.get_extracted_atoms())

    def get_extracted_atoms(self) -> atomList:
        return self.llm_instance.get_extracted_atoms()

    def explain(self) -> str:  # pragma: no cover
        raise NotImplementedError("explain() is not implemented yet")

    def cleanup(self) -> None:
        self.llm_instance.cleanup()
        log.info("[success]LGX cleaned up[/]")