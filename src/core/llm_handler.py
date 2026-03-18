from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import ollama
from typeguard import typechecked

from src.config import (
    OLLAMA_URL,
    OLLAMA_API_KEY,
    ENABLE_PROMPT_CACHE,
    DB_FILENAME,
    ENABLE_CONDITION_CACHE,
    CONDITION_CACHE_MODE,
    LLM_TEMPERATURE
)
from src.core.atom_list import atomList
from src.core.cache import ConditionCache
from src.core.predicate import predicate
from src.core.prompt_database import promptDatabase
from src.core.yaml_parser import application_configuration, behaviour_configuration
from src.helpers.console import console, get_logger
from src.helpers.filter_atoms import filter_asp_atoms

log = get_logger(__name__)


@typechecked
@dataclass(frozen=True)
class llmHandler:
    llm_model: str
    application_config: application_configuration
    behaviour_config: behaviour_configuration
    prompt_database: Optional[promptDatabase]
    predicate_condition_cache: ConditionCache
    _llm_chat_fn: Any

    atom_database: atomList = field(init=False, default_factory=lambda: atomList(atoms=[]))
    statistics: dict = field(init=False, default_factory=dict)

    @classmethod
    def create(
        cls,
        llm_model: str,
        application_cfg: application_configuration,
        behaviour_cfg: behaviour_configuration,
    ) -> "llmHandler":
        console.rule("[bold cyan]LGX — initialising handler[/]")


        db: Optional[promptDatabase] = None
        if ENABLE_PROMPT_CACHE:
            db_path = os.getenv("LGX_DB_FILENAME", DB_FILENAME)
            log.info(f"[info]Prompt cache[/] enabled → [bold]{db_path}[/]")
            db = promptDatabase.initialize(db_path)
        else:
            log.info("[warning]Prompt cache disabled[/]")


        cache_mode = os.getenv("LGX_CONDITION_CACHE_MODE", CONDITION_CACHE_MODE)
        use_condition_cache = (
            os.getenv("LGX_ENABLE_CONDITION_CACHE", str(ENABLE_CONDITION_CACHE)).lower() == "true"
        )
        cache = ConditionCache(use_condition_cache, cache_mode)
        log.info(
            f"[info]Condition cache[/] {'[success]ON[/]' if use_condition_cache else '[warning]OFF[/]'}"
            f" mode=[bold]{cache_mode}[/]"
        )

        headers: dict[str, str] = {}
        api_key = os.getenv("LGX_OLLAMA_API_KEY", OLLAMA_API_KEY)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        host = os.getenv("LGX_OLLAMA_URL", OLLAMA_URL)
        log.info(f"[llm]Ollama[/] connecting to [bold]{host}[/] model=[bold]{llm_model}[/]")
        client = ollama.Client(host=host, headers=headers)

        llm_handler = cls(
            llm_model=llm_model,
            application_config=application_cfg,
            behaviour_config=behaviour_cfg,
            prompt_database=db,
            predicate_condition_cache=cache,
            _llm_chat_fn=client.chat,
        )

        llm_handler.cleanup()  # ensure clean state on init
        return llm_handler

    def __del__(self) -> None:
        if self.prompt_database is not None:
            self.prompt_database.close()

    def cleanup(self) -> None:
        self.atom_database.atoms.clear()
        self.predicate_condition_cache.clear()
        self.statistics.clear()

        self.statistics["llm_calls"] = 0
        self.statistics["total_time"] = 0.0
        self.statistics["total_in_tokens"] = 0
        self.statistics["total_out_tokens"] = 0

        log.debug("[info]Handler cleaned up[/]")

    def craft_message_history(self, prompt: str, context: str) -> list:
        return [
            {"role": "system", "content": f"{self.behaviour_config.init}{context}"},
            {"role": "user",   "content": prompt},
        ]

    def invoke_llm_constrained(
        self, prompt: str, class_response: Any, context: str
    ) -> Optional[Any]:
        configuration = {"temperature": int(os.getenv("LGX_LLM_TEMPERATURE", str(LLM_TEMPERATURE)))}
        grammar  = class_response.model_json_schema(mode="serialization")
        history  = self.craft_message_history(prompt, context)

        if self.prompt_database is not None:
            cached = self.prompt_database.get_cached_response(history, self.llm_model, configuration)
            if cached:
                log.debug("[cache.hit]Cache HIT[/] — skipping LLM call")
                try:
                    json_validated = class_response.model_validate_json(cached[0])
                    log.debug("[cache.valid]Cache response valid JSON[/]")
                    self.statistics["llm_calls"] += 1
                    self.statistics["total_time"] += cached[1]
                    self.statistics["total_in_tokens"] += cached[2]
                    self.statistics["total_out_tokens"] += cached[3]
                    return json_validated
                except Exception as exc:
                    log.warning(f"[warning]Cache parse error:[/] {exc}")

        try:
            log.debug(f"[llm]LLM call[/] model=[bold]{self.llm_model}[/]")
            resp = self._llm_chat_fn(
                model=self.llm_model,
                messages=history,
                options=configuration,
                format=grammar,
            )
            content = resp["message"]["content"]

            if self.prompt_database is not None:
                self.prompt_database.cache_response(
                    history,
                    self.llm_model,
                    configuration,
                    content,
                    resp["prompt_eval_count"],
                    resp["eval_count"],
                    resp["total_duration"],
                )

                self.statistics["llm_calls"] += 1
                self.statistics["total_time"] += resp["total_duration"]
                self.statistics["total_in_tokens"] += resp["prompt_eval_count"]
                self.statistics["total_out_tokens"] += resp["eval_count"]

            return class_response.model_validate_json(content)

        except Exception as exc:
            log.error(f"[error]LLM error:[/] {exc}", exc_info=True)
            return None

    def _structured_output_call(
        self, prompt: str
    ) -> Generator[tuple[Any, predicate], None, None]:
        behaviour_context = ""
        behaviour_mapping = self.behaviour_config.mapping.replace("{input}", prompt)

        if self.application_config.context:
            behaviour_context = self.behaviour_config.context.replace(
                "{context}", f"{self.application_config.context}"
            )

        predicates = self.application_config.predicates or []
    
        for pred in predicates:
            if pred.has_condition():
                if self.predicate_condition_cache.skip_logic_solver(pred.conditions):
                    log.debug(
                        f"[cache.hit]Condition cache HIT[/] → skipping solver for"
                        f" [predicate]{pred}[/]"
                    )
                    if not self.predicate_condition_cache.get(pred.conditions):
                        log.debug(
                            f"[cache]Cached result is FALSE[/]"
                            f" — skipping [predicate]{pred}[/]"
                        )
                        continue

                else:
                    call_oracle, condition_results = pred.execute_condition(
                        self.atom_database
                    )

                    if self.predicate_condition_cache.enabled:
                        for cond, val in condition_results:
                            self.predicate_condition_cache.update(cond, val)

                    if not call_oracle:
                        log.debug(
                            f"[warning]Condition FALSE[/]"
                            f" — skipping [predicate]{pred}[/]"
                        )
                        continue

            appl_mapping = behaviour_mapping.replace(
                "{instructions}", pred.prompt_description
            )
            appl_mapping = appl_mapping.replace("{atom}", pred.predicate_formatted)

            log.info(f"[predicate]▸ Extracting[/] [bold]{pred}[/]")
            yield self.invoke_llm_constrained(
                appl_mapping, pred.get_grammar(), behaviour_context
            ), pred

    def run(self, prompt: str) -> atomList:
        console.rule("[bold cyan]LGX — inference run[/]")
        log.info(f"[info]Input prompt[/]: {prompt[:120]}{'…' if len(prompt) > 120 else ''}")

        for response, pred in self._structured_output_call(prompt):
            if response is None:
                log.warning(f"[warning]No response[/] for predicate [predicate]{pred}[/]")
                continue

            atom_strings   = pred.parse_response(response)
            extracted_facts = filter_asp_atoms("\n".join(atom_strings))

            if extracted_facts.not_empty():
                log.info(
                    f"[success]✓[/] [predicate]{pred}[/] → "
                    f"[bold]{len(extracted_facts.atoms)}[/] atom(s)"
                )
                kb_result = pred.execute_knowledge(self.atom_database, extracted_facts)
                if kb_result.not_empty():
                    log.info(
                        f"[success]✓[/] KB execution for [predicate]{pred}[/] → "
                        f"[bold]{len(kb_result.atoms)}[/] new atom(s)"
                    )
                    # Invalidate non-monotone cache entries as new atoms might affect conditions
                    self.predicate_condition_cache.invalidate()
                    
                self.atom_database.atoms.extend(kb_result.atoms)
            else:
                log.debug(f"[atom]Empty extraction[/] for [predicate]{pred}[/]")

        stats = self.predicate_condition_cache.get_stats()
        console.print(
            f"[dim]Cache stats:[/] hits={stats.get('hit_monotone',0)+stats.get('hit_non_monotone',0)}"
            f"  misses={stats.get('miss_monotone',0)+stats.get('miss_non_monotone',0)}"
            f"  skips={stats.get('solver_skip',0)}"
        )
        console.rule("[bold green]Done[/]")
        return self.atom_database

    def get_extracted_atoms(self) -> atomList:
        return self.atom_database
    
    def get_statistics(self) -> dict:
        stats = self.predicate_condition_cache.get_stats()
        return {
            "cache_hits": stats.get("hit_monotone", 0) + stats.get("hit_non_monotone", 0),
            "cache_misses": stats.get("miss_monotone", 0) + stats.get("miss_non_monotone", 0),
            "solver_skips": stats.get("solver_skip", 0),
            "total_time": self.statistics.get("total_time", 0.0),
            "llm_calls": self.statistics.get("llm_calls", 0),
            "average_time_per_call": (self.statistics.get("total_time", 0.0) / self.statistics.get("llm_calls", 1)) if self.statistics.get("llm_calls", 0) > 0 else 0.0,
            "total_in_tokens": self.statistics.get("total_in_tokens", 0),
            "total_out_tokens": self.statistics.get("total_out_tokens", 0),
        }