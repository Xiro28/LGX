from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Union

from typeguard import typechecked

from src.core.predicate_condition import predicate_condition
from src.helpers.console import get_logger

log = get_logger(__name__)

CacheMode = Literal["all", "monotone", "non_monotone"]


@typechecked
@dataclass(frozen=True)
class ConditionCache:
    enabled:    bool
    cache_mode: CacheMode

    monotone_cache:     Dict[str, bool] = field(init=False, default_factory=dict)
    non_monotone_cache: Dict[str, bool] = field(init=False, default_factory=dict)

    stats: Dict[str, int] = field(
        init=False,
        default_factory=lambda: {
            "hit_monotone":         0,
            "hit_non_monotone":     0,
            "miss_monotone":        0,
            "miss_non_monotone":    0,
            "solver_skip":          0,
            "invalidate_monotone":  0,
            "invalidate_non_monotone": 0,
        },
    )

    def _use_monotone_cache(self) -> bool:
        return self.cache_mode in ("all", "monotone")

    def _use_non_monotone_cache(self) -> bool:
        return self.cache_mode in ("all", "non_monotone")

    def _select_cache(self, condition: predicate_condition) -> Dict[str, bool] | None:
        if condition.monotone and self._use_monotone_cache():
            return self.monotone_cache
        
        return self.non_monotone_cache

    def update(
        self,
        condition: Union[List[predicate_condition], predicate_condition],
        value: bool,
    ) -> None:
        """Store *value* for every condition in *condition*.

        Accepts either a single ``predicate_condition`` or a list of them.

        Bug note: the original implementation used ``condition.condition``
        inside the loop instead of ``cond.condition``, causing an
        ``AttributeError`` when *condition* was a list.
        """
        if not self.enabled:
            return

        if isinstance(condition, predicate_condition):
            condition = [condition]

        for cond in condition:                       # ← loop variable is `cond`
            cache = self._select_cache(cond)
            if cache is not None:
                cache[cond.condition] = value        # ← was: condition.condition (BUG)
                log.debug(
                    f"[cache.{'hit' if value else 'miss'}]Cache UPDATE[/]"
                    f" {'mono' if cond.monotone else 'non-mono'}:"
                    f" {cond.condition!r} = {value}"
                )

    def skip_logic_solver(
        self,
        conditions: Union[predicate_condition, List[predicate_condition], None],
    ) -> bool:
        # Return True (and bump hit stats) when *all* conditions are cached.
        if not self.enabled or conditions is None:
            return False

        if isinstance(conditions, predicate_condition):
            conditions = [conditions]

        for condition in conditions:
            key = condition.condition
            if not key:
                continue

            cache = self._select_cache(condition)
            if cache is None:
                return False

            if key in cache:
                if condition.monotone:
                    self.stats["hit_monotone"] += 1
                else:
                    self.stats["hit_non_monotone"] += 1
            else:
                if condition.monotone:
                    self.stats["miss_monotone"] += 1
                else:
                    self.stats["miss_non_monotone"] += 1
                return False


        self.stats["solver_skip"] += 1
        return True

    def get(
        self,
        conditions: Union[predicate_condition, List[predicate_condition]],
    ) -> bool:
        """Return the AND of cached values for all *conditions*.

        Returns ``False`` when the cache is disabled or any key is absent.
        """
        if not self.enabled:
            return False

        if isinstance(conditions, predicate_condition):
            conditions = [conditions]

        result = True
        for condition in conditions:
            key = condition.condition
            if not key:
                continue
            cache = self._select_cache(condition)
            if cache is None:
                return False
            result = result and cache.get(key, False)

        log.debug(
            f"[cache.get]Cache GET[/] conditions={[c.condition for c in conditions]}"
            f" result={result}"
        )

        return result

    def invalidate(self, monotone: bool = False) -> None:
        if monotone and self._use_monotone_cache():
            self.monotone_cache.clear()
            self.stats["invalidate_monotone"] += 1
            log.debug("[warning]Monotone cache invalidated[/]")
        elif not monotone and self._use_non_monotone_cache():
            self.non_monotone_cache.clear()
            self.stats["invalidate_non_monotone"] += 1
            log.debug("[warning]Non-monotone cache invalidated[/]")

    def invalidate_all(self) -> None:
        self.invalidate(monotone=True)
        self.invalidate(monotone=False)

    def clear(self) -> None:
        self.monotone_cache.clear()
        self.non_monotone_cache.clear()
        for key in self.stats:
            self.stats[key] = 0
        log.debug("[info]Condition cache cleared[/]")

    def get_stats(self) -> Dict[str, int]:
        return self.stats