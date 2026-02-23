from typeguard import typechecked
from dataclasses import dataclass, field
from typing import Dict, List, Union, Literal

from src.core.predicate_condition import predicate_condition

CacheMode = Literal["all", "monotone", "non_monotone"]

@typechecked
@dataclass(frozen=True)
class ConditionCache:
    enabled: bool
    cache_mode: CacheMode

    monotone_cache: Dict[str, bool] = field(init=False, default_factory=dict)
    non_monotone_cache: Dict[str, bool] = field(init=False, default_factory=dict)

    stats: Dict[str, int] = field(init=False,
        default_factory=lambda: {
            "hit_monotone": 0,
            "hit_non_monotone": 0,
            "miss_monotone": 0,
            "miss_non_monotone": 0,
            "solver_skip": 0,
            "invalidate_monotone": 0,
            "invalidate_non_monotone": 0,
        }
    )

    def _use_monotone_cache(self) -> bool:
        return self.cache_mode in ("all", "monotone")

    def _use_non_monotone_cache(self) -> bool:
        return self.cache_mode in ("all", "non_monotone")

    def _select_cache(self, condition: predicate_condition) -> Dict[str, bool] | None:
        if condition.monotone and self._use_monotone_cache():
            return self.monotone_cache
        if not condition.monotone and self._use_non_monotone_cache():
            return self.non_monotone_cache
        return None 

    def update(self, condition: list[predicate_condition] | predicate_condition, value: bool) -> None:
        if not self.enabled:
            return
        
        if isinstance(condition, predicate_condition):
            condition = [condition]

        for cond in condition:
            cache = self._select_cache(cond)
            if cache is not None:
                cache[cond.condition] = value

    def skip_logic_solver(self,conditions: Union[predicate_condition, List[predicate_condition], None]) -> bool:

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

    def get(self, conditions: Union[predicate_condition, List[predicate_condition]]) -> bool:

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

        return result

    def invalidate(self, monotone: bool = False) -> None:
        if monotone and self._use_monotone_cache():
            self.monotone_cache.clear()
            self.stats["invalidate_monotone"] += 1
        elif not monotone and self._use_non_monotone_cache():
            self.non_monotone_cache.clear()
            self.stats["invalidate_non_monotone"] += 1

    def invalidate_all(self) -> None:
        self.invalidate(monotone=True)
        self.invalidate(monotone=False)

    def clear(self) -> None:
        self.monotone_cache.clear()
        self.non_monotone_cache.clear()

        for key in self.stats:
            self.stats[key] = 0

    def get_stats(self) -> Dict[str, int]:
        return self.stats
