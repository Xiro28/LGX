from __future__ import annotations

import pytest

from src.core.cache import ConditionCache
from src.core.predicate_condition import predicate_condition

@pytest.fixture
def cond_mono() -> predicate_condition:
    return predicate_condition("edge(_, _)", True)


@pytest.fixture
def cond_non_mono() -> predicate_condition:
    return predicate_condition("not error(_)", False)

class TestConditionCacheConstruction:

    def test_defaults_empty(self, condition_cache_all):
        assert condition_cache_all.monotone_cache    == {}
        assert condition_cache_all.non_monotone_cache == {}

    @pytest.mark.parametrize("mode", ["all", "monotone", "non_monotone"])
    def test_valid_modes_accepted(self, mode):
        cache = ConditionCache(enabled=True, cache_mode=mode)
        assert cache.cache_mode == mode


# ── update / get ──────────────────────────────────────────────────────────────

class TestUpdateAndGet:

    def test_update_monotone(self, condition_cache_all, cond_mono):
        condition_cache_all.update(cond_mono, True)
        assert condition_cache_all.get(cond_mono) is True

    def test_update_non_monotone(self, condition_cache_all, cond_non_mono):
        condition_cache_all.update(cond_non_mono, False)
        assert condition_cache_all.get(cond_non_mono) is False

    def test_update_list(self, condition_cache_all, cond_mono, cond_non_mono):
        condition_cache_all.update([cond_mono, cond_non_mono], True)
        assert condition_cache_all.monotone_cache.get(cond_mono.condition) is True

    def test_get_returns_false_for_missing_key(self, condition_cache_all, cond_mono):
        assert condition_cache_all.get(cond_mono) is False

    def test_disabled_cache_update_is_noop(self, condition_cache_disabled, cond_mono):
        condition_cache_disabled.update(cond_mono, True)
        assert condition_cache_disabled.monotone_cache == {}

    def test_disabled_cache_get_returns_false(self, condition_cache_disabled, cond_mono):
        condition_cache_disabled.update(cond_mono, True)
        assert condition_cache_disabled.get(cond_mono) is False


# ── skip_logic_solver ─────────────────────────────────────────────────────────

class TestSkipLogicSolver:

    def test_skip_when_all_cached(self, condition_cache_all, cond_mono):
        condition_cache_all.update(cond_mono, True)
        assert condition_cache_all.skip_logic_solver(cond_mono) is True

    def test_no_skip_when_not_cached(self, condition_cache_all, cond_mono):
        assert condition_cache_all.skip_logic_solver(cond_mono) is False

    def test_returns_false_for_none(self, condition_cache_all):
        assert condition_cache_all.skip_logic_solver(None) is False

    def test_disabled_always_returns_false(self, condition_cache_disabled, cond_mono):
        condition_cache_disabled.update(cond_mono, True)
        assert condition_cache_disabled.skip_logic_solver(cond_mono) is False

    def test_list_of_conditions_all_cached(self, condition_cache_all, cond_mono, cond_non_mono):
        condition_cache_all.update(cond_mono, True)
        condition_cache_all.update(cond_non_mono, True)
        assert condition_cache_all.skip_logic_solver([cond_mono, cond_non_mono]) is True

    def test_list_with_one_uncached_returns_false(self, condition_cache_all, cond_mono, cond_non_mono):
        condition_cache_all.update(cond_mono, True)
        # cond_non_mono is NOT cached
        assert condition_cache_all.skip_logic_solver([cond_mono, cond_non_mono]) is False


# ── invalidate / clear ────────────────────────────────────────────────────────

class TestInvalidateAndClear:

    def test_invalidate_monotone_only(self, condition_cache_all, cond_mono, cond_non_mono):
        condition_cache_all.update(cond_mono,     True)
        condition_cache_all.update(cond_non_mono, True)
        condition_cache_all.invalidate(monotone=True)
        assert condition_cache_all.monotone_cache    == {}
        assert condition_cache_all.non_monotone_cache != {}

    def test_invalidate_non_monotone_only(self, condition_cache_all, cond_mono, cond_non_mono):
        condition_cache_all.update(cond_mono,     True)
        condition_cache_all.update(cond_non_mono, True)
        condition_cache_all.invalidate(monotone=False)
        assert condition_cache_all.non_monotone_cache == {}
        assert condition_cache_all.monotone_cache     != {}

    def test_invalidate_all(self, condition_cache_all, cond_mono, cond_non_mono):
        condition_cache_all.update(cond_mono,     True)
        condition_cache_all.update(cond_non_mono, True)
        condition_cache_all.invalidate_all()
        assert condition_cache_all.monotone_cache    == {}
        assert condition_cache_all.non_monotone_cache == {}

    def test_clear_resets_stats(self, condition_cache_all, cond_mono):
        condition_cache_all.update(cond_mono, True)
        condition_cache_all.skip_logic_solver(cond_mono)
        condition_cache_all.clear()
        assert all(v == 0 for v in condition_cache_all.get_stats().values())


# ── stats ─────────────────────────────────────────────────────────────────────

class TestStats:

    def test_miss_increments_on_cache_miss(self, condition_cache_all, cond_mono):
        condition_cache_all.skip_logic_solver(cond_mono)
        stats = condition_cache_all.get_stats()
        assert stats["miss_monotone"] == 1

    def test_hit_increments_on_cache_hit(self, condition_cache_all, cond_mono):
        condition_cache_all.update(cond_mono, True)
        condition_cache_all.skip_logic_solver(cond_mono)
        stats = condition_cache_all.get_stats()
        assert stats["hit_monotone"] == 1