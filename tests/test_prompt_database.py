"""
tests/test_prompt_database.py
─────────────────────────────────────────────────────────────────────────────
Tests for the promptDatabase SQLite-backed cache.
Uses an in-memory SQLite DB so no temp files are created.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.core.prompt_database import promptDatabase


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db() -> promptDatabase:
    """In-memory SQLite promptDatabase."""
    conn   = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompt_cache (
            prompt          TEXT,
            llm_model       TEXT,
            configuration   TEXT,
            response        TEXT,
            token_in        INTEGER,
            token_out       INTEGER,
            extraction_time INTEGER,
            PRIMARY KEY (prompt, llm_model, configuration)
        )
    """)
    conn.commit()
    return promptDatabase(connection=conn, cursor=cursor)


@pytest.fixture
def sample_message() -> list:
    return [{"role": "user", "content": "hello"}]


@pytest.fixture
def sample_config() -> dict:
    return {"temperature": 0}


# ── initialize ────────────────────────────────────────────────────────────────

def test_initialize_creates_db_file(tmp_path):
    db_file = tmp_path / "test.db"
    pdb = promptDatabase.initialize(str(db_file))
    assert db_file.exists()
    pdb.close()


def test_initialize_creates_table(tmp_path):
    db_file = tmp_path / "test.db"
    pdb = promptDatabase.initialize(str(db_file))
    pdb.cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prompt_cache'"
    )
    assert pdb.cursor.fetchone() is not None
    pdb.close()


def test_initialize_idempotent(tmp_path):
    """Calling initialize twice on the same file must not raise."""
    db_file = str(tmp_path / "test.db")
    pdb1 = promptDatabase.initialize(db_file)
    pdb1.close()
    pdb2 = promptDatabase.initialize(db_file)
    pdb2.close()


# ── cache_response / get_cached_response ─────────────────────────────────────

class TestCacheAndGet:

    def test_cache_then_get_returns_row(self, db, sample_message, sample_config):
        db.cache_response(sample_message, "model-x", sample_config, '{"a":1}', 5, 10, 200)
        row = db.get_cached_response(sample_message, "model-x", sample_config)
        assert row is not None
        assert row[0] == '{"a":1}'
        assert row[1] == 5
        assert row[2] == 10
        assert row[3] == 200

    def test_get_returns_none_for_missing_entry(self, db, sample_message, sample_config):
        result = db.get_cached_response(sample_message, "unknown-model", sample_config)
        assert result is None

    def test_cache_is_keyed_by_model(self, db, sample_message, sample_config):
        db.cache_response(sample_message, "model-a", sample_config, "resp-a", 1, 1, 1)
        db.cache_response(sample_message, "model-b", sample_config, "resp-b", 2, 2, 2)

        row_a = db.get_cached_response(sample_message, "model-a", sample_config)
        row_b = db.get_cached_response(sample_message, "model-b", sample_config)

        assert row_a[0] == "resp-a"
        assert row_b[0] == "resp-b"

    def test_cache_is_keyed_by_configuration(self, db, sample_message):
        cfg0 = {"temperature": 0}
        cfg1 = {"temperature": 1}
        db.cache_response(sample_message, "m", cfg0, "cold", 1, 1, 1)
        db.cache_response(sample_message, "m", cfg1, "warm", 1, 1, 1)

        assert db.get_cached_response(sample_message, "m", cfg0)[0] == "cold"
        assert db.get_cached_response(sample_message, "m", cfg1)[0] == "warm"

    def test_upsert_overwrites_existing_entry(self, db, sample_message, sample_config):
        db.cache_response(sample_message, "m", sample_config, "v1", 1, 1, 1)
        db.cache_response(sample_message, "m", sample_config, "v2", 2, 2, 2)
        row = db.get_cached_response(sample_message, "m", sample_config)
        assert row[0] == "v2"


# ── delete_cached_response ────────────────────────────────────────────────────

class TestDeleteCachedResponse:

    def test_delete_removes_entry(self, db, sample_message, sample_config):
        db.cache_response(sample_message, "m", sample_config, "resp", 1, 1, 1)
        db.delete_cached_response(sample_message, "m", sample_config)
        assert db.get_cached_response(sample_message, "m", sample_config) is None

    def test_delete_nonexistent_is_noop(self, db, sample_message, sample_config):
        """Deleting a non-existent key must not raise."""
        db.delete_cached_response(sample_message, "ghost-model", sample_config)

    def test_delete_does_not_affect_other_entries(self, db, sample_message, sample_config):
        db.cache_response(sample_message, "m1", sample_config, "r1", 1, 1, 1)
        db.cache_response(sample_message, "m2", sample_config, "r2", 1, 1, 1)
        db.delete_cached_response(sample_message, "m1", sample_config)

        assert db.get_cached_response(sample_message, "m1", sample_config) is None
        assert db.get_cached_response(sample_message, "m2", sample_config) is not None


# ── close ─────────────────────────────────────────────────────────────────────

def test_close_makes_connection_unusable(db, sample_message, sample_config):
    db.close()
    with pytest.raises(Exception):
        db.get_cached_response(sample_message, "m", sample_config)