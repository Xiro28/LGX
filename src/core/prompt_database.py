from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Optional

from typeguard import typechecked

from src.helpers.console import get_logger

log = get_logger(__name__)

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS prompt_cache (
        prompt         TEXT,
        llm_model      TEXT,
        configuration  TEXT,
        response       TEXT,
        token_in       INTEGER,
        token_out      INTEGER,
        extraction_time INTEGER,
        PRIMARY KEY (prompt, llm_model, configuration)
    )
"""


@typechecked
@dataclass(frozen=True)
class promptDatabase:
    connection: sqlite3.Connection
    cursor: sqlite3.Cursor


    @staticmethod
    def initialize(db_path: str) -> "promptDatabase":
        connection = sqlite3.connect(db_path)
        cursor     = connection.cursor()
        cursor.execute(_CREATE_TABLE)
        connection.commit()
        log.info(f"[info]Prompt DB[/] opened → [bold]{db_path}[/]")
        return promptDatabase(connection=connection, cursor=cursor)


    def get_cached_response(
        self, message: list, llm_model: str, configuration: Dict
    ) -> Optional[tuple]:
        self.cursor.execute(
            "SELECT response, token_in, token_out, extraction_time "
            "FROM prompt_cache "
            "WHERE prompt = ? AND llm_model = ? AND configuration = ?",
            (str(message), llm_model, str(configuration)),
        )
        result = self.cursor.fetchone()
        if result:
            log.debug(
                f"[cache.hit]DB HIT[/] model={llm_model}"
                f" tokens_in={result[1]} tokens_out={result[2]}"
            )
        else:
            log.debug(f"[cache.miss]DB MISS[/] model={llm_model}")

        return result

    def cache_response(
        self,
        message: list,
        llm_model: str,
        configuration: Dict,
        response: str,
        token_in: int,
        token_out: int,
        extraction_time: int,
    ) -> None:
        self.cursor.execute(
            "INSERT OR REPLACE INTO prompt_cache "
            "(prompt, llm_model, configuration, response, token_in, token_out, extraction_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(message), llm_model, str(configuration), response, token_in, token_out, extraction_time),
        )
        self.connection.commit()
        log.debug(
            f"[info]Cached response[/] model={llm_model}"
            f" tokens_in={token_in} tokens_out={token_out}"
            f" duration={extraction_time}ms"
        )

    def delete_cached_response(
        self, message: list, llm_model: str, configuration: Dict
    ) -> None:
        self.cursor.execute(
            "DELETE FROM prompt_cache WHERE prompt = ? AND llm_model = ? AND configuration = ?",
            (str(message), llm_model, str(configuration)),
        )
        self.connection.commit()
        log.debug(f"[warning]Deleted cached response[/] model={llm_model}")

    def close(self) -> None:
        self.connection.close()
        log.debug("[info]Prompt DB connection closed[/]")