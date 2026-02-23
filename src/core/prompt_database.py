from typeguard import typechecked
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sqlite3

@typechecked
@dataclass(frozen=True)
class promptDatabase:
    
    connection: sqlite3.Connection
    cursor: sqlite3.Cursor

    @staticmethod
    def initialize(db_path: str) -> 'promptDatabase':
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_cache (
                prompt TEXT,
                llm_model TEXT,
                configuration TEXT,
                response TEXT,
                token_in INTEGER,
                token_out INTEGER,
                extraction_time INTEGER,
                PRIMARY KEY (prompt, llm_model, configuration)
            )
        ''')

        return promptDatabase(connection=connection, cursor=cursor)

    def get_cached_response(self, message: list, llm_model: str, configuration: Dict) -> Optional[Dict]:
        self.cursor.execute('SELECT response, token_in, token_out, extraction_time FROM prompt_cache WHERE prompt = ? and llm_model = ? and configuration = ?', (str(message), llm_model, str(configuration)))
        return self.cursor.fetchone()
    
    def cache_response(self, message: list, llm_model: str, configuration: Dict, response: str, token_in: int, token_out: int, extraction_time: int):
        self.cursor.execute('INSERT INTO prompt_cache (prompt, llm_model, configuration, response, token_in, token_out, extraction_time) VALUES (?, ?, ?, ?, ?, ?, ?)', (str(message), llm_model, str(configuration), response, token_in, token_out, extraction_time ))
        self.connection.commit()

    def delete_cached_response(self, message: list, llm_model: str, configuration: Dict):
        self.cursor.execute('DELETE FROM prompt_cache WHERE prompt = ? and llm_model = ? and configuration = ?', (str(message), llm_model, str(configuration)))
        self.connection.commit()

    def close(self):
        self.connection.close()