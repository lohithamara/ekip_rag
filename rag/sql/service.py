from pathlib import Path
import sqlite3

import pandas as pd

from rag.sql.exceptions import (
    UnsupportedQueryError,
)


class SQLService:

    def connect(
        self,
        database: str,
    ):

        return sqlite3.connect(
            database
        )

    def execute(
        self,
        database: str,
        query: str,
    ):

        normalized = (
            query
            .strip()
            .lower()
        )

        if not normalized.startswith(
            "select"
        ):
            raise UnsupportedQueryError(
                "Only SELECT queries are allowed."
            )

        connection = self.connect(
            database
        )

        try:

            dataframe = pd.read_sql_query(
                query,
                connection,
            )

        finally:

            connection.close()

        return dataframe