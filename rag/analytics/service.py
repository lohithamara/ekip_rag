from pathlib import Path

import pandas as pd

from rag.analytics.exceptions import (
    UnsupportedOperationError,
    ColumnNotFoundError,
)


class AnalyticsService:

    def load(
        self,
        path: str,
    ) -> pd.DataFrame:

        suffix = (
            Path(path)
            .suffix
            .lower()
        )

        if suffix == ".csv":
            return pd.read_csv(
                path
            )

        if suffix in (
            ".xlsx",
            ".xls",
        ):
            return pd.read_excel(
                path
            )

        raise UnsupportedOperationError(
            f"{suffix} files are not supported."
        )

    def _require_column(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> None:

        if column not in dataframe.columns:

            raise ColumnNotFoundError(
                f"Column '{column}' does not exist."
            )

    def execute(
        self,
        dataframe: pd.DataFrame,
        operation: str,
        parameters: dict,
    ):

        operation = operation.lower()

        if operation == "count":

            return len(
                dataframe
            )

        if operation == "columns":

            return list(
                dataframe.columns
            )

        if operation == "shape":

            return dataframe.shape

        if operation == "describe":

            return dataframe.describe()

        if operation == "sum":

            column = parameters[
                "column"
            ]

            self._require_column(
                dataframe,
                column,
            )

            return dataframe[
                column
            ].sum()

        if operation == "mean":

            column = parameters[
                "column"
            ]

            self._require_column(
                dataframe,
                column,
            )

            return dataframe[
                column
            ].mean()

        if operation == "min":

            column = parameters[
                "column"
            ]

            self._require_column(
                dataframe,
                column,
            )

            return dataframe[
                column
            ].min()

        if operation == "max":

            column = parameters[
                "column"
            ]

            self._require_column(
                dataframe,
                column,
            )

            return dataframe[
                column
            ].max()

        if operation == "top_n":

            column = parameters[
                "column"
            ]

            self._require_column(
                dataframe,
                column,
            )

            n = parameters.get(
                "n",
                5,
            )

            return dataframe.nlargest(
                n,
                column,
            )

        if operation == "bottom_n":

            column = parameters[
                "column"
            ]

            self._require_column(
                dataframe,
                column,
            )

            n = parameters.get(
                "n",
                5,
            )

            return dataframe.nsmallest(
                n,
                column,
            )

        raise UnsupportedOperationError(
            f"Unsupported operation '{operation}'."
        )