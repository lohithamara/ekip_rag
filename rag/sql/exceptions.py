class SQLError(
    Exception,
):
    pass


class UnsupportedQueryError(
    SQLError,
):
    pass