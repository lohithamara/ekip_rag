class AnalyticsError(
    Exception,
):
    pass


class UnsupportedOperationError(
    AnalyticsError,
):
    pass


class ColumnNotFoundError(
    AnalyticsError,
):
    pass