class DocumentNotFoundError(
    FileNotFoundError,
):
    pass


class DocumentAccessError(
    PermissionError,
):
    pass

class DocumentNotFoundError(Exception):

    def __init__(
        self,
        document: int | str,
    ):
        super().__init__(
            f"Document '{document}' not found."
        )


class DuplicateDocumentError(Exception):

    def __init__(
        self,
        filename: str,
    ):
        super().__init__(
            f"Document '{filename}' already exists."
        )


class DepartmentNotFoundError(Exception):

    def __init__(
        self,
        department: str,
    ):
        super().__init__(
            f"Department '{department}' not found."
        )

class DepartmentAccessDeniedError(Exception):
    
    def __init__(
        self,
        department: str,
    ):
        super().__init__(
            f"Access to department '{department}' denied."
        )