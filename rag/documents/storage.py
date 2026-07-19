from pathlib import Path


class LocalDocumentStorage:

    def __init__(
        self,
        root: str,
    ):
        self.root = Path(root)

    def exists(
        self,
        path: str,
    ) -> bool:

        return (
            self.root
            .joinpath(path)
            .exists()
        )

    def resolve(
        self,
        path: str,
    ) -> Path:

        return (
            self.root
            .joinpath(path)
        )

    def open(
        self,
        path: str,
        mode: str = "rb",
    ):

        return (
            self.resolve(path)
            .open(mode)
        )