import tempfile
from pathlib import Path


class TemporaryWorkspace:

    def __init__(self):
        self._temp_directory = None

    def __enter__(self) -> Path:

        self._temp_directory = tempfile.TemporaryDirectory()

        return Path(self._temp_directory.name)

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        if self._temp_directory:
            self._temp_directory.cleanup()