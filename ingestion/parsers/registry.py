from pathlib import Path

from ingestion.parsers.base import BaseParser


class ParserRegistry:

    def __init__(self):
        self._parsers: dict[str, BaseParser] = {}

    def register(self, parser: BaseParser) -> None:

        for extension in parser.supported_extensions:

            normalized_extension = extension.lower()

            if not normalized_extension.startswith("."):
                normalized_extension = f".{normalized_extension}"

            if normalized_extension in self._parsers:
                raise ValueError(
                    f"Parser already registered for "
                    f"{normalized_extension}"
                )

            self._parsers[normalized_extension] = parser

    def get_parser(self, file_path: Path) -> BaseParser:

        extension = file_path.suffix.lower()

        parser = self._parsers.get(extension)

        if parser is None:
            raise ValueError(
                f"No parser registered for extension: {extension}"
            )

        return parser

    def supports(self, file_path: Path) -> bool:

        extension = file_path.suffix.lower()

        return extension in self._parsers

    @property
    def registered_extensions(self) -> set[str]:
        return set(self._parsers.keys())