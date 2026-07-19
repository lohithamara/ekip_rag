from pathlib import Path

from ingestion.parsers.registry import ParserRegistry
from ingestion.parsers.text_parser import TextParser


def main():

    registry = ParserRegistry()

    registry.register(TextParser())

    print(
        "Registered extensions:",
        registry.registered_extensions,
    )

    txt_file = Path("example.txt")

    parser = registry.get_parser(txt_file)

    print(
        "Selected parser:",
        parser.__class__.__name__,
    )


if __name__ == "__main__":
    main()