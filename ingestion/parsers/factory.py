from ingestion.parsers.csv_parser import CSVParser
from ingestion.parsers.docx_parser import DOCXParser
from ingestion.parsers.html_parser import HTMLParser
from ingestion.parsers.json_parser import JSONParser
from ingestion.parsers.markdown_parser import MarkdownParser
from ingestion.parsers.pdf_parser import PDFParser
from ingestion.parsers.registry import ParserRegistry
from ingestion.parsers.text_parser import TextParser
from ingestion.parsers.xlsx_parser import XLSXParser


def create_default_parser_registry() -> ParserRegistry:

    registry = ParserRegistry()

    registry.register(TextParser())
    registry.register(MarkdownParser())
    registry.register(JSONParser())
    registry.register(CSVParser())
    registry.register(XLSXParser())
    registry.register(DOCXParser())
    registry.register(HTMLParser())
    registry.register(PDFParser())

    return registry