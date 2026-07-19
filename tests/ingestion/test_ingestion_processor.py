import sys

from ingestion.parsers.json_parser import JSONParser
from ingestion.parsers.markdown_parser import MarkdownParser
from ingestion.parsers.registry import ParserRegistry
from ingestion.parsers.text_parser import TextParser
from ingestion.parsers.csv_parser import CSVParser
from ingestion.parsers.xlsx_parser import XLSXParser
from ingestion.parsers.docx_parser import DOCXParser
from ingestion.parsers.html_parser import HTMLParser
from ingestion.parsers.pdf_parser import PDFParser
from ingestion.services.ingestion_processor import IngestionProcessor
from ingestion.storage.s3_client import S3Client


def main():

    # ----------------------------------------
    # 1. Read requested extension
    # ----------------------------------------

    requested_extension = (
        sys.argv[1].lower()
        if len(sys.argv) > 1
        else ".txt"
    )

    if not requested_extension.startswith("."):
        requested_extension = f".{requested_extension}"

    print(f"Requested extension: {requested_extension}")

    # ----------------------------------------
    # 2. Initialize S3
    # ----------------------------------------

    s3 = S3Client()

    # ----------------------------------------
    # 3. Create parser registry
    # ----------------------------------------

    registry = ParserRegistry()

    registry.register(TextParser())
    registry.register(MarkdownParser())
    registry.register(JSONParser())
    registry.register(CSVParser())
    registry.register(XLSXParser())
    registry.register(DOCXParser())
    registry.register(HTMLParser())
    registry.register(PDFParser())

    print(
        "Registered parsers:",
        registry.registered_extensions,
    )

    # ----------------------------------------
    # 4. Create ingestion processor
    # ----------------------------------------

    processor = IngestionProcessor(
        s3_client=s3,
        parser_registry=registry,
    )

    # ----------------------------------------
    # 5. Get S3 objects
    # ----------------------------------------

    objects = s3.list_objects(
        prefix="original/tenant_1/"
    )

    # ----------------------------------------
    # 6. Filter by requested extension
    # ----------------------------------------

    matching_objects = [
        obj
        for obj in objects
        if obj["Key"].lower().endswith(
            requested_extension
        )
    ]

    print(
        f"Matching objects found: "
        f"{len(matching_objects)}"
    )

    if not matching_objects:
        raise RuntimeError(
            f"No {requested_extension} "
            f"objects found in S3."
        )

    # ----------------------------------------
    # 7. Select one matching document
    # ----------------------------------------

    selected_object = matching_objects[0]

    s3_key = selected_object["Key"]

    # ----------------------------------------
    # 8. Extract metadata from S3 key
    # ----------------------------------------

    parts = s3_key.split("/")

    if len(parts) < 5:
        raise ValueError(
            f"Unexpected S3 key structure: {s3_key}"
        )

    tenant_id = parts[1]

    department = parts[2]

    content_hash = parts[3]

    print("\nSELECTED S3 OBJECT")
    print("------------------")

    print(f"S3 Key     : {s3_key}")
    print(f"Tenant     : {tenant_id}")
    print(f"Department : {department}")
    print(f"Hash       : {content_hash}")

# ----------------------------------------
# 9. Process document
# ----------------------------------------

    raw_document = processor.process(
        s3_key=s3_key,
        tenant_id=tenant_id,
        department=department,
        content_hash=content_hash,
    )


# ----------------------------------------
# 10. Display basic result
# ----------------------------------------

    print("\nRAW DOCUMENT")
    print("------------------")

    print(f"Document ID : {raw_document.metadata.document_id}")
    print(f"Filename    : {raw_document.metadata.filename}")
    print(f"File Type   : {raw_document.metadata.file_type}")
    print(f"Parser      : {raw_document.parser_name}")
    print(f"Parser Ver. : {raw_document.parser_version}")
    print(f"Extra Meta  : {raw_document.metadata.extra}")
    print(f"Warnings    : {raw_document.warnings}")


    print("\nRAW TEXT PREVIEW")
    print("------------------")

    print(raw_document.raw_text[:2000])


# ----------------------------------------
# 11. Display table information
# ----------------------------------------

    print("\nTABLE INFORMATION")
    print("------------------")

    print(f"Number of tables: {len(raw_document.tables)}")

    for table in raw_document.tables:

        print(f"\nTable ID    : {table.table_id}")
        print(f"Page Number : {table.page_number}")
        print(f"Sheet Name  : {table.metadata.get('sheet_name')}")
        print(f"Headers     : {table.headers}")
        print(f"Number Rows : {len(table.rows)}")

        print("First 3 Rows:")

        for row in table.rows[:3]:
            print(row)


# ----------------------------------------
# 12. Display PDF page information
# ----------------------------------------

    print("\nPAGE INFORMATION")
    print("------------------")

    print(f"Number of pages: {len(raw_document.pages)}")

    for page in raw_document.pages[:5]:

        print(f"\nPage Number : {page.page_number}")

        print(f"OCR Used    : {page.ocr_used}")

        print(f"Text Length : {len(page.text)}")

        print(f"Page Meta   : {page.metadata}")

        print(f"Text Preview: {page.text[:300]}")

if __name__ == "__main__":
    main()