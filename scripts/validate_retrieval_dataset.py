import json
from collections import Counter
from pathlib import Path

from rag.chunking.storage.serializer import ChunkSerializer


EVALUATION_FILE = Path(
    "data/evaluation/retrieval_queries.json"
)

CHUNK_ROOT = Path(
    "data/processed/chunks"
)


def main():

    queries = json.loads(
        EVALUATION_FILE.read_text(
            encoding="utf-8"
        )
    )

    chunks = []

    for path in CHUNK_ROOT.rglob(
        "chunks.jsonl"
    ):
        chunks.extend(
            ChunkSerializer.load_jsonl(path)
        )

    corpus_documents = {
        chunk.source_filename
        for chunk in chunks
    }

    errors = []

    seen_queries = set()

    departments = Counter()

    relevant_documents = Counter()

    for index, item in enumerate(
        queries,
        start=1,
    ):

        query = item.get(
            "query",
            ""
        ).strip()

        tenant_id = item.get(
            "tenant_id",
            ""
        ).strip()

        department = item.get(
            "department",
            ""
        ).strip()

        relevant = item.get(
            "relevant_documents",
            [],
        )

        if not query:
            errors.append(
                f"Query {index}: empty query."
            )

        if query in seen_queries:
            errors.append(
                f"Query {index}: duplicate query."
            )

        seen_queries.add(query)

        if not tenant_id:
            errors.append(
                f"Query {index}: empty tenant."
            )

        if not department:
            errors.append(
                f"Query {index}: empty department."
            )

        if not relevant:
            errors.append(
                f"Query {index}: no relevant documents."
            )

        for filename in relevant:

            if filename not in corpus_documents:
                errors.append(
                    f"Query {index}: missing corpus document "
                    f"{filename}"
                )

            relevant_documents[filename] += 1

        departments[department] += 1

    print()
    print("=" * 60)
    print("RETRIEVAL DATASET VALIDATION")
    print("=" * 60)

    print(
        f"Queries             : {len(queries)}"
    )

    print(
        f"Corpus documents    : "
        f"{len(corpus_documents)}"
    )

    print(
        f"Unique labels       : "
        f"{len(relevant_documents)}"
    )

    print(
        f"Errors              : {len(errors)}"
    )

    print()
    print("DEPARTMENT DISTRIBUTION")
    print("-" * 60)

    for department, count in sorted(
        departments.items()
    ):
        print(
            f"{department:<20} {count}"
        )

    print()
    print("MOST USED LABELS")
    print("-" * 60)

    for filename, count in (
        relevant_documents.most_common(10)
    ):
        print(
            f"{filename:<45} {count}"
        )

    if errors:

        print()
        print("ERROR DETAILS")
        print("-" * 60)

        for error in errors:
            print(error)

    print()
    print("=" * 60)
    print("FINAL STATUS")
    print("=" * 60)

    if errors:
        print(
            "FAIL: retrieval dataset is invalid."
        )
        raise SystemExit(1)

    print(
        "PASS: retrieval dataset is valid."
    )


if __name__ == "__main__":
    main()