from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


REQUIRED_PAYLOAD_FIELDS = {
    "chunk_id",
    "document_id",
    "tenant_id",
    "department",
    "content_hash",
    "source_filename",
    "source_file_type",
    "chunk_index",
    "chunk_type",
    "strategy",
    "text",
}


def main():

    config = VectorDBConfig()

    store = QdrantVectorStore(config)

    total_points = 0
    unique_chunk_ids = set()
    documents = set()
    tenants = set()
    departments = set()

    errors = []

    print()
    print("===================================")
    print("VECTOR CORPUS VALIDATION")
    print("===================================")

    try:

        if not store.collection_exists():
            raise RuntimeError(
                "Vector collection does not exist."
            )

        info = store.get_collection_info()

        offset = None

        while True:

            points, offset = store.client.scroll(
                collection_name=(
                    config.collection_name
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            for point in points:

                total_points += 1

                payload = point.payload or {}

                missing_fields = (
                    REQUIRED_PAYLOAD_FIELDS
                    - set(payload)
                )

                if missing_fields:

                    errors.append(
                        f"Point {point.id} missing "
                        f"payload fields: "
                        f"{sorted(missing_fields)}"
                    )

                    continue

                chunk_id = payload["chunk_id"]

                if chunk_id in unique_chunk_ids:

                    errors.append(
                        f"Duplicate chunk_id: "
                        f"{chunk_id}"
                    )

                unique_chunk_ids.add(
                    chunk_id
                )

                documents.add(
                    payload["document_id"]
                )

                tenants.add(
                    payload["tenant_id"]
                )

                departments.add(
                    payload["department"]
                )

                vector = point.vector

                if vector is None:

                    errors.append(
                        f"Point {point.id} "
                        f"has no vector."
                    )

                elif len(vector) != (
                    config.vector_size
                ):

                    errors.append(
                        f"Point {point.id} has "
                        f"dimension {len(vector)}, "
                        f"expected "
                        f"{config.vector_size}."
                    )

                if not payload["text"].strip():

                    errors.append(
                        f"Point {point.id} "
                        f"has empty text."
                    )

            if offset is None:
                break

        if total_points != info.points_count:

            errors.append(
                "Scrolled point count does not "
                "match collection points_count."
            )

        print()
        print("===================================")
        print("VECTOR CORPUS VALIDATION SUMMARY")
        print("===================================")

        print(
            f"Collection name      : "
            f"{config.collection_name}"
        )

        print(
            f"Collection points    : "
            f"{info.points_count}"
        )

        print(
            f"Points checked       : "
            f"{total_points}"
        )

        print(
            f"Unique chunk IDs     : "
            f"{len(unique_chunk_ids)}"
        )

        print(
            f"Unique documents     : "
            f"{len(documents)}"
        )

        print(
            f"Unique tenants       : "
            f"{len(tenants)}"
        )

        print(
            f"Unique departments   : "
            f"{len(departments)}"
        )

        print(
            f"Vector dimension     : "
            f"{config.vector_size}"
        )

        print(
            f"Errors               : "
            f"{len(errors)}"
        )

        if errors:

            print()
            print("ERROR DETAILS")
            print("-----------------------------------")

            for error in errors:
                print(error)

        print()
        print("===================================")
        print("FINAL STATUS")
        print("===================================")

        if errors:
            print(
                "FAIL: vector corpus is invalid."
            )
            raise SystemExit(1)

        print(
            "PASS: vector corpus is valid."
        )

    finally:

        store.close()


if __name__ == "__main__":
    main()