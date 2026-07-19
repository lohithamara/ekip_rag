from api.dependencies import (
    create_container,
)

from qdrant_client.models import (
    PayloadSchemaType,
)


def main():

    container = create_container()

    vector_store = (
        container.vector_store
    )

    try:

        collection_name = (
            vector_store
            .config
            .collection_name
        )

        print(
            "Creating payload indexes..."
        )

        vector_store.client.create_payload_index(
            collection_name=(
                collection_name
            ),
            field_name="tenant_id",
            field_schema=(
                PayloadSchemaType.KEYWORD
            ),
            wait=True,
        )

        print(
            "Created tenant_id index."
        )


        vector_store.client.create_payload_index(
            collection_name=(
                collection_name
            ),
            field_name="department",
            field_schema=(
                PayloadSchemaType.KEYWORD
            ),
            wait=True,
        )

        print(
            "Created department index."
        )

        vector_store.client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

        print(
            "Created document_id index."
        )

        print(
            "Payload indexes created "
            "successfully."
        )

    finally:

        vector_store.close()


if __name__ == "__main__":
    main()