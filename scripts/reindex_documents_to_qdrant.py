from sqlalchemy import select

from api.dependencies import (
    create_container,
)

from database.session import (
    SessionLocal,
)

from database.models.document import (
    Document,
)


def main():

    print()
    print("=" * 70)
    print("QDRANT CLOUD REINDEX")
    print("=" * 70)

    # ---------------------------------
    # CREATE APPLICATION SERVICES
    # ---------------------------------

    container = create_container()

    worker = container.document_worker

    vector_store = container.vector_store

    session = SessionLocal()

    try:

        # ---------------------------------
        # CREATE COLLECTION IF NEEDED
        # ---------------------------------

        created = (
            vector_store.create_collection()
        )

        if created:

            print(
                "Created Qdrant collection:",
                vector_store.config.collection_name,
            )

        else:

            print(
                "Qdrant collection already exists:",
                vector_store.config.collection_name,
            )


        # ---------------------------------
        # LOAD DOCUMENTS
        # ---------------------------------

        documents = list(

            session.scalars(

                select(Document)

                .where(
                    Document.status.in_(
                        [
                            "COMPLETED",
                            "READY",
                        ]
                    )
                )

                .order_by(
                    Document.id
                )

            ).all()

        )


        print(
            f"Documents found: "
            f"{len(documents)}"
        )

        print("=" * 70)


        successful = 0
        failed = 0


        # ---------------------------------
        # REINDEX
        # ---------------------------------

        for document in documents:

            print()
            print(
                f"Reindexing document "
                f"{document.id}: "
                f"{document.filename}"
            )

            try:

                worker.process_document(
                    document.id
                )

                successful += 1

                print(
                    f"SUCCESS: "
                    f"{document.filename}"
                )


            except Exception as exc:

                failed += 1

                print(
                    f"FAILED: "
                    f"{document.filename}"
                )

                print(
                    f"Reason: {exc}"
                )


        # ---------------------------------
        # SUMMARY
        # ---------------------------------

        print()
        print("=" * 70)
        print("REINDEX COMPLETE")
        print("=" * 70)

        print(
            f"Successful : {successful}"
        )

        print(
            f"Failed     : {failed}"
        )

        print("=" * 70)


        # ---------------------------------
        # QDRANT COLLECTION INFO
        # ---------------------------------

        info = (
            vector_store
            .get_collection_info()
        )

        print()
        print(
            "Qdrant collection:",
            vector_store.config.collection_name,
        )

        print(
            "Points count:",
            info.points_count,
        )


    finally:

        session.close()

        vector_store.close()


if __name__ == "__main__":

    main()