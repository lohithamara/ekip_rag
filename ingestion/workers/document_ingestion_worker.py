from rag.chunking.config import ChunkingConfig


class DocumentIngestionWorker:

    def __init__(
        self,
        document_repository,
        department_repository,
        ingestion_processor,
        cleaning_service,
        chunking_service,
        embedding_service,
        vector_store
    ):

        self.document_repository = (
            document_repository
        )

        self.department_repository = (
            department_repository   
        )

        self.ingestion_processor = (
            ingestion_processor
        )

        self.cleaning_service = (
            cleaning_service
        )

        self.chunking_service = (
            chunking_service
        )

        self.embedding_service = (
            embedding_service
        )

        self.vector_store = (
            vector_store
        )

    def process_document(
        self,
        document_id: int,
    ):
        try:

            document = self.document_repository.get_by_id(
                document_id
            )

            if document is None:
                return

            print(f"Processing document {document_id}")

            department = self.department_repository.get_by_id(
                document.department_id
            )

            if department is None:
                raise ValueError(
                    f"Department not found for document {document_id}"
                )

            raw_document = self.ingestion_processor.process(
                s3_key=document.s3_key,
                tenant_id=str(document.tenant_id),
                department=department.name,
                content_hash=document.content_hash,
                document_id=document_id,
            )

            print(f"Raw document extracted for document {document_id}")

            clean_document = self.cleaning_service.clean_document(
                raw_document
            )

            print(f"cleaning completed for document {document_id}")

            chunk_result = self.chunking_service.chunk_document(
                clean_document,
                ChunkingConfig(
                    strategy="recursive",
                ),
            )
            print("Total chunks:", len(chunk_result.chunks))

            # for i, chunk in enumerate(chunk_result.chunks[:5]):
            #     print("="*50)
            #     print("Chunk", i)
            #     print("Length:", len(chunk.text))
            #     print(chunk.text[:500])
            print(f"Chunking completed for document {document_id}, {len(chunk_result.chunks)} chunks created")

            embedding_result = self.embedding_service.embed_chunks(
                chunk_result.chunks
            )

            print(f"Embedding completed for document {document_id}")
            # print("Chunk department:", chunk_result.chunks[0].department)
            # print("Chunk tenant:", chunk_result.chunks[0].tenant_id)
            points = self.vector_store.build_points(
                chunk_result.chunks,
                embedding_result.records,
            )

            print(f"Uploading embeddings for document {document_id}")

            self.vector_store.upsert_points(points)

            self.document_repository.update_status(
                document_id,
                "READY",
            )

        except Exception as e:

            print(f"Document {document_id} failed")

            print(e)

            self.document_repository.update_status(
                document_id,
                "FAILED",
            )

            raise