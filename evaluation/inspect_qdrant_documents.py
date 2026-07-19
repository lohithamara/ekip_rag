from rag.vector_store.config import VectorDBConfig
from rag.vector_store.qdrant_store import QdrantVectorStore


config = VectorDBConfig()

vector_store = QdrantVectorStore(config)

documents = {}
offset = None

while True:
    points, offset = vector_store.client.scroll(
        collection_name=config.collection_name,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    for point in points:
        payload = point.payload or {}

        document_id = payload.get("document_id")
        tenant_id = payload.get("tenant_id")
        filename = payload.get("source_filename")

        key = (document_id, tenant_id, filename)

        documents[key] = documents.get(key, 0) + 1

    if offset is None:
        break


print("\nDocuments currently in Qdrant")
print("-" * 80)

for (document_id, tenant_id, filename), chunk_count in sorted(
    documents.items(),
    key=lambda item: str(item[0]),
):
    print(
        f"Document ID: {document_id} | "
        f"Tenant ID: {tenant_id} | "
        f"File: {filename} | "
        f"Chunks: {chunk_count}"
    )

print("-" * 80)
print(f"Total unique documents: {len(documents)}")

vector_store.close()