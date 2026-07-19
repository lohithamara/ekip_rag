import os

from dataclasses import dataclass


@dataclass(frozen=True)
class VectorDBConfig:

    collection_name: str = "ekip_chunks"

    vector_size: int = 384

    distance: str = "cosine"

    upsert_batch_size: int = 100

    url: str | None = os.getenv(
        "QDRANT_URL"
    )

    api_key: str | None = os.getenv(
        "QDRANT_API_KEY"
    )