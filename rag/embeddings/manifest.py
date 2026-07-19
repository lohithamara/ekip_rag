import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.chunking.schemas import Chunk
from rag.embeddings.config import EmbeddingConfig


@dataclass(frozen=True)
class EmbeddingManifest:
    document_id: str
    tenant_id: str
    department: str

    source_chunk_hash: str
    source_chunk_count: int

    model_name: str
    embedding_version: str
    normalize_embeddings: bool

    embedding_dimension: int
    total_embeddings: int


def calculate_chunk_hash(
    chunks: tuple[Chunk, ...],
) -> str:

    hasher = hashlib.sha256()

    for chunk in chunks:
        hasher.update(
            chunk.chunk_id.encode("utf-8")
        )
        hasher.update(
            chunk.text.encode("utf-8")
        )

    return hasher.hexdigest()


def create_embedding_manifest(
    chunks: tuple[Chunk, ...],
    config: EmbeddingConfig,
    dimension: int,
    total_embeddings: int,
) -> EmbeddingManifest:

    if not chunks:
        raise ValueError(
            "Cannot create manifest for empty chunks."
        )

    first_chunk = chunks[0]

    return EmbeddingManifest(
        document_id=first_chunk.document_id,
        tenant_id=first_chunk.tenant_id,
        department=first_chunk.department,
        source_chunk_hash=calculate_chunk_hash(
            chunks
        ),
        source_chunk_count=len(chunks),
        model_name=config.model_name,
        embedding_version=(
            config.embedding_version
        ),
        normalize_embeddings=(
            config.normalize_embeddings
        ),
        embedding_dimension=dimension,
        total_embeddings=total_embeddings,
    )


def write_manifest(
    manifest: EmbeddingManifest,
    file_path: Path,
) -> None:

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(manifest),
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_manifest(
    file_path: Path,
) -> EmbeddingManifest:

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return EmbeddingManifest(**data)