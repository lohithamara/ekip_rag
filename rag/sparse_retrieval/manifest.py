import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.chunking.schemas import Chunk
from rag.sparse_retrieval.config import (
    SparseRetrievalConfig,
)


@dataclass(frozen=True)
class SparseIndexManifest:

    chunk_count: int
    corpus_hash: str
    tokenizer_version: str
    bm25_version: str
    k1: float
    b: float


def compute_corpus_hash(
    chunks: list[Chunk],
) -> str:

    hasher = hashlib.sha256()

    for chunk in sorted(
        chunks,
        key=lambda item: item.chunk_id,
    ):
        hasher.update(
            chunk.chunk_id.encode()
        )

        hasher.update(
            chunk.content_hash.encode()
        )

    return hasher.hexdigest()


def build_manifest(
    chunks: list[Chunk],
    config: SparseRetrievalConfig,
) -> SparseIndexManifest:

    return SparseIndexManifest(
        chunk_count=len(chunks),
        corpus_hash=compute_corpus_hash(chunks),
        tokenizer_version=config.tokenizer_version,
        bm25_version=config.bm25_version,
        k1=config.k1,
        b=config.b,
    )


def save_manifest(
    manifest: SparseIndexManifest,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(manifest),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_manifest(
    path: Path,
) -> SparseIndexManifest:

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return SparseIndexManifest(**data)