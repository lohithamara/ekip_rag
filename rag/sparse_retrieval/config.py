from dataclasses import dataclass


@dataclass(frozen=True)
class SparseRetrievalConfig:

    index_path: str = (
        "data/sparse_index/bm25.pkl"
    )
    manifest_path: str = (
        "data/sparse_index/manifest.json"
    )
    tokenizer_version: str = "1.0"

    bm25_version: str = "1.0"

    k1: float = 1.5

    b: float = 0.75