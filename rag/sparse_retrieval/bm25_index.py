from rank_bm25 import BM25Okapi

from rag.chunking.schemas import Chunk
from rag.sparse_retrieval.config import (
    SparseRetrievalConfig,
)
from rag.sparse_retrieval.tokenizer import tokenize

import pickle
from pathlib import Path

class BM25Index:

    def __init__(
        self,
        config: SparseRetrievalConfig,
    ):
        self.config = config
        self.chunks = []
        self.index = None

    def build(
        self,
        chunks: list[Chunk],
    ) -> int:

        self.chunks = list(chunks)

        corpus = [
            tokenize(chunk.text)
            for chunk in self.chunks
        ]

        self.index = BM25Okapi(
            corpus,
            k1=self.config.k1,
            b=self.config.b,
        )

        return len(self.chunks)

    def save(self) -> None:

        if self.index is None:
            raise ValueError(
                "BM25 index has not been built."
            )

        path = Path(self.config.index_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open("wb") as file:
            pickle.dump(self, file)


    @classmethod
    def load(
        cls,
        config: SparseRetrievalConfig,
    ):

        path = Path(config.index_path)

        if not path.is_file():
            raise FileNotFoundError(
                "BM25 index file does not exist."
            )

        with path.open("rb") as file:
            saved_index = pickle.load(file)

        if not isinstance(saved_index, cls):
            raise ValueError(
                "Invalid BM25 index file."
            )

        return saved_index

    def search(
        self,
        query: str,
        limit: int = 5,
        tenant_id: str | None = None,
        department: str | None = None,
    ) -> list[tuple[Chunk, float]]:

        if self.index is None:
            raise ValueError(
                "BM25 index has not been built."
            )

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "Search limit must be greater than zero."
            )

        scores = self.index.get_scores(
            tokenize(query)
        )

        results = []

        for chunk, score in zip(
            self.chunks,
            scores,
        ):

            if (
                tenant_id is not None
                and chunk.tenant_id != tenant_id
            ):
                continue

            if (
                department is not None
                and chunk.department != department
            ):
                continue

            if score == 0:
                continue

            results.append(
                (chunk, float(score))
            )

        results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return results[:limit]