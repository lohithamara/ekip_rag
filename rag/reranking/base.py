from abc import ABC
from abc import abstractmethod

from rag.retrieval.schemas import (
    RetrievalResult,
)


class BaseReranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Re-rank retrieved chunks for a query.

        Parameters
        ----------
        query:
            User query.

        results:
            Retrieved chunks.

        top_k:
            Number of chunks to return.

        Returns
        -------
        list[RetrievalResult]
            Ranked retrieval results.
        """

        raise NotImplementedError