from rag.reranking.base import BaseReranker
from rag.reranking.config import RerankerConfig
from rag.reranking.cross_encoder import (
    CrossEncoderReranker,
)


def create_reranker(
    config: RerankerConfig,
) -> BaseReranker:
    """
    Create a reranker instance.
    """

    return CrossEncoderReranker(config)