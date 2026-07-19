from rag.chunking.config import ChunkingConfig
from rag.chunking.tokenization.base import (
    BaseTokenCounter,
)
from rag.chunking.tokenization.token_counter import (
    CharacterCounter,
    WhitespaceTokenCounter,
)


def create_token_counter(
    config: ChunkingConfig,
) -> BaseTokenCounter:

    if config.length_unit == "characters":

        return CharacterCounter()

    if config.length_unit == "tokens":

        return WhitespaceTokenCounter()

    raise ValueError(
        f"Unsupported length unit: "
        f"{config.length_unit}"
    )