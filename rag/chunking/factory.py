from rag.chunking.base import BaseChunkingStrategy
from rag.chunking.strategies.fixed_size import FixedSizeChunkingStrategy
from rag.chunking.strategies.page_aware import PageAwareChunkingStrategy
from rag.chunking.strategies.paragraph import ParagraphChunkingStrategy
from rag.chunking.strategies.recursive import RecursiveChunkingStrategy
from rag.chunking.strategies.sentence import SentenceChunkingStrategy
from rag.chunking.strategies.structure_aware import StructureAwareChunkingStrategy
from rag.chunking.strategies.table_aware import TableAwareChunkingStrategy
from rag.chunking.strategies.json_aware import JSONAwareChunkingStrategy

CHUNKING_STRATEGIES: dict[
    str,
    type[BaseChunkingStrategy],
] = {
    FixedSizeChunkingStrategy.STRATEGY_NAME: FixedSizeChunkingStrategy,
    RecursiveChunkingStrategy.STRATEGY_NAME: RecursiveChunkingStrategy,
    SentenceChunkingStrategy.STRATEGY_NAME: SentenceChunkingStrategy,
    ParagraphChunkingStrategy.STRATEGY_NAME: ParagraphChunkingStrategy,
    PageAwareChunkingStrategy.STRATEGY_NAME: PageAwareChunkingStrategy,
    StructureAwareChunkingStrategy.STRATEGY_NAME: StructureAwareChunkingStrategy,
    TableAwareChunkingStrategy.STRATEGY_NAME: TableAwareChunkingStrategy,
    JSONAwareChunkingStrategy.STRATEGY_NAME: JSONAwareChunkingStrategy
}


def get_chunking_strategy(
    name: str,
) -> type[BaseChunkingStrategy]:

    try:
        return CHUNKING_STRATEGIES[name]
    except KeyError:
        available = ", ".join(
            sorted(CHUNKING_STRATEGIES)
        )
        raise ValueError(
            f"Unknown chunking strategy: {name}. "
            f"Available strategies: {available}"
        )


def get_chunking_strategy_names() -> list[str]:
    return sorted(CHUNKING_STRATEGIES)