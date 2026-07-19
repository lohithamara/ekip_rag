import pytest

from rag.chunking.factory import (
    get_chunking_strategy,
    get_chunking_strategy_names,
)
from rag.chunking.strategies.fixed_size import (
    FixedSizeChunkingStrategy,
)
from rag.chunking.strategies.json_aware import (
    JSONAwareChunkingStrategy,
)
from rag.chunking.strategies.page_aware import (
    PageAwareChunkingStrategy,
)
from rag.chunking.strategies.paragraph import (
    ParagraphChunkingStrategy,
)
from rag.chunking.strategies.recursive import (
    RecursiveChunkingStrategy,
)
from rag.chunking.strategies.sentence import (
    SentenceChunkingStrategy,
)
from rag.chunking.strategies.structure_aware import (
    StructureAwareChunkingStrategy,
)
from rag.chunking.strategies.table_aware import (
    TableAwareChunkingStrategy,
)


@pytest.mark.parametrize(
    ("strategy_name", "expected_strategy"),
    [
        (
            "fixed_size",
            FixedSizeChunkingStrategy,
        ),
        (
            "recursive",
            RecursiveChunkingStrategy,
        ),
        (
            "sentence",
            SentenceChunkingStrategy,
        ),
        (
            "paragraph",
            ParagraphChunkingStrategy,
        ),
        (
            "page_aware",
            PageAwareChunkingStrategy,
        ),
        (
            "structure_aware",
            StructureAwareChunkingStrategy,
        ),
        (
            "table_aware",
            TableAwareChunkingStrategy,
        ),
        (
            "json_aware",
            JSONAwareChunkingStrategy,
        ),
    ],
)
def test_get_chunking_strategy(
    strategy_name,
    expected_strategy,
):

    strategy = get_chunking_strategy(
        strategy_name
    )

    assert strategy is expected_strategy


def test_json_aware_is_registered():

    names = get_chunking_strategy_names()

    assert "json_aware" in names


def test_factory_rejects_unknown_strategy():

    with pytest.raises(
        ValueError,
        match="Unknown chunking strategy",
    ):
        get_chunking_strategy(
            "unknown"
        )