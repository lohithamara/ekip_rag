import pytest

from rag.chunking.routing.router import ChunkingRouter
from rag.chunking.routing.schemas import DocumentProfile


def make_profile(**overrides):
    values = {
        "document_id": "test-document",
        "file_type": ".txt",
        "character_count": 1000,
        "token_count": 200,
        "page_count": 0,
        "pages_with_text": 0,
        "table_count": 0,
        "tables_with_rows": 0,
        "heading_count": 0,
        "paragraph_count": 10,
        "average_paragraph_tokens": 20.0,
        "has_pages": False,
        "has_tables": False,
        "has_structure": False,
        "is_short_document": True,
        "structure_confidence": 0.0,
        "metadata": {},
    }

    values.update(overrides)

    return DocumentProfile(**values)


@pytest.mark.parametrize(
    ("profile", "expected_decisions"),
    [
        (
            make_profile(
                file_type=".pdf",
                has_pages=True,
                page_count=3,
                pages_with_text=3,
            ),
            [
                ("page_aware", "text"),
            ],
        ),
        (
            make_profile(
                file_type=".pdf",
                has_pages=True,
                page_count=3,
                pages_with_text=3,
                has_tables=True,
                table_count=2,
                tables_with_rows=2,
            ),
            [
                ("page_aware", "text"),
                ("table_aware", "tables"),
            ],
        ),
        (
            make_profile(
                file_type=".xlsx",
                has_tables=True,
                table_count=2,
                tables_with_rows=2,
            ),
            [
                ("table_aware", "tables"),
            ],
        ),
        (
            make_profile(
                file_type=".csv",
                has_tables=False,
            ),
            [
                ("recursive", "text"),
            ],
        ),
        (
            make_profile(
                file_type=".md",
                has_structure=True,
                heading_count=5,
                structure_confidence=1.0,
                is_short_document=False,
            ),
            [
                ("structure_aware", "text"),
            ],
        ),
        (
            make_profile(
                file_type=".docx",
                has_structure=False,
            ),
            [
                ("recursive", "text"),
            ],
        ),
        (
            make_profile(
                file_type=".html",
                has_structure=True,
                has_tables=True,
                heading_count=3,
                table_count=1,
                tables_with_rows=1,
                structure_confidence=0.8,
            ),
            [
                ("structure_aware", "text"),
                ("table_aware", "tables"),
            ],
        ),
        (
            make_profile(
                file_type=".txt",
            ),
            [
                ("recursive", "text"),
            ],
        ),
        (
            make_profile(
                file_type=".unknown",
            ),
            [
                ("recursive", "text"),
            ],
        ),
    ],
)
def test_router_selects_expected_decisions(
    profile,
    expected_decisions,
):
    result = ChunkingRouter().route(profile)

    actual_decisions = [
        (
            decision.strategy,
            decision.content_scope,
        )
        for decision in result.decisions
    ]

    assert actual_decisions == expected_decisions


def test_router_returns_valid_result():
    profile = make_profile(
        file_type=".pdf",
        has_pages=True,
        has_tables=True,
    )

    result = ChunkingRouter().route(profile)

    assert result.document_id == profile.document_id
    assert result.router_version == ChunkingRouter.ROUTER_VERSION
    assert result.decisions

    priorities = [
        decision.priority
        for decision in result.decisions
    ]

    assert priorities == sorted(
        priorities,
        reverse=True,
    )

    identities = [
        (
            decision.strategy,
            decision.content_scope,
        )
        for decision in result.decisions
    ]

    assert len(identities) == len(set(identities))

    for decision in result.decisions:
        assert decision.strategy
        assert decision.content_scope
        assert decision.reason

def test_routes_json_to_json_aware():

    profile = make_profile(
        file_type=".json",
    )

    result = ChunkingRouter().route(
        profile
    )

    assert len(result.decisions) == 1

    decision = result.decisions[0]

    assert decision.strategy == "json_aware"
    assert decision.content_scope == "text"
    assert decision.priority == 100