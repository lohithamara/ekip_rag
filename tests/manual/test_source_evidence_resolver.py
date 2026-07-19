from rag.evaluation.citation_faithfulness import (
    ClaimCitationPair,
)

from rag.evaluation.evidence import (
    SourceEvidenceResolver,
)

from rag.generation.schemas import (
    ContextChunk,
)


def make_chunk(
    chunk_id: str,
    source_filename: str,
    text: str,
) -> ContextChunk:

    return ContextChunk(
        chunk_id=chunk_id,
        document_id=(
            f"document_{chunk_id}"
        ),
        source_filename=(
            source_filename
        ),
        text=text,
        token_count=len(
            text.split()
        ),
        metadata={},
    )


def main():

    resolver = (
        SourceEvidenceResolver()
    )

    context_chunks = (
        make_chunk(
            chunk_id="chunk_1",
            source_filename="a.pdf",
            text="First evidence chunk.",
        ),
        make_chunk(
            chunk_id="chunk_2",
            source_filename="b.pdf",
            text="Second evidence chunk.",
        ),
        make_chunk(
            chunk_id="chunk_3",
            source_filename="a.pdf",
            text="Additional evidence.",
        ),
    )

    print()
    print("=" * 80)
    print("SOURCE EVIDENCE RESOLVER TEST")
    print("=" * 80)
    print()

    # ------------------------------------------------------------
    # TEST 1
    # Single source
    # ------------------------------------------------------------

    single = resolver.resolve(
        pair=ClaimCitationPair(
            claim="Claim A.",
            citations=(
                "a.pdf",
            ),
        ),
        context_chunks=context_chunks,
    )

    assert single.claim == (
        "Claim A."
    )

    assert single.citations == (
        "a.pdf",
    )

    assert len(
        single.source_evidence
    ) == 1

    assert (
        single
        .source_evidence[0]
        .source_filename
        == "a.pdf"
    )

    assert len(
        single
        .source_evidence[0]
        .chunks
    ) == 2

    assert not single.missing_sources

    print(
        "Single source resolution     : PASS"
    )

    # ------------------------------------------------------------
    # TEST 2
    # Multiple sources
    # ------------------------------------------------------------

    multiple = resolver.resolve(
        pair=ClaimCitationPair(
            claim="Claim B.",
            citations=(
                "a.pdf",
                "b.pdf",
            ),
        ),
        context_chunks=context_chunks,
    )

    assert len(
        multiple.source_evidence
    ) == 2

    assert tuple(
        evidence.source_filename
        for evidence
        in multiple.source_evidence
    ) == (
        "a.pdf",
        "b.pdf",
    )

    assert not multiple.missing_sources

    print(
        "Multiple source resolution   : PASS"
    )

    # ------------------------------------------------------------
    # TEST 3
    # Missing source
    # ------------------------------------------------------------

    missing = resolver.resolve(
        pair=ClaimCitationPair(
            claim="Claim C.",
            citations=(
                "missing.pdf",
            ),
        ),
        context_chunks=context_chunks,
    )

    assert not (
        missing.source_evidence
    )

    assert (
        missing.missing_sources
        == (
            "missing.pdf",
        )
    )

    print(
        "Missing source detection     : PASS"
    )

    # ------------------------------------------------------------
    # TEST 4
    # Mixed present and missing sources
    # ------------------------------------------------------------

    mixed = resolver.resolve(
        pair=ClaimCitationPair(
            claim="Claim D.",
            citations=(
                "a.pdf",
                "missing.pdf",
            ),
        ),
        context_chunks=context_chunks,
    )

    assert len(
        mixed.source_evidence
    ) == 1

    assert (
        mixed
        .source_evidence[0]
        .source_filename
        == "a.pdf"
    )

    assert (
        mixed.missing_sources
        == (
            "missing.pdf",
        )
    )

    print(
        "Mixed source resolution      : PASS"
    )

    # ------------------------------------------------------------
    # TEST 5
    # Empty citation collection
    # ------------------------------------------------------------

    empty = resolver.resolve(
        pair=ClaimCitationPair(
            claim="Claim E.",
            citations=(),
        ),
        context_chunks=context_chunks,
    )

    assert not empty.source_evidence

    assert not empty.missing_sources

    print(
        "Empty citation handling      : PASS"
    )

    # ------------------------------------------------------------
    # TEST 6
    # Batch resolution
    # ------------------------------------------------------------

    batch = resolver.resolve_all(
        pairs=(
            ClaimCitationPair(
                claim="Claim A.",
                citations=(
                    "a.pdf",
                ),
            ),
            ClaimCitationPair(
                claim="Claim B.",
                citations=(
                    "b.pdf",
                ),
            ),
        ),
        context_chunks=context_chunks,
    )

    assert len(batch) == 2

    assert (
        batch[0]
        .source_evidence[0]
        .source_filename
        == "a.pdf"
    )

    assert (
        batch[1]
        .source_evidence[0]
        .source_filename
        == "b.pdf"
    )

    print(
        "Batch resolution             : PASS"
    )

    print()
    print("=" * 80)
    print("FINAL STATUS: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()