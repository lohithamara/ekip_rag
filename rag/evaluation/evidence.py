from dataclasses import dataclass

from rag.evaluation.citation_faithfulness import (
    ClaimCitationPair,
)

from rag.generation.schemas import (
    ContextChunk,
)


@dataclass(frozen=True, slots=True)
class SourceEvidence:

    source_filename: str

    chunks: tuple[
        ContextChunk,
        ...,
    ]


@dataclass(frozen=True, slots=True)
class ClaimEvidence:

    claim: str

    citations: tuple[
        str,
        ...,
    ]

    source_evidence: tuple[
        SourceEvidence,
        ...,
    ]

    missing_sources: tuple[
        str,
        ...,
    ]


class SourceEvidenceResolver:

    def resolve(
        self,
        pair: ClaimCitationPair,
        context_chunks: list[
            ContextChunk
        ]
        | tuple[
            ContextChunk,
            ...,
        ],
    ) -> ClaimEvidence:

        chunks_by_source = (
            self._group_chunks_by_source(
                context_chunks
            )
        )

        source_evidence = []

        missing_sources = []

        for source in pair.citations:

            chunks = chunks_by_source.get(
                source,
                (),
            )

            if not chunks:

                missing_sources.append(
                    source
                )

                continue

            source_evidence.append(
                SourceEvidence(
                    source_filename=source,
                    chunks=chunks,
                )
            )

        return ClaimEvidence(
            claim=pair.claim,
            citations=pair.citations,
            source_evidence=tuple(
                source_evidence
            ),
            missing_sources=tuple(
                missing_sources
            ),
        )

    def resolve_all(
        self,
        pairs: list[
            ClaimCitationPair
        ]
        | tuple[
            ClaimCitationPair,
            ...,
        ],
        context_chunks: list[
            ContextChunk
        ]
        | tuple[
            ContextChunk,
            ...,
        ],
    ) -> tuple[
        ClaimEvidence,
        ...,
    ]:

        return tuple(
            self.resolve(
                pair=pair,
                context_chunks=(
                    context_chunks
                ),
            )
            for pair in pairs
        )

    @staticmethod
    def _group_chunks_by_source(
        context_chunks: list[
            ContextChunk
        ]
        | tuple[
            ContextChunk,
            ...,
        ],
    ) -> dict[
        str,
        tuple[
            ContextChunk,
            ...,
        ],
    ]:

        grouped = {}

        for chunk in context_chunks:

            source = (
                chunk.source_filename
            )

            if not source:
                continue

            grouped.setdefault(
                source,
                [],
            ).append(chunk)

        return {
            source: tuple(chunks)
            for source, chunks
            in grouped.items()
        }