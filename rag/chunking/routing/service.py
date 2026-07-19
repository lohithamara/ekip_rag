from dataclasses import replace
from ingestion.schemas.documents import (
    CleanDocument,
)

from rag.chunking.config import ChunkingConfig
from rag.chunking.routing.profiler import (
    DocumentProfiler,
)
from rag.chunking.routing.results import (
    RoutedChunkingResult,
)
from rag.chunking.routing.router import (
    ChunkingRouter,
    RoutingDecision,
)
from rag.chunking.schemas import Chunk
from rag.chunking.service import ChunkingService

class RoutedChunkingService:

    def __init__(
        self,
        profiler: DocumentProfiler,
        router: ChunkingRouter,
        chunking_service: ChunkingService,
    ):
        self.profiler = profiler
        self.router = router
        self.chunking_service = chunking_service

    def chunk_document(
        self,
        document: CleanDocument,
        base_config: ChunkingConfig,
    ) -> RoutedChunkingResult:

        profile = self.profiler.profile(
            document
        )

        routing_result = self.router.route(
            profile
        )

        strategy_results = []

        combined_chunks = []

        for decision in routing_result.decisions:

            config = self._create_strategy_config(
                base_config=base_config,
                strategy=decision.strategy,
            )

            result = (
                self.chunking_service.chunk_document(
                    document=document,
                    config=config,
                )
            )

            self._validate_decision_result(
                decision=decision,
                result=result,
            )

            strategy_results.append(result)

            combined_chunks.extend(
                result.chunks
            )

        self._validate_combined_chunks(
            document=document,
            chunks=combined_chunks,
        )

        return RoutedChunkingResult(
            document_id=(
                document.metadata.document_id
            ),
            tenant_id=(
                document.metadata.tenant_id
            ),
            router_version=(
                routing_result.router_version
            ),
            routing_result=routing_result,
            strategy_results=tuple(
                strategy_results
            ),
            chunks=tuple(combined_chunks),
            total_chunks=len(combined_chunks),
        )

    @staticmethod
    def _create_strategy_config(
        base_config: ChunkingConfig,
        strategy: str,
    ) -> ChunkingConfig:
        return replace(
            base_config,
            strategy=strategy,
        )

    @staticmethod
    def _validate_decision_result(
        decision: RoutingDecision,
        result,
    ) -> None:

        if result.strategy != decision.strategy:

            raise ValueError(
                "Routing decision/result strategy "
                "mismatch."
            )

        if (
            decision.content_scope == "text"
            and any(
                chunk.chunk_type != "text"
                for chunk in result.chunks
            )
        ):

            raise ValueError(
                f"Strategy {decision.strategy} "
                "produced non-text chunks for "
                "text routing scope."
            )

        if (
            decision.content_scope == "tables"
            and any(
                chunk.chunk_type != "table"
                for chunk in result.chunks
            )
        ):

            raise ValueError(
                f"Strategy {decision.strategy} "
                "produced non-table chunks for "
                "table routing scope."
            )

    @staticmethod
    def _validate_combined_chunks(
        document: CleanDocument,
        chunks: list[Chunk],
    ) -> None:

        chunk_ids = [
            chunk.chunk_id
            for chunk in chunks
        ]

        if len(chunk_ids) != len(set(chunk_ids)):

            raise ValueError(
                "Duplicate chunk IDs detected in "
                "routed chunking output."
            )

        identities = [
            (
                chunk.strategy,
                chunk.chunk_index,
            )
            for chunk in chunks
        ]

        if len(identities) != len(
            set(identities)
        ):

            raise ValueError(
                "Duplicate strategy/chunk_index "
                "identities detected."
            )

        for chunk in chunks:

            if (
                chunk.document_id
                != document.metadata.document_id
            ):

                raise ValueError(
                    "Combined chunk document_id "
                    "mismatch."
                )

            if (
                chunk.tenant_id
                != document.metadata.tenant_id
            ):

                raise ValueError(
                    "Combined chunk tenant_id "
                    "mismatch."
                )