from ingestion.schemas.documents import CleanDocument

from rag.chunking.chunk_factory import ChunkFactory
from rag.chunking.config import ChunkingConfig
from rag.chunking.factory import get_chunking_strategy
from rag.chunking.schemas import ChunkingResult
from rag.chunking.tokenization.factory import create_token_counter


class ChunkingService:

    def chunk_document(
        self,
        document: CleanDocument,
        config: ChunkingConfig,
    ) -> ChunkingResult:

        strategy_class = get_chunking_strategy(
            config.strategy
        )

        token_counter = create_token_counter(config)

        chunk_factory = ChunkFactory(
            config=config,
            token_counter=token_counter,
        )

        strategy = strategy_class(
            config=config,
            token_counter=token_counter,
            chunk_factory=chunk_factory,
        )

        result = strategy.chunk(document)

        self._validate_result(
            document=document,
            config=config,
            result=result,
        )

        return result

    @staticmethod
    def _validate_result(
        document: CleanDocument,
        config: ChunkingConfig,
        result: ChunkingResult,
    ) -> None:

        if result.document_id != document.metadata.document_id:
            raise ValueError(
                "Chunking result document_id mismatch."
            )

        if result.tenant_id != document.metadata.tenant_id:
            raise ValueError(
                "Chunking result tenant_id mismatch."
            )

        if result.strategy != config.strategy:
            raise ValueError(
                "Chunking result strategy mismatch."
            )

        if result.chunking_version != config.chunking_version:
            raise ValueError(
                "Chunking result version mismatch."
            )

        if result.total_chunks != len(result.chunks):
            raise ValueError(
                "Chunking result total_chunks mismatch."
            )

        for expected_index, chunk in enumerate(result.chunks):

            if chunk.chunk_index != expected_index:
                raise ValueError(
                    "Chunk indices are not sequential."
                )

            if chunk.document_id != result.document_id:
                raise ValueError(
                    "Chunk document_id mismatch."
                )

            if chunk.tenant_id != result.tenant_id:
                raise ValueError(
                    "Chunk tenant_id mismatch."
                )

            if chunk.strategy != result.strategy:
                raise ValueError(
                    "Chunk strategy mismatch."
                )

            if chunk.chunking_version != result.chunking_version:
                raise ValueError(
                    "Chunk version mismatch."
                )

            if not chunk.text.strip():
                raise ValueError(
                    "Empty chunk detected."
                )